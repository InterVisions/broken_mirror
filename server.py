"""
InterVisions So-B-IT Broken Mirror — Interactive CLIP Bias Audit Tool
Backend server: FastAPI + WebSocket + PyTorch/CLIP inference

Usage:
    python server.py --model ViT-B/32 --port 8765 --max-labels 20 --device cuda
"""

import argparse
import asyncio
import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from sklearn.manifold import TSNE

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sobit-mirror")

# ── Global state (populated in startup) ─────────────────────────────────────
MODEL = None
PREPROCESS = None
TOKENIZER = None
DEVICE = "cpu"
TAXONOMY = {}
TEXT_EMBEDDINGS = None       # (N, D) tensor — one per So-B-IT term
TEXT_LABELS = []             # list of dicts: {word, category, color}
FAIRFACE_EMBEDDINGS = {}     # key -> (M, D) tensor for zero-shot classification
TSNE_COORDS = None           # (N, 2) numpy array — precomputed 2-D layout for terms
MAX_LABELS = 20
CLIP_BACKEND = "open_clip"   # or "openai_clip"
args = None                  # parsed CLI args (set in main)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIP Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_clip_model(model_name: str, device: str):
    """
    Try open_clip first (wider model zoo), fall back to openai clip.
    Returns (model, preprocess, tokenizer, backend_name).
    """
    global CLIP_BACKEND

    # ── Try open_clip ────────────────────────────────────────────────────────
    try:
        import open_clip
        oc_map = {
            "ViT-B/32":  ("ViT-B-32",  "openai"),
            "ViT-B/16":  ("ViT-B-16",  "openai"),
            "ViT-L/14":  ("ViT-L-14",  "openai"),
            "ViT-H/14":  ("ViT-H-14",  "laion2b_s32b_b79k"),
        }
        if model_name in oc_map:
            arch, pretrained = oc_map[model_name]
        else:
            parts = model_name.split(":")
            arch = parts[0]
            pretrained = parts[1] if len(parts) > 1 else "openai"

        log.info(f"Loading open_clip model {arch} (pretrained={pretrained}) …")
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(arch)
        model.eval()
        CLIP_BACKEND = "open_clip"
        log.info("✓ Loaded via open_clip")
        return model, preprocess, tokenizer, "open_clip"
    except Exception as e:
        log.warning(f"open_clip failed ({e}), trying openai clip …")

    # ── Fallback: openai clip ────────────────────────────────────────────────
    import clip as openai_clip
    log.info(f"Loading openai clip model {model_name} …")
    model, preprocess = openai_clip.load(model_name, device=device)
    model.eval()

    def tokenizer(texts):
        return openai_clip.tokenize(texts, truncate=True)

    CLIP_BACKEND = "openai_clip"
    log.info("✓ Loaded via openai clip")
    return model, preprocess, tokenizer, "openai_clip"


# ═══════════════════════════════════════════════════════════════════════════════
#  Text embedding helpers
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_texts(texts: list[str]) -> torch.Tensor:
    """Encode a list of text prompts → (N, D) L2-normalised embeddings."""
    tokens = TOKENIZER(texts)
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.to(DEVICE)
    else:
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    feats = MODEL.encode_text(tokens)
    feats = F.normalize(feats, dim=-1)
    return feats


@torch.no_grad()
def encode_image_tensor(img: Image.Image) -> torch.Tensor:
    """Encode a single PIL image → (1, D) L2-normalised embedding."""
    tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    feats = MODEL.encode_image(tensor)
    feats = F.normalize(feats, dim=-1)
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  Precomputation at startup
# ═══════════════════════════════════════════════════════════════════════════════

def make_fairface_prompt(attr: str, value: str) -> str:
    """Generate a natural-language CLIP prompt for a FairFace attribute/value pair."""
    if attr == "age":
        return f"a photo of a {value}"
    elif attr == "gender":
        return f"a photo of a {value.lower()} person"
    elif attr == "race":
        return f"a photo of a {value} person"
    else:
        return f"a photo of a {value} person"


def make_prompt(word: str, category: str) -> str:
    """Generate a CLIP prompt for a taxonomy word."""
    occupations_like = {"Occupation", "Political"}
    if category in occupations_like:
        return f"a photo of a {word}"
    else:
        return f"a photo of a {word} person"


def build_text_embeddings(taxonomy: dict):
    """
    Build text embeddings for every So-B-IT term and FairFace labels.
    Returns text_embeddings (Tensor), text_labels (list of dicts),
    fairface_embeddings (dict of Tensors).
    """
    all_prompts = []
    labels = []

    for cat_name, cat_data in taxonomy["categories"].items():
        color = cat_data["color"]
        for word in cat_data["words"]:
            prompt = make_prompt(word, cat_name)
            all_prompts.append(prompt)
            labels.append({"word": word, "category": cat_name, "color": color})

    log.info(f"Encoding {len(all_prompts)} So-B-IT prompts …")
    text_emb = encode_texts(all_prompts)
    log.info(f"✓ Text embeddings shape: {text_emb.shape}")

    ff = taxonomy.get("fairface_labels", {})
    ff_emb = {}
    for attr, values in ff.items():
        prompts = [make_fairface_prompt(attr, v) for v in values]
        ff_emb[attr] = {"labels": values, "embeddings": encode_texts(prompts)}
        log.info(f"  FairFace/{attr}: {len(values)} classes")

    return text_emb, labels, ff_emb


def compute_tsne(embeddings: torch.Tensor, perplexity: int = 30, seed: int = 42) -> np.ndarray:
    """Run t-SNE on the text embeddings → (N, 2) coordinates."""
    log.info("Computing t-SNE layout …")
    X = embeddings.cpu().numpy().astype(np.float32)
    n = X.shape[0]
    perp = min(perplexity, max(5, n // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, max_iter=1000)
    coords = tsne.fit_transform(X)
    coords -= coords.mean(axis=0)
    scale = np.abs(coords).max()
    if scale > 0:
        coords /= scale
    log.info(f"✓ t-SNE done, shape {coords.shape}")
    return coords


def interpolate_tsne_position(new_emb: torch.Tensor) -> np.ndarray:
    """
    Project a new embedding into the existing t-SNE space via weighted
    nearest-neighbour interpolation. Same logic used for the user dot.
    """
    sims = (new_emb @ TEXT_EMBEDDINGS.T).squeeze(0).cpu().numpy()
    k = min(10, len(sims))
    top_idx = np.argsort(sims)[::-1][:k]
    weights = np.maximum(sims[top_idx], 0)
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return np.array([0.0, 0.0])
    weights /= w_sum
    return (TSNE_COORDS[top_idx] * weights[:, None]).sum(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference on a single frame
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def process_frame(img: Image.Image, top_k: int = 15) -> dict:
    t0 = time.time()

    img_emb = encode_image_tensor(img)  # (1, D)
    sims = (img_emb @ TEXT_EMBEDDINGS.T).squeeze(0).cpu().numpy()  # (N,)

    top_idx = np.argsort(sims)[::-1][:top_k]
    top_terms = []
    for idx in top_idx:
        idx = int(idx)
        entry = TEXT_LABELS[idx]
        top_terms.append({
            "word": entry["word"],
            "category": entry["category"],
            "color": entry["color"],
            "similarity": round(float(sims[idx]), 4),
        })

    fairface = {}
    for attr, data in FAIRFACE_EMBEDDINGS.items():
        logits = (img_emb @ data["embeddings"].T).squeeze(0)
        probs = F.softmax(logits * 100, dim=0).cpu().numpy()
        fairface[attr] = {
            label: round(float(p), 4)
            for label, p in zip(data["labels"], probs)
        }

    user_tsne = project_to_tsne(img_emb)
    elapsed = time.time() - t0

    return {
        "top_terms": top_terms,
        "fairface": fairface,
        "user_tsne": user_tsne,
        "inference_ms": round(elapsed * 1000, 1),
    }


def project_to_tsne(img_emb: torch.Tensor) -> list:
    sims = (img_emb @ TEXT_EMBEDDINGS.T).squeeze(0).cpu().numpy()
    k = min(10, len(sims))
    top_idx = np.argsort(sims)[::-1][:k]
    weights = sims[top_idx]
    weights = np.maximum(weights, 0)
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return [0.0, 0.0]
    weights /= w_sum
    pos = (TSNE_COORDS[top_idx] * weights[:, None]).sum(axis=0)
    return [round(float(pos[0]), 4), round(float(pos[1]), 4)]


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI application
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="InterVisions - So-B-IT Broken Mirror")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/init")
async def api_init():
    """Return taxonomy metadata, t-SNE coords, and config for the frontend."""
    terms = []
    for i, label in enumerate(TEXT_LABELS):
        terms.append({
            "word": label["word"],
            "category": label["category"],
            "color": label["color"],
            "x": round(float(TSNE_COORDS[i, 0]), 4),
            "y": round(float(TSNE_COORDS[i, 1]), 4),
        })

    categories = {}
    for cat_name, cat_data in TAXONOMY["categories"].items():
        categories[cat_name] = {"color": cat_data["color"], "count": len(cat_data["words"])}

    return {
        "terms": terms,
        "categories": categories,
        "fairface_labels": TAXONOMY.get("fairface_labels", {}),
        "max_labels": MAX_LABELS,
        "model": args.model,
    }


# ── Custom word endpoint ──────────────────────────────────────────────────────

CUSTOM_COLOR = "#FFFFFF"

class AddWordRequest(BaseModel):
    word: str
    category: str = "Custom"
    color: str = CUSTOM_COLOR


@app.post("/api/add_word")
async def add_word(req: AddWordRequest):
    """
    Embed a new word live and append it to the global state.
    The t-SNE position is interpolated from the nearest existing terms
    so the existing layout is not disturbed.
    """
    global TEXT_EMBEDDINGS, TEXT_LABELS, TSNE_COORDS

    word = req.word.strip().lower()
    if not word:
        return {"status": "error", "message": "Empty word"}

    # Duplicate check (same word + same category)
    if any(l["word"] == word and l["category"] == req.category for l in TEXT_LABELS):
        return {"status": "duplicate", "word": word}

    # Embed the new word using the same prompt template
    prompt = make_prompt(word, req.category)
    log.info(f"Embedding custom word: '{word}' (prompt: '{prompt}')")
    new_emb = encode_texts([prompt])  # (1, D)

    # Interpolate position in existing t-SNE space
    tsne_pos = interpolate_tsne_position(new_emb)

    # Append to global tensors / lists — thread-safe enough for a workshop context
    TEXT_EMBEDDINGS = torch.cat([TEXT_EMBEDDINGS, new_emb], dim=0)
    TEXT_LABELS.append({"word": word, "category": req.category, "color": req.color})
    TSNE_COORDS = np.vstack([TSNE_COORDS, tsne_pos])

    # Also register the word in the in-memory taxonomy so /api/init stays consistent
    if req.category in TAXONOMY["categories"]:
        if word not in TAXONOMY["categories"][req.category]["words"]:
            TAXONOMY["categories"][req.category]["words"].append(word)
    else:
        TAXONOMY["categories"][req.category] = {"color": req.color, "words": [word]}

    log.info(f"✓ Added '{word}' at t-SNE ({tsne_pos[0]:.3f}, {tsne_pos[1]:.3f})")

    return {
        "status": "ok",
        "word": word,
        "category": req.category,
        "color": req.color,
        "x": round(float(tsne_pos[0]), 4),
        "y": round(float(tsne_pos[1]), 4),
    }


@app.delete("/api/custom_words")
async def clear_custom_words():
    """Remove all custom words from the session (does not affect So-B-IT terms)."""
    global TEXT_EMBEDDINGS, TEXT_LABELS, TSNE_COORDS

    keep = [i for i, l in enumerate(TEXT_LABELS) if l["category"] != "Custom"]
    TEXT_EMBEDDINGS = TEXT_EMBEDDINGS[keep]
    TEXT_LABELS = [TEXT_LABELS[i] for i in keep]
    TSNE_COORDS = TSNE_COORDS[keep]

    if "Custom" in TAXONOMY["categories"]:
        TAXONOMY["categories"]["Custom"]["words"] = []

    log.info("Cleared all custom words")
    return {"status": "ok", "removed": len(TEXT_LABELS) - len(keep)}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                img_b64 = msg["data"]
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")

                top_k = msg.get("top_k", MAX_LABELS)
                result = process_frame(img, top_k=top_k)
                result["type"] = "result"
                await ws.send_text(json.dumps(result))

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="InterVisions So-B-IT Broken Mirror — CLIP Bias Audit Tool")
    p.add_argument("--model", default="ViT-B/32")
    p.add_argument("--device", default="auto")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--max-labels", type=int, default=20)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--taxonomy", default=None)
    p.add_argument("--tsne-perplexity", type=int, default=30)
    return p.parse_args()


def main():
    global MODEL, PREPROCESS, TOKENIZER, DEVICE, TAXONOMY
    global TEXT_EMBEDDINGS, TEXT_LABELS, FAIRFACE_EMBEDDINGS, TSNE_COORDS
    global MAX_LABELS, args

    args = parse_args()
    MAX_LABELS = args.max_labels

    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    log.info(f"Device: {DEVICE}")

    MODEL, PREPROCESS, TOKENIZER, _ = load_clip_model(args.model, DEVICE)

    tax_path = args.taxonomy or str(Path(__file__).parent / "config" / "sobit_taxonomy.json")
    with open(tax_path) as f:
        TAXONOMY = json.load(f)
    log.info(f"Loaded taxonomy from {tax_path}")

    # Register Custom category so it's available from the start
    if "Custom" not in TAXONOMY["categories"]:
        TAXONOMY["categories"]["Custom"] = {"color": CUSTOM_COLOR, "words": []}

    TEXT_EMBEDDINGS, TEXT_LABELS, FAIRFACE_EMBEDDINGS = build_text_embeddings(TAXONOMY)
    TSNE_COORDS = compute_tsne(TEXT_EMBEDDINGS, perplexity=args.tsne_perplexity)

    import uvicorn
    log.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
