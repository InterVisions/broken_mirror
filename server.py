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
        # Map friendly names to open_clip identifiers
        oc_map = {
            "ViT-B/32":  ("ViT-B-32",  "openai"),
            "ViT-B/16":  ("ViT-B-16",  "openai"),
            "ViT-L/14":  ("ViT-L-14",  "openai"),
            "ViT-H/14":  ("ViT-H-14",  "laion2b_s32b_b79k"),
        }
        if model_name in oc_map:
            arch, pretrained = oc_map[model_name]
        else:
            # Allow direct open_clip spec like "ViT-B-32:laion2b"
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

    if CLIP_BACKEND == "open_clip":
        feats = MODEL.encode_text(tokens)
    else:
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
        # Age labels are already full noun phrases (e.g. "young adult", "elderly person")
        # so we just wrap them directly — no trailing "person"
        return f"a photo of a {value}"
    elif attr == "gender":
        return f"a photo of a {value.lower()} person"
    elif attr == "race":
        return f"a photo of a {value} person"
    else:
        return f"a photo of a {value} person"

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

    # FairFace zero-shot labels
    ff = taxonomy.get("fairface_labels", {})
    ff_emb = {}
    for attr, values in ff.items():
        prompts = [make_fairface_prompt(attr, v) for v in values]
        ff_emb[attr] = {"labels": values, "embeddings": encode_texts(prompts)}
        log.info(f"  FairFace/{attr}: {len(values)} classes")

    return text_emb, labels, ff_emb


def make_prompt(word: str, category: str) -> str:
    """Generate a CLIP prompt for a taxonomy word."""
    # Use templates from the paper
    occupations_like = {"Occupation", "Political"}
    if category in occupations_like:
        return f"a photo of a {word}"
    else:
        return f"a photo of a {word} person"


def compute_tsne(embeddings: torch.Tensor, perplexity: int = 30, seed: int = 42) -> np.ndarray:
    """Run t-SNE on the text embeddings → (N, 2) coordinates."""
    log.info("Computing t-SNE layout …")
    X = embeddings.cpu().numpy().astype(np.float32)
    n = X.shape[0]
    perp = min(perplexity, max(5, n // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, max_iter=1000)
    coords = tsne.fit_transform(X)
    # Normalise to [-1, 1] range for the frontend
    coords -= coords.mean(axis=0)
    scale = np.abs(coords).max()
    if scale > 0:
        coords /= scale
    log.info(f"✓ t-SNE done, shape {coords.shape}")
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference on a single frame
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def process_frame(img: Image.Image, top_k: int = 15) -> dict:
    """
    Given a webcam frame (PIL), return:
    - top_k closest So-B-IT terms with similarity scores
    - zero-shot FairFace classification probabilities
    - user embedding projected to t-SNE space
    """
    t0 = time.time()

    # Encode image
    img_emb = encode_image_tensor(img)  # (1, D)

    # Cosine similarities to all So-B-IT terms
    sims = (img_emb @ TEXT_EMBEDDINGS.T).squeeze(0).cpu().numpy()  # (N,)

    # Top-k
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

    # Zero-shot FairFace classification
    fairface = {}
    for attr, data in FAIRFACE_EMBEDDINGS.items():
        logits = (img_emb @ data["embeddings"].T).squeeze(0)  # (C,)
        probs = F.softmax(logits * 100, dim=0).cpu().numpy()  # temperature-scaled
        fairface[attr] = {
            label: round(float(p), 4)
            for label, p in zip(data["labels"], probs)
        }

    # Project user embedding into t-SNE space (approximate via nearest-neighbor interpolation)
    user_tsne = project_to_tsne(img_emb)

    elapsed = time.time() - t0

    return {
        "top_terms": top_terms,
        "fairface": fairface,
        "user_tsne": user_tsne,
        "inference_ms": round(elapsed * 1000, 1),
    }


def project_to_tsne(img_emb: torch.Tensor) -> list:
    """
    Project user embedding into the precomputed t-SNE space.
    Uses weighted average of K nearest text embedding positions.
    """
    sims = (img_emb @ TEXT_EMBEDDINGS.T).squeeze(0).cpu().numpy()
    # Use top-10 neighbours, weighted by similarity
    k = min(10, len(sims))
    top_idx = np.argsort(sims)[::-1][:k]
    weights = sims[top_idx]
    weights = np.maximum(weights, 0)  # clip negatives
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

# Serve static files (frontend)
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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                # Decode base64 JPEG
                img_b64 = msg["data"]
                # Strip data URL prefix if present
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
    p.add_argument("--model", default="ViT-B/32",
                   help="CLIP model name (default: ViT-B/32)")
    p.add_argument("--device", default="auto",
                   help="Device: cuda, cpu, or auto (default: auto)")
    p.add_argument("--port", type=int, default=8765,
                   help="Server port (default: 8765)")
    p.add_argument("--host", default="0.0.0.0",
                   help="Server host (default: 0.0.0.0)")
    p.add_argument("--max-labels", type=int, default=20,
                   help="Max number of labels to show in the t-SNE plot (default: 20)")
    p.add_argument("--top-k", type=int, default=15,
                   help="Default number of top terms returned (default: 15)")
    p.add_argument("--taxonomy", default=None,
                   help="Path to custom taxonomy JSON (default: config/sobit_taxonomy.json)")
    p.add_argument("--tsne-perplexity", type=int, default=30,
                   help="t-SNE perplexity (default: 30)")
    return p.parse_args()


def main():
    global MODEL, PREPROCESS, TOKENIZER, DEVICE, TAXONOMY
    global TEXT_EMBEDDINGS, TEXT_LABELS, FAIRFACE_EMBEDDINGS, TSNE_COORDS
    global MAX_LABELS, args

    args = parse_args()
    MAX_LABELS = args.max_labels

    # Device
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    log.info(f"Device: {DEVICE}")

    # Load model
    MODEL, PREPROCESS, TOKENIZER, _ = load_clip_model(args.model, DEVICE)

    # Load taxonomy
    tax_path = args.taxonomy or str(Path(__file__).parent / "config" / "sobit_taxonomy.json")
    with open(tax_path) as f:
        TAXONOMY = json.load(f)
    log.info(f"Loaded taxonomy from {tax_path}")

    # Build text embeddings
    TEXT_EMBEDDINGS, TEXT_LABELS, FAIRFACE_EMBEDDINGS = build_text_embeddings(TAXONOMY)

    # Precompute t-SNE
    TSNE_COORDS = compute_tsne(TEXT_EMBEDDINGS, perplexity=args.tsne_perplexity)

    # Run server
    import uvicorn
    log.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
