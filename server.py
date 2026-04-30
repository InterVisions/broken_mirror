"""
InterVisions So-B-IT Broken Mirror — Interactive CLIP Bias Audit Tool
Backend server: FastAPI + WebSocket + PyTorch/CLIP inference

Usage:
    python server.py --model ViT-B/32 --port 8765 --max-labels 20 --device cuda
"""

import argparse
import asyncio
import base64
import csv
import json
import logging
import time
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
import umap

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sobit-mirror")

# ── Global state ─────────────────────────────────────────────────────────────
MODEL = None
PREPROCESS = None
TOKENIZER = None
DEVICE = "cpu"
TAXONOMY = {}
FAIRFACE_EMBEDDINGS = {}   # always English prompts — language-independent

# Per-language state: LANG_STATE[lang] = {text_embeddings, text_labels, tsne_coords,
#                                         umap_model, umap_mean, umap_scale}
LANG_STATE = {}
SUPPORTED_LANGS = ['en']   # extended to ['en','es'] at startup if translations present

MAX_LABELS = 20
CLIP_BACKEND = "open_clip"
args = None

# ── Session / CSV logging ─────────────────────────────────────────────────────
SESSION_NAME = None
CSV_PATH = None
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def _auto_open_session(name: str = "default"):
    global SESSION_NAME, CSV_PATH
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{safe}_{timestamp}.csv"
    CSV_PATH = LOGS_DIR / filename
    SESSION_NAME = name
    LOGS_DIR.mkdir(exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
    log.info(f"Auto-opened CSV log: {CSV_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLIP Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_clip_model(model_name: str, device: str):
    global CLIP_BACKEND

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
    tensor = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    feats = MODEL.encode_image(tensor)
    feats = F.normalize(feats, dim=-1)
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt builders
# ═══════════════════════════════════════════════════════════════════════════════

def make_prompt(word: str, category: str, lang: str = 'en') -> str:
    if lang == 'es':
        return f"una fotografía de {word}"
    occupations_like = {"Occupation", "Political"}
    if category in occupations_like:
        return f"a photo of a {word}"
    return f"a photo of a {word} person"


def make_fairface_prompt(attr: str, value: str) -> str:
    if attr == "age":
        return f"a photo of a {value}"
    return f"a photo of a {value.lower() if attr == 'gender' else value} person"


def get_translated_word(taxonomy: dict, en_word: str, lang: str) -> str:
    """Return the translated word for a given language, falling back to English."""
    if lang == 'en':
        return en_word
    return (taxonomy
            .get('translations', {})
            .get(lang, {})
            .get('words', {})
            .get(en_word, en_word))


# ═══════════════════════════════════════════════════════════════════════════════
#  Precomputation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_text_embeddings_for_lang(taxonomy: dict, lang: str):
    """Encode all So-B-IT terms in the given language. Returns (tensor, labels)."""
    all_prompts = []
    labels = []
    for cat_name, cat_data in taxonomy["categories"].items():
        color = cat_data["color"]
        for en_word in cat_data["words"]:
            word = get_translated_word(taxonomy, en_word, lang)
            prompt = make_prompt(word, cat_name, lang)
            all_prompts.append(prompt)
            # always store the translated display word, but keep en_word for CSV/dedup
            labels.append({
                "word":    word,
                "en_word": en_word,
                "category": cat_name,
                "color":   color,
            })

    log.info(f"[{lang}] Encoding {len(all_prompts)} prompts …")
    text_emb = encode_texts(all_prompts)
    log.info(f"[{lang}] ✓ Embeddings shape: {text_emb.shape}")
    return text_emb, labels


def build_fairface_embeddings(taxonomy: dict):
    ff = taxonomy.get("fairface_labels", {})
    ff_emb = {}
    for attr, values in ff.items():
        prompts = [make_fairface_prompt(attr, v) for v in values]
        ff_emb[attr] = {"labels": values, "embeddings": encode_texts(prompts)}
        log.info(f"  FairFace/{attr}: {len(values)} classes")
    return ff_emb


def compute_umap_layout(embeddings: torch.Tensor, n_neighbors: int = 15, seed: int = 42):
    """Fit UMAP and return (coords, model, mean, scale) — no global side-effects."""
    log.info("Computing UMAP layout …")
    X = embeddings.cpu().numpy().astype(np.float32)
    model = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
        metric='cosine', random_state=seed, n_jobs=1
    )
    coords = model.fit_transform(X)
    mean = coords.mean(axis=0)
    coords = coords - mean
    scale = np.abs(coords).max()
    if scale > 0:
        coords /= scale
    log.info(f"✓ UMAP done, shape {coords.shape}")
    return coords, model, mean, scale


def init_lang_state(taxonomy: dict, lang: str, n_neighbors: int):
    """Build embeddings + UMAP for one language and store in LANG_STATE[lang]."""
    text_emb, labels = build_text_embeddings_for_lang(taxonomy, lang)
    coords, umap_model, umap_mean, umap_scale = compute_umap_layout(text_emb, n_neighbors)
    LANG_STATE[lang] = {
        'text_embeddings': text_emb,
        'text_labels':     labels,
        'tsne_coords':     coords,
        'umap_model':      umap_model,
        'umap_mean':       umap_mean,
        'umap_scale':      umap_scale,
    }
    log.info(f"[{lang}] ✓ Language state ready")


def project_new_term(new_emb: torch.Tensor, lang: str = 'en') -> np.ndarray:
    st = LANG_STATE[lang]
    X = new_emb.cpu().numpy().astype(np.float32)
    coords = st['umap_model'].transform(X)[0]
    coords = (coords - st['umap_mean']) / st['umap_scale']
    return coords


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference on a single frame
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def process_frame(img: Image.Image, top_k: int = 15,
                  categories: set = None, lang: str = 'en') -> dict:
    if lang not in LANG_STATE:
        lang = 'en'
    st = LANG_STATE[lang]

    t0 = time.time()
    img_emb = encode_image_tensor(img)
    sims = (img_emb @ st['text_embeddings'].T).squeeze(0).cpu().numpy()

    if categories:
        candidate_idx = [i for i, l in enumerate(st['text_labels'])
                         if l["category"] in categories]
    else:
        candidate_idx = list(range(len(st['text_labels'])))

    candidate_idx = np.array(candidate_idx)
    top_idx = candidate_idx[np.argsort(sims[candidate_idx])[::-1][:top_k]]

    top_terms = []
    for idx in top_idx:
        idx = int(idx)
        entry = st['text_labels'][idx]
        top_terms.append({
            "word":     entry["word"],
            "en_word":  entry["en_word"],
            "category": entry["category"],
            "color":    entry["color"],
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

    user_tsne = project_to_tsne(img_emb, categories=categories, lang=lang)
    elapsed = time.time() - t0

    return {
        "top_terms":    top_terms,
        "fairface":     fairface,
        "user_tsne":    user_tsne,
        "inference_ms": round(elapsed * 1000, 1),
    }


def project_to_tsne(img_emb: torch.Tensor, categories: set = None,
                    lang: str = 'en') -> list:
    if lang not in LANG_STATE:
        lang = 'en'
    st = LANG_STATE[lang]
    mode = args.projection if args else 'top1'

    sims = (img_emb @ st['text_embeddings'].T).squeeze(0).cpu().numpy()
    if categories:
        candidate_idx = np.array([i for i, l in enumerate(st['text_labels'])
                                   if l["category"] in categories])
    else:
        candidate_idx = np.arange(len(sims))

    if len(candidate_idx) == 0:
        return [0.0, 0.0]

    if mode == 'transform':
        X = img_emb.cpu().numpy().astype(np.float32)
        coords = st['umap_model'].transform(X)[0]
        coords = (coords - st['umap_mean']) / st['umap_scale']
        return [round(float(coords[0]), 4), round(float(coords[1]), 4)]

    if mode == 'top1':
        best = candidate_idx[np.argmax(sims[candidate_idx])]
        pos = st['tsne_coords'][int(best)]
        return [round(float(pos[0]), 4), round(float(pos[1]), 4)]

    k = min(10, len(candidate_idx))
    top_idx = candidate_idx[np.argsort(sims[candidate_idx])[::-1][:k]]

    if mode == 'softmax':
        tau = args.projection_tau if args else 0.1
        logits = sims[top_idx] / tau
        logits -= logits.max()
        weights = np.exp(logits)
    else:
        weights = np.maximum(sims[top_idx], 0)

    w_sum = weights.sum()
    if w_sum < 1e-8:
        return [0.0, 0.0]
    weights /= w_sum
    pos = (st['tsne_coords'][top_idx] * weights[:, None]).sum(axis=0)
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


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(str(STATIC_DIR / "logo.png"))


@app.get("/api/init")
async def api_init(lang: str = 'en'):
    """Return taxonomy metadata, UMAP coords, and translations for the frontend."""
    if lang not in LANG_STATE:
        lang = 'en'
    st = LANG_STATE[lang]

    terms = []
    for i, label in enumerate(st['text_labels']):
        terms.append({
            "word":     label["word"],
            "en_word":  label["en_word"],
            "category": label["category"],
            "color":    label["color"],
            "x": round(float(st['tsne_coords'][i, 0]), 4),
            "y": round(float(st['tsne_coords'][i, 1]), 4),
        })

    categories = {}
    for cat_name, cat_data in TAXONOMY["categories"].items():
        categories[cat_name] = {"color": cat_data["color"], "count": len(cat_data["words"])}

    # Send category label translations so the frontend doesn't need them hardcoded
    cat_labels = (TAXONOMY
                  .get('translations', {})
                  .get(lang, {})
                  .get('categories', {}))

    return {
        "terms":           terms,
        "categories":      categories,
        "category_labels": cat_labels,
        "fairface_labels": TAXONOMY.get("fairface_labels", {}),
        "max_labels":      MAX_LABELS,
        "model":           args.model,
        "lang":            lang,
        "supported_langs": SUPPORTED_LANGS,
    }


# ── Session / CSV helpers ─────────────────────────────────────────────────────

CSV_COLUMNS = ["timestamp", "session", "word", "en_word", "category", "lang", "tsne_x", "tsne_y"]


def open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
    log.info(f"CSV log opened at {path}")


def append_csv(row: dict):
    if CSV_PATH is None:
        return
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


class StartSessionRequest(BaseModel):
    name: str = ""


@app.post("/api/session/start")
async def session_start(req: StartSessionRequest):
    global SESSION_NAME, CSV_PATH
    raw = req.name.strip() or "unnamed"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{safe}_{timestamp}.csv"
    CSV_PATH = LOGS_DIR / filename
    SESSION_NAME = raw
    open_csv(CSV_PATH)
    return {"status": "ok", "session": SESSION_NAME, "file": filename}


@app.get("/api/session")
async def session_info():
    return {
        "active":  SESSION_NAME is not None,
        "session": SESSION_NAME,
        "file":    CSV_PATH.name if CSV_PATH else None,
    }


from fastapi.responses import StreamingResponse

@app.get("/api/export")
async def export_csv():
    if CSV_PATH is None or not CSV_PATH.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No active session or file not found.")
    content = CSV_PATH.read_text(encoding="utf-8")
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{CSV_PATH.name}"'},
    )


# ── Custom word endpoint ──────────────────────────────────────────────────────

CUSTOM_COLOR = "#FFFFFF"


class AddWordRequest(BaseModel):
    word: str
    category: str = "Custom"
    color: str = CUSTOM_COLOR
    lang: str = "en"


@app.post("/api/add_word")
async def add_word(req: AddWordRequest):
    word = req.word.strip().lower()
    lang = req.lang if req.lang in LANG_STATE else 'en'

    if not word:
        return {"status": "error", "message": "Empty word"}

    st = LANG_STATE[lang]
    if any(l["en_word"] == word and l["category"] == req.category
           for l in st['text_labels']):
        return {"status": "duplicate", "word": word}

    prompt = make_prompt(word, req.category, lang)
    log.info(f"[{lang}] Embedding custom word: '{word}' (prompt: '{prompt}')")
    new_emb = encode_texts([prompt])

    tsne_pos = project_new_term(new_emb, lang)

    st['text_embeddings'] = torch.cat([st['text_embeddings'], new_emb], dim=0)
    st['text_labels'].append({
        "word":    word,
        "en_word": word,
        "category": req.category,
        "color":   req.color,
    })
    st['tsne_coords'] = np.vstack([st['tsne_coords'], tsne_pos])

    if req.category in TAXONOMY["categories"]:
        if word not in TAXONOMY["categories"][req.category]["words"]:
            TAXONOMY["categories"][req.category]["words"].append(word)
    else:
        TAXONOMY["categories"][req.category] = {"color": req.color, "words": [word]}

    log.info(f"[{lang}] ✓ Added '{word}' at ({tsne_pos[0]:.3f}, {tsne_pos[1]:.3f})")

    append_csv({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session":   SESSION_NAME or "",
        "word":      word,
        "en_word":   word,
        "category":  req.category,
        "lang":      lang,
        "tsne_x":    round(float(tsne_pos[0]), 4),
        "tsne_y":    round(float(tsne_pos[1]), 4),
    })

    return {
        "status":   "ok",
        "word":     word,
        "category": req.category,
        "color":    req.color,
        "x": round(float(tsne_pos[0]), 4),
        "y": round(float(tsne_pos[1]), 4),
    }


@app.delete("/api/custom_words/{word}")
async def remove_custom_word(word: str, lang: str = 'en'):
    if lang not in LANG_STATE:
        lang = 'en'
    st = LANG_STATE[lang]
    keep = [i for i, l in enumerate(st['text_labels'])
            if not (l["en_word"] == word and l["category"] == "Custom")]
    if len(keep) == len(st['text_labels']):
        return {"status": "not_found", "word": word}

    st['text_embeddings'] = st['text_embeddings'][keep]
    st['text_labels']     = [st['text_labels'][i] for i in keep]
    st['tsne_coords']     = st['tsne_coords'][keep]

    if "Custom" in TAXONOMY["categories"] and word in TAXONOMY["categories"]["Custom"]["words"]:
        TAXONOMY["categories"]["Custom"]["words"].remove(word)

    log.info(f"[{lang}] Removed custom word: '{word}'")
    return {"status": "ok", "word": word}


@app.delete("/api/custom_words")
async def clear_custom_words(lang: str = 'en'):
    if lang not in LANG_STATE:
        lang = 'en'
    st = LANG_STATE[lang]
    keep = [i for i, l in enumerate(st['text_labels']) if l["category"] != "Custom"]
    removed = len(st['text_labels']) - len(keep)

    st['text_embeddings'] = st['text_embeddings'][keep]
    st['text_labels']     = [st['text_labels'][i] for i in keep]
    st['tsne_coords']     = st['tsne_coords'][keep]

    if "Custom" in TAXONOMY["categories"]:
        TAXONOMY["categories"]["Custom"]["words"] = []

    log.info(f"[{lang}] Cleared {removed} custom words")
    return {"status": "ok", "removed": removed}


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

                top_k     = msg.get("top_k", MAX_LABELS)
                categories = set(msg["categories"]) if msg.get("categories") else None
                lang      = msg.get("lang", "en")

                result = process_frame(img, top_k=top_k, categories=categories, lang=lang)
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
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--projection", default="top1",
                   choices=["softmax", "weighted", "top1", "transform"])
    p.add_argument("--projection-tau", type=float, default=0.1)
    p.add_argument("--langs", default=None,
                   help="Comma-separated languages to load (default: all available). E.g. en,es")
    return p.parse_args()


def main():
    global MODEL, PREPROCESS, TOKENIZER, DEVICE, TAXONOMY
    global FAIRFACE_EMBEDDINGS, MAX_LABELS, args
    global SUPPORTED_LANGS

    args = parse_args()
    MAX_LABELS = args.max_labels

    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    log.info(f"Device: {DEVICE}")

    MODEL, PREPROCESS, TOKENIZER, _ = load_clip_model(args.model, DEVICE)

    tax_path = args.taxonomy or str(Path(__file__).parent / "config" / "sobit_taxonomy.json")
    with open(tax_path, encoding="utf-8") as f:
        TAXONOMY = json.load(f)
    log.info(f"Loaded taxonomy from {tax_path}")

    if "Custom" not in TAXONOMY["categories"]:
        TAXONOMY["categories"]["Custom"] = {"color": CUSTOM_COLOR, "words": []}

    # Determine which languages to initialise
    available = ['en']
    if 'translations' in TAXONOMY:
        available += [l for l in TAXONOMY['translations'] if l != 'en']

    if args.langs:
        requested = [l.strip() for l in args.langs.split(',')]
        langs_to_init = [l for l in requested if l in available]
    else:
        langs_to_init = available

    SUPPORTED_LANGS = langs_to_init
    log.info(f"Initialising languages: {SUPPORTED_LANGS}")

    # Build FairFace embeddings once (English prompts, language-independent)
    FAIRFACE_EMBEDDINGS = build_fairface_embeddings(TAXONOMY)

    # Build embeddings + UMAP for each language
    for lang in SUPPORTED_LANGS:
        init_lang_state(TAXONOMY, lang, args.umap_neighbors)

    _auto_open_session("default")

    import uvicorn
    log.info(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
