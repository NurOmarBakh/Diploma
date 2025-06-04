# src/embed.py

import os
import json
import pickle
import logging
import yaml

import numpy as np
from sentence_transformers import SentenceTransformer

def load_config():
    src  = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed")

cfg  = load_config()
ROOT = project_root()

RAW_JSON = os.path.join(ROOT, cfg["data"]["raw_json_dir"])
EMB_DIR  = os.path.join(ROOT, cfg["data"]["embeddings_dir"])
os.makedirs(EMB_DIR, exist_ok=True)

texts = []
meta  = []

for fn in os.listdir(RAW_JSON):
    if not fn.endswith(".json"):
        continue
    page = json.load(open(os.path.join(RAW_JSON, fn), encoding="utf-8"))
    for ch in page["chunks"]:
        texts.append(ch["text"])
        meta.append({
            "page_url":   page["page_url"],
            "page_title": page["page_title"],
            "page_lang":  page["page_lang"],
            **ch
        })

logger.info(f"Loaded {len(texts)} chunks")

model = SentenceTransformer(cfg["embed"]["model_name"])
embs  = model.encode(texts, batch_size=cfg["embed"]["batch_size"], show_progress_bar=True)

np.save(os.path.join(EMB_DIR, "embeddings.npy"), embs.astype("float32"))
with open(os.path.join(EMB_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(meta, f)

logger.info("Embeddings and metadata saved.")
