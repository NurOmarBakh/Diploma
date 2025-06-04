# src/indexer.py

import os
import pickle
import logging
import yaml

import numpy as np
import faiss

def load_config():
    src  = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("indexer")

cfg  = load_config()
ROOT = project_root()

EMB_DIR    = os.path.join(ROOT, cfg["data"]["embeddings_dir"])
INDEX_DIR  = os.path.join(ROOT, cfg["data"]["faiss_index_dir"])
FACTORY    = cfg["index"]["factory_string"]
os.makedirs(INDEX_DIR, exist_ok=True)

emb_path  = os.path.join(EMB_DIR, "embeddings.npy")
meta_path = os.path.join(EMB_DIR, "metadata.pkl")
idx_path  = os.path.join(INDEX_DIR, "index.faiss")
m_out     = os.path.join(INDEX_DIR, "metadata.pkl")

emb = np.load(emb_path).astype("float32")
emb = np.ascontiguousarray(emb)
logger.info(f"Loaded embeddings: {emb.shape}")

with open(meta_path, "rb") as f:
    meta = pickle.load(f)

d = emb.shape[1]
logger.info(f"Building index [{FACTORY}] on dim={d}")
idx = faiss.index_factory(d, FACTORY)

if not idx.is_trained:
    logger.info("Training…")
    idx.train(emb)
    logger.info("Trained.")

logger.info("Adding vectors…")
idx.add(emb)
logger.info(f"Total vectors: {idx.ntotal}")

logger.info(f"Writing index → {idx_path}")
faiss.write_index(idx, idx_path)
with open(m_out, "wb") as f:
    pickle.dump(meta, f)
logger.info("Saved index and metadata.")
