"""
Offline pipeline: load dataset -> enrich -> fit embedder -> build FAISS
index -> train compatibility model -> persist all artifacts to
data/artifacts/. Run this once after cloning, and again any time
products.csv/outfits.csv change.

    python scripts/build_index.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.compatibility_model import CompatibilityModel, build_training_table
from src.data_loader import load_products, load_outfits, summarize_dataset
from src.embeddings import get_embedder, TfidfEmbedder
from src.vector_store import ProductVectorStore


def main():
    t0 = time.time()
    print(f"[1/5] Loading dataset from {config.RAW_DIR} ...")
    products = load_products()
    outfits = load_outfits()
    print(summarize_dataset(products, outfits).report())

    print(f"\n[2/5] Fitting embedder (backend={config.EMBEDDING_BACKEND}) ...")
    embedder = get_embedder()
    if isinstance(embedder, TfidfEmbedder):
        vectors = embedder.fit_transform(products["text_blob"].tolist())
        embedder.save()
    else:
        vectors = embedder.embed_products(products)
    print(f"    embedding matrix shape: {vectors.shape}")

    print("\n[3/5] Building FAISS index ...")
    store = ProductVectorStore().build(vectors, products["id"].tolist())
    store.save()
    print(f"    indexed {len(products)} products into FAISS (IndexFlatIP)")

    print("\n[4/5] Training pairwise compatibility model ...")
    table = build_training_table(products, outfits)
    print(f"    training table: {len(table)} pairs "
          f"({int(table['label'].sum())} positive / {len(table) - int(table['label'].sum())} negative)")
    model = CompatibilityModel().fit(table)
    model.save()
    if model.train_auc is not None:
        print(f"    held-out AUC: {model.train_auc:.3f}")
        print(f"    feature weights: {model.feature_importances()}")
    else:
        print("    (model fell back to rule-only scoring -- training table too small/degenerate)")

    products.to_csv(config.PRODUCTS_ENRICHED_PATH, index=False)
    print(f"\n[5/5] Saved enriched products to {config.PRODUCTS_ENRICHED_PATH}")
    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts written to {config.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
