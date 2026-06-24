"""
Thin FAISS wrapper for similarity search over product embeddings.

We use a flat, exact, cosine-similarity index (inner product over
L2-normalized vectors). For 68 products this is obviously overkill on
performance grounds, but it is the same API you would use for a 500k-SKU
catalog -- swapping IndexFlatIP for IndexIVFFlat/HNSW later is a one-line
change in `build()`, which is the point of using FAISS here rather than a
hand-rolled cosine loop: the retrieval code does not need to change at all
when the catalog grows.
"""
from __future__ import annotations

import faiss
import joblib
import numpy as np

from . import config


class ProductVectorStore:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.id_map: list[str] = []  # row i of the index -> product id

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build(self, vectors: np.ndarray, product_ids: list[str]):
        vectors = self._normalize(vectors.astype("float32"))
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.id_map = list(product_ids)
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        query_vector = self._normalize(query_vector.reshape(1, -1).astype("float32"))
        scores, indices = self.index.search(query_vector, min(top_k, len(self.id_map)))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.id_map[idx], float(score)))
        return results

    def save(self, index_path=config.FAISS_INDEX_PATH, id_map_path=config.FAISS_ID_MAP_PATH):
        faiss.write_index(self.index, str(index_path))
        joblib.dump(self.id_map, id_map_path)

    @classmethod
    def load(cls, index_path=config.FAISS_INDEX_PATH, id_map_path=config.FAISS_ID_MAP_PATH) -> "ProductVectorStore":
        store = cls()
        store.index = faiss.read_index(str(index_path))
        store.id_map = joblib.load(id_map_path)
        return store
