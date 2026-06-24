"""
Text/image embedding backends.

Default backend ("tfidf"): scikit-learn TF-IDF over each product's
text_blob. Zero downloads, deterministic, instant -- this is what ships
active by default so the prototype runs in any environment, including
fully offline ones.

Optional backend ("clip" / "fashionclip"): joint image+text embeddings
using a Hugging Face CLIP / FashionCLIP checkpoint via `transformers`.
This is the "bonus" path called out in the assignment (FashionCLIP /
Vision Transformers / image embeddings). It requires internet access on
first run to download model weights, and an extra `pip install -r
requirements-clip.txt`. The rest of the pipeline (FAISS index,
compatibility model, recommender) is agnostic to which backend produced
the vectors, as long as dimensionality is consistent -- swap the backend
with `EMBEDDING_BACKEND=fashionclip` in `.env` and re-run
`scripts/build_index.py`.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config


class TfidfEmbedder:
    """Default, dependency-light embedder."""

    name = "tfidf"

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(
            max_features=2048,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(texts)
        return matrix.toarray().astype("float32")

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("Embedder not fitted yet -- call fit_transform first.")
        return self.vectorizer.transform(texts).toarray().astype("float32")

    def save(self, path=config.TFIDF_VECTORIZER_PATH):
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path=config.TFIDF_VECTORIZER_PATH) -> "TfidfEmbedder":
        obj = cls()
        obj.vectorizer = joblib.load(path)
        return obj


class ClipEmbedder:
    """Optional CLIP / FashionCLIP backend. Imports `transformers`/`torch`
    lazily so the rest of the system works without these (heavy) deps
    installed. Produces a joint image+text embedding per product by
    averaging the image embedding (from the product photo) and the text
    embedding (from the text_blob), L2-normalized.
    """

    name = "clip"
    # Swap this for "patrickjohncyh/fashion-clip" to use FashionCLIP instead
    # of generic CLIP -- that single line is the entire "upgrade".
    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or self.MODEL_ID
        self._model = None
        self._processor = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP backend requires `torch` and `transformers`. "
                "Install with: pip install -r requirements-clip.txt"
            ) from exc
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model.eval()

    def embed_products(self, products: pd.DataFrame) -> np.ndarray:
        import torch
        from PIL import Image

        self._lazy_load()
        vectors = []
        with torch.no_grad():
            for _, row in products.iterrows():
                text_inputs = self._processor(
                    text=[row["text_blob"][:200]], return_tensors="pt", padding=True, truncation=True
                )
                text_emb = self._model.get_text_features(**text_inputs)[0]

                image_path = config.RAW_DIR / row["image"]
                if image_path.exists():
                    image = Image.open(image_path).convert("RGB")
                    image_inputs = self._processor(images=image, return_tensors="pt")
                    image_emb = self._model.get_image_features(**image_inputs)[0]
                    combined = (text_emb + image_emb) / 2.0
                else:
                    combined = text_emb

                combined = combined / combined.norm()
                vectors.append(combined.numpy())
        return np.vstack(vectors).astype("float32")


def get_embedder():
    """Factory respecting EMBEDDING_BACKEND in config/.env."""
    if config.EMBEDDING_BACKEND in ("clip", "fashionclip"):
        model_id = (
            "patrickjohncyh/fashion-clip"
            if config.EMBEDDING_BACKEND == "fashionclip"
            else ClipEmbedder.MODEL_ID
        )
        return ClipEmbedder(model_id=model_id)
    return TfidfEmbedder()
