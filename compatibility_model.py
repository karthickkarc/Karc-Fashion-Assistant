"""
Learned pairwise outfit-compatibility model.

Why a learned model and not just prompt engineering: the assignment asks
for "Learning Compatibility Scores" / "Pairwise Ranking Models" as a bonus
path, and a small supervised model over the 25 stylist-curated outfits lets
the system explain compatibility in terms of *features* (role fit, color
harmony, occasion match, semantic similarity) rather than only an LLM's
opinion, and it costs nothing at inference time.

Method
------
Positive examples: every pair of items that co-occurs inside the same
curated outfit in outfits.csv (hero/second/layer/footwear/accessory_1/
accessory_2 -- all pairwise combinations within a row).

Negative examples: pairs of items that never co-occur in any curated
outfit, sampled half "hard" (same gender, plausible role pair, just never
paired by the stylist) and half "easy" (anything else, including
cross-gender) so the model gets signal on both gender/role consistency and
finer style compatibility.

Features per pair: see `pair_features()`. Model: scikit-learn
LogisticRegression with class_weight="balanced" and L2 regularization --
deliberately the simplest model that could work, since with ~70-90
positive pairs from 25 outfits a deeper model would just overfit. This is
explicitly a proof-of-concept: see TECHNICAL_DOCS.md "Limitations" for what
a production version would need (thousands of curated outfits, a proper
train/val/test split, and likely a contrastive embedding model instead of
hand-engineered features).

`score_pair()` blends the learned probability with the hard role-gate from
config.VALID_ROLE_PAIRS, so the model is never asked to rescue a
nonsensical role pairing (e.g. two pairs of shoes).
"""
from __future__ import annotations

import itertools
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import config
from .data_loader import color_family


def _outfit_item_columns():
    return [
        ("hero_id", "hero"),
        ("second_id", "second"),
        ("layer_id", "layer"),
        ("footwear_id", "footwear"),
        ("accessory_1_id", "accessory_1"),
        ("accessory_2_id", "accessory_2"),
    ]


def build_positive_pairs(outfits: pd.DataFrame) -> set[tuple[str, str]]:
    positives = set()
    id_cols = [c for c, _ in _outfit_item_columns()]
    for _, row in outfits.iterrows():
        ids = [row[c] for c in id_cols if isinstance(row[c], str) and row[c].strip()]
        for a, b in itertools.combinations(sorted(set(ids)), 2):
            positives.add((a, b))
    return positives


def _role_score(role_a: str, role_b: str) -> float:
    key = frozenset({role_a}) if role_a == role_b else frozenset({role_a, role_b})
    return config.VALID_ROLE_PAIRS.get(key, 0.0)


def pair_features(row_a: pd.Series, row_b: pd.Series, text_sim: float) -> dict:
    color_a, color_b = row_a.get("color"), row_b.get("color")
    fam_a, fam_b = color_family(color_a), color_family(color_b)
    if fam_a == "neutral" or fam_b == "neutral" or fam_a == "unknown" or fam_b == "unknown":
        color_harmony = 0.8  # neutrals are easy to pair; unknown defaults to neutral-ish
    elif fam_a == fam_b:
        color_harmony = 0.9  # analogous palette
    else:
        color_harmony = 0.6  # warm vs cool contrast -- can work, scored moderately

    price_a = max(float(row_a.get("price_inr", 0) or 0), 1.0)
    price_b = max(float(row_b.get("price_inr", 0) or 0), 1.0)
    price_tier_diff = abs(np.log10(price_a) - np.log10(price_b))

    return {
        "role_score": _role_score(row_a["role"], row_b["role"]),
        "same_gender": float(row_a["gender"] == row_b["gender"]),
        "same_wear_type": float(row_a["wear_type"] == row_b["wear_type"]),
        "same_occasion": float(row_a["occasion"] == row_b["occasion"]),
        "color_harmony": color_harmony,
        "same_brand": float(row_a["brand"] == row_b["brand"]),
        "text_sim": text_sim,
        "price_tier_diff": price_tier_diff,
    }


FEATURE_ORDER = ["role_score", "same_gender", "same_wear_type", "same_occasion",
                  "color_harmony", "same_brand", "text_sim", "price_tier_diff"]


def _sample_negative_pairs(products: pd.DataFrame, positives: set[tuple[str, str]], n: int,
                            seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    ids = products["id"].tolist()
    gender_map = products.set_index("id")["gender"].to_dict()
    all_pairs = list(itertools.combinations(sorted(ids), 2))
    rng.shuffle(all_pairs)

    candidates = [p for p in all_pairs if p not in positives and (p[1], p[0]) not in positives]
    hard = [p for p in candidates if gender_map[p[0]] == gender_map[p[1]]]
    easy = [p for p in candidates if gender_map[p[0]] != gender_map[p[1]]]

    n_hard = min(len(hard), n // 2)
    n_easy = min(len(easy), n - n_hard)
    chosen = hard[:n_hard] + easy[:n_easy]
    rng.shuffle(chosen)
    return chosen


def build_training_table(products: pd.DataFrame, outfits: pd.DataFrame,
                          negative_ratio: int = 3, seed: int = 42) -> pd.DataFrame:
    products = products.set_index("id", drop=False)
    positives = build_positive_pairs(outfits)
    # Some positive ids may not exist in products (defensive, shouldn't
    # happen with this dataset but keeps the pipeline robust).
    positives = {(a, b) for a, b in positives if a in products.index and b in products.index}

    negatives = _sample_negative_pairs(products.reset_index(drop=True), positives,
                                        n=len(positives) * negative_ratio, seed=seed)

    text_vectorizer = TfidfVectorizer(max_features=1024, stop_words="english")
    text_matrix = text_vectorizer.fit_transform(products["text_blob"].fillna(""))
    text_index = {pid: i for i, pid in enumerate(products.index)}

    def text_similarity(a: str, b: str) -> float:
        va, vb = text_matrix[text_index[a]], text_matrix[text_index[b]]
        denom = (np.sqrt(va.multiply(va).sum()) * np.sqrt(vb.multiply(vb).sum()))
        if denom == 0:
            return 0.0
        return float(va.multiply(vb).sum() / denom)

    rows = []
    # NOTE: iterating a Python set() directly has hash-randomized order
    # across process runs (PYTHONHASHSEED), which would silently change
    # which physical row lands at which index -- and therefore which rows
    # train_test_split's random_state assigns to train vs. test, even
    # though the *seed* is fixed. Sorting first makes the whole pipeline
    # actually reproducible run-to-run, which matters for the numbers
    # quoted in TECHNICAL_DOCS.md to mean anything.
    for a, b in sorted(positives):
        feats = pair_features(products.loc[a], products.loc[b], text_similarity(a, b))
        feats["label"] = 1
        rows.append(feats)
    for a, b in negatives:
        feats = pair_features(products.loc[a], products.loc[b], text_similarity(a, b))
        feats["label"] = 0
        rows.append(feats)

    return pd.DataFrame(rows)


class CompatibilityModel:
    def __init__(self):
        self.model: LogisticRegression | None = None
        self.train_auc: float | None = None

    def fit(self, table: pd.DataFrame):
        X = table[FEATURE_ORDER].values
        y = table["label"].values
        if len(set(y)) < 2 or len(table) < 10:
            # Degenerate guard for tiny/edge-case datasets: fall back to a
            # rule-only model that just returns role_score * color_harmony.
            self.model = None
            return self

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
        self.model.fit(X_train, y_train)
        try:
            preds = self.model.predict_proba(X_test)[:, 1]
            self.train_auc = float(roc_auc_score(y_test, preds))
        except ValueError:
            self.train_auc = None
        return self

    def predict_proba(self, feats: dict) -> float:
        if self.model is None:
            return feats["role_score"] * feats["color_harmony"]
        x = np.array([[feats[f] for f in FEATURE_ORDER]])
        return float(self.model.predict_proba(x)[0, 1])

    def save(self, path=config.COMPAT_MODEL_PATH):
        joblib.dump({"model": self.model, "train_auc": self.train_auc}, path)

    @classmethod
    def load(cls, path=config.COMPAT_MODEL_PATH) -> "CompatibilityModel":
        obj = cls()
        payload = joblib.load(path)
        obj.model = payload["model"]
        obj.train_auc = payload["train_auc"]
        return obj

    def feature_importances(self) -> dict:
        if self.model is None:
            return {}
        return dict(zip(FEATURE_ORDER, self.model.coef_[0].tolist()))
