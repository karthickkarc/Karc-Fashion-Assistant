"""
Loads products.csv / outfits.csv, enriches each product with a derived
`role` (topwear/bottomwear/onepiece/footwear/layer/accessory) and a derived
`color` extracted from its text fields, and builds the combined text blob
used for embeddings.

This module is intentionally pandas-only (no model downloads) so that
`python scripts/build_index.py` can run on a totally offline machine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from . import config


def _extract_color(text: str) -> str | None:
    """Pick the first known color word that appears in `text`, preferring
    multi-word/compound matches is unnecessary here since our lexicon is
    single tokens; we just scan in lexicon order so more distinctive colors
    (e.g. 'navy') are not masked by generic ones."""
    if not isinstance(text, str):
        return None
    lowered = text.lower()
    for color in config.COLOR_LEXICON:
        if re.search(rf"\b{re.escape(color)}\b", lowered):
            return "grey" if color == "gray" else color
    return None


def color_family(color: str | None) -> str:
    if color is None:
        return "unknown"
    if color in config.NEUTRAL_COLORS:
        return "neutral"
    if color in config.WARM_COLORS:
        return "warm"
    if color in config.COOL_COLORS:
        return "cool"
    return "unknown"


def load_products() -> pd.DataFrame:
    """Load products.csv and enrich it with `role`, `color`, `color_family`
    and a combined `text_blob` column used downstream for TF-IDF / CLIP
    text embeddings."""
    df = pd.read_csv(config.PRODUCTS_CSV)

    df["role"] = df["category_label"].map(config.CATEGORY_TO_ROLE)
    unmapped = df[df["role"].isna()]
    if len(unmapped):
        # Fail loudly rather than silently dropping items -- a new category
        # in a larger dataset must be mapped in config.CATEGORY_TO_ROLE.
        raise ValueError(
            f"Unmapped category_label values found, please add them to "
            f"CATEGORY_TO_ROLE in src/config.py: {sorted(unmapped['category_label'].unique())}"
        )

    search_text = (
        df["name"].fillna("") + " " + df["description"].fillna("") + " " + df["tags"].fillna("")
    )
    df["color"] = search_text.map(_extract_color)
    df["color_family"] = df["color"].map(color_family)

    df["text_blob"] = (
        df["name"].fillna("") + ". " +
        df["category_label"].fillna("") + ". " +
        df["occasion"].fillna("") + " occasion. " +
        df["wear_type"].fillna("") + " wear. " +
        df["description"].fillna("") + " " +
        df["tags"].fillna("").str.replace(";", " ")
    ).str.strip()

    return df


def load_outfits() -> pd.DataFrame:
    return pd.read_csv(config.OUTFITS_CSV)


@dataclass
class DatasetSummary:
    n_products: int
    n_outfits: int
    gender_counts: dict
    occasion_counts: dict
    role_counts: dict
    wear_type_counts: dict
    products_missing_color: int

    def report(self) -> str:
        lines = [
            f"Products: {self.n_products}",
            f"Curated outfits: {self.n_outfits}",
            f"Gender split: {self.gender_counts}",
            f"Occasion split: {self.occasion_counts}",
            f"Role split: {self.role_counts}",
            f"Wear-type split: {self.wear_type_counts}",
            f"Products with no detectable color in text: {self.products_missing_color}",
        ]
        return "\n".join(lines)


def summarize_dataset(products: pd.DataFrame | None = None,
                       outfits: pd.DataFrame | None = None) -> DatasetSummary:
    products = products if products is not None else load_products()
    outfits = outfits if outfits is not None else load_outfits()
    return DatasetSummary(
        n_products=len(products),
        n_outfits=len(outfits),
        gender_counts=products["gender"].value_counts().to_dict(),
        occasion_counts=products["occasion"].value_counts().to_dict(),
        role_counts=products["role"].value_counts().to_dict(),
        wear_type_counts=products["wear_type"].value_counts().to_dict(),
        products_missing_color=int(products["color"].isna().sum()),
    )


if __name__ == "__main__":
    products = load_products()
    outfits = load_outfits()
    print(summarize_dataset(products, outfits).report())
