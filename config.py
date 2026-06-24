"""
Central configuration for the Dare XAI Fashion Outfit Recommendation System.

Holds filesystem paths, the category -> outfit-role mapping, the role
compatibility matrix, and small lexicons (colors, occasions, styles) used
across the pipeline. Keeping these in one place makes the system easy to
extend to a larger dataset later -- most "add a new category" changes only
touch this file.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # no-op if .env doesn't exist; never raises
except ImportError:
    pass  # python-dotenv not installed -- fall back to real environment variables only

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = DATA_DIR / "artifacts"

PRODUCTS_CSV = RAW_DIR / "products.csv"
OUTFITS_CSV = RAW_DIR / "outfits.csv"
IMAGES_DIR = RAW_DIR / "images"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = ARTIFACT_DIR / "tfidf_matrix.joblib"
FAISS_INDEX_PATH = ARTIFACT_DIR / "product_index.faiss"
FAISS_ID_MAP_PATH = ARTIFACT_DIR / "product_id_map.joblib"
COMPAT_MODEL_PATH = ARTIFACT_DIR / "compatibility_model.joblib"
PRODUCTS_ENRICHED_PATH = ARTIFACT_DIR / "products_enriched.csv"

# --------------------------------------------------------------------------
# Embedding backend
# --------------------------------------------------------------------------
# "tfidf"      -> scikit-learn TF-IDF text embedding. Zero downloads, works
#                 fully offline, deterministic. Used as the default so the
#                 system runs out-of-the-box in any environment.
# "clip"       -> OpenAI CLIP (via `transformers`) joint image+text embedding.
# "fashionclip"-> FashionCLIP checkpoint (via `transformers`), domain-tuned
#                 on fashion product images -- the "bonus" path described in
#                 the assignment. Requires internet access to download
#                 weights from Hugging Face on first run.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "tfidf").lower()

# --------------------------------------------------------------------------
# LLM backend (conversational NLU + reasoning polish)
# --------------------------------------------------------------------------
# "none"   -> rule-based NLU + template explanations (default, no API key
#             needed, fully deterministic, used for grading/demo safety).
# "gemini" / "openai" / "anthropic" -> use the respective API if a key is
#             present in the environment (see .env.example).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none").lower()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# --------------------------------------------------------------------------
# Outfit roles
# --------------------------------------------------------------------------
# Every `category_label` in products.csv is bucketed into one outfit role.
# "onepiece" items (dresses, sarees, sherwanis, suits...) stand in for both
# topwear AND bottomwear by themselves.
ROLE_TOPWEAR = "topwear"
ROLE_BOTTOMWEAR = "bottomwear"
ROLE_ONEPIECE = "onepiece"
ROLE_FOOTWEAR = "footwear"
ROLE_LAYER = "layer"
ROLE_ACCESSORY = "accessory"

CATEGORY_TO_ROLE = {
    # topwear
    "Formal Shirts": ROLE_TOPWEAR,
    "Casual Shirts": ROLE_TOPWEAR,
    "Party Shirts": ROLE_TOPWEAR,
    "Linen Shirts": ROLE_TOPWEAR,
    "Tshirts": ROLE_TOPWEAR,
    "Polo Tshirts": ROLE_TOPWEAR,
    "Tops": ROLE_TOPWEAR,
    "Sweatshirts": ROLE_TOPWEAR,
    "Sweaters": ROLE_TOPWEAR,
    "Activewear": ROLE_TOPWEAR,
    # bottomwear
    "Trousers": ROLE_BOTTOMWEAR,
    "Jeans": ROLE_BOTTOMWEAR,
    "Chinos": ROLE_BOTTOMWEAR,
    "Shorts": ROLE_BOTTOMWEAR,
    "Skirts": ROLE_BOTTOMWEAR,
    "Track Pants": ROLE_BOTTOMWEAR,
    "Leggings": ROLE_BOTTOMWEAR,
    # onepiece / full outfit sets
    "Party Dresses": ROLE_ONEPIECE,
    "Casual Dresses": ROLE_ONEPIECE,
    "Maxi Dresses": ROLE_ONEPIECE,
    "Co Ord Sets": ROLE_ONEPIECE,
    "Kurta Sets": ROLE_ONEPIECE,
    "Sharara Sets": ROLE_ONEPIECE,
    "Salwar Suits": ROLE_ONEPIECE,
    "Wedding Sarees": ROLE_ONEPIECE,
    "Suits": ROLE_ONEPIECE,
    "Sherwanis": ROLE_ONEPIECE,
    # footwear
    "Heels": ROLE_FOOTWEAR,
    "Flats": ROLE_FOOTWEAR,
    "Sandals": ROLE_FOOTWEAR,
    "Boots": ROLE_FOOTWEAR,
    "Sneakers": ROLE_FOOTWEAR,
    "Running Shoes": ROLE_FOOTWEAR,
    "Loafers": ROLE_FOOTWEAR,
    "Formal Shoes": ROLE_FOOTWEAR,
    "Ethnic Footwear": ROLE_FOOTWEAR,
    # layer
    "Blazers": ROLE_LAYER,
    "Denim Jackets": ROLE_LAYER,
    "Long Coats": ROLE_LAYER,
    "Nehru Jackets": ROLE_LAYER,
    # accessory
    "Clutches": ROLE_ACCESSORY,
    "Handbags": ROLE_ACCESSORY,
    "Necklaces": ROLE_ACCESSORY,
    "Watches": ROLE_ACCESSORY,
    "Earrings": ROLE_ACCESSORY,
    "Sunglasses": ROLE_ACCESSORY,
    "Caps": ROLE_ACCESSORY,
}

# Which role-pairs are even allowed to be scored for compatibility. This is
# a hard rule-based gate applied *before* the learned model ever sees a
# pair -- it encodes "a shirt does not get judged against another shirt".
VALID_ROLE_PAIRS = {
    frozenset({ROLE_TOPWEAR, ROLE_BOTTOMWEAR}): 1.0,
    frozenset({ROLE_TOPWEAR, ROLE_FOOTWEAR}): 1.0,
    frozenset({ROLE_TOPWEAR, ROLE_LAYER}): 1.0,
    frozenset({ROLE_TOPWEAR, ROLE_ACCESSORY}): 0.8,
    frozenset({ROLE_BOTTOMWEAR, ROLE_FOOTWEAR}): 1.0,
    frozenset({ROLE_BOTTOMWEAR, ROLE_LAYER}): 0.8,
    frozenset({ROLE_BOTTOMWEAR, ROLE_ACCESSORY}): 0.6,
    frozenset({ROLE_ONEPIECE, ROLE_FOOTWEAR}): 1.0,
    frozenset({ROLE_ONEPIECE, ROLE_LAYER}): 1.0,
    frozenset({ROLE_ONEPIECE, ROLE_ACCESSORY}): 0.8,
    frozenset({ROLE_LAYER, ROLE_FOOTWEAR}): 0.7,
    frozenset({ROLE_LAYER, ROLE_ACCESSORY}): 0.6,
    frozenset({ROLE_FOOTWEAR, ROLE_ACCESSORY}): 0.5,
    frozenset({ROLE_ACCESSORY}): 0.3,  # accessory + accessory (same role)
}

ALL_ROLES = [ROLE_TOPWEAR, ROLE_BOTTOMWEAR, ROLE_ONEPIECE, ROLE_FOOTWEAR, ROLE_LAYER, ROLE_ACCESSORY]

# --------------------------------------------------------------------------
# Lexicons for the rule-based NLU fallback (no LLM key required)
# --------------------------------------------------------------------------
OCCASION_KEYWORDS = {
    "office": ["office", "work", "business meeting", "meeting", "interview", "boardroom", "corporate", "9 to 5"],
    "party": ["party", "night out", "club", "cocktail", "dinner date", "date night", "clubbing"],
    "wedding": ["wedding", "shaadi", "marriage", "reception", "engagement"],
    "festive": ["festive", "festival", "diwali", "puja", "navratri", "eid", "celebration"],
    "casual": ["casual", "everyday", "weekend", "hangout", "errands", "college", "brunch"],
    "sports": ["sports", "gym", "workout", "running", "yoga", "athleisure", "training"],
    "vacation": ["vacation", "beach", "holiday", "travel", "trip", "resort"],
    "winter": ["winter", "cold", "chilly", "snow"],
}

GENDER_KEYWORDS = {
    "men": ["male", "man", "men", "guy", "boyfriend", "husband", "boy", "groom"],
    "women": ["female", "woman", "women", "girl", "girlfriend", "wife", "lady", "bride"],
}

STYLE_KEYWORDS = {
    "western": ["western", "smart casual", "formal", "modern"],
    "ethnic": ["ethnic", "traditional", "indowestern", "indo-western", "desi"],
}

COLOR_LEXICON = [
    "black", "white", "navy", "blue", "red", "maroon", "wine", "magenta", "pink", "beige",
    "brown", "grey", "gray", "green", "olive", "yellow", "gold", "silver", "ivory", "cream",
    "mustard", "purple", "lavender", "orange", "tan", "khaki", "denim", "indigo", "burgundy",
    "charcoal", "mocha", "sage", "blush", "teal", "rust", "peach", "coral",
]

NEUTRAL_COLORS = {"black", "white", "navy", "grey", "gray", "beige", "brown", "ivory", "cream",
                   "tan", "khaki", "denim", "charcoal", "mocha"}
WARM_COLORS = {"red", "maroon", "wine", "magenta", "pink", "yellow", "gold", "mustard", "orange",
               "burgundy", "blush", "rust", "peach", "coral"}
COOL_COLORS = {"blue", "green", "olive", "purple", "lavender", "indigo", "sage", "teal", "silver"}
