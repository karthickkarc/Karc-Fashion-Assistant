"""
demo.py -- runs the ENTIRE system end-to-end, in the correct order, with
clean, narrated console output. This is the one file to run on screen
while recording the assignment's demo video: each section prints a
banner you can read aloud, then the real output of the real code (no
mocked data, no pre-baked numbers).

    python demo.py              # runs straight through
    python demo.py --pause      # waits for Enter between sections,
                                 # so you can talk over each one while
                                 # recording without the output racing
                                 # ahead of you

Order of execution (mirrors the two pipelines in ARCHITECTURE.md):
    1. Dataset analysis              (src/data_loader.py)
    2. Offline indexing pipeline     (embeddings -> FAISS -> compatibility model)
    3. Model evaluation              (leave-one-out Recall@3)
    4. Outfit Compatibility Engine   (recommend_from_item)
    5. Conversational Assistant      (multi-turn chat_assistant.py)
    6. Wrap-up summary
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.chat_assistant import ChatAssistant
from src.compatibility_model import CompatibilityModel, build_training_table
from src.data_loader import load_outfits, load_products, summarize_dataset
from src.embeddings import TfidfEmbedder, get_embedder
from src.recommender import OutfitRecommender
from src.user_profile import UserProfile
from src.vector_store import ProductVectorStore
from scripts.evaluate import main as run_leave_one_out_evaluation

PAUSE = False


def banner(step: str, title: str):
    width = 78
    print("\n" + "=" * width)
    print(f" {step}  {title}")
    print("=" * width)
    if PAUSE:
        input("   [press Enter to run this section] ")


def sub(text: str):
    print(f"\n--- {text} ---")


def kv(label: str, value):
    print(f"  {label:<32s} {value}")


def pause_after():
    if PAUSE:
        input("\n   [press Enter to continue to the next section] ")


# ============================================================ SECTION 1
def section_dataset_analysis():
    banner("STEP 1 / 6", "Dataset analysis")
    products = load_products()
    outfits = load_outfits()
    summary = summarize_dataset(products, outfits)

    sub("Headline numbers")
    kv("Products:", summary.n_products)
    kv("Curated stylist outfits:", summary.n_outfits)
    kv("Distinct categories:", products["category_label"].nunique())
    kv("Brands:", products["brand"].nunique())
    kv("Price range (INR):", f"{int(products['price_inr'].min())} - {int(products['price_inr'].max())}")
    kv("Products with no detectable color in text:", summary.products_missing_color)

    sub("Gender split")
    for k, v in summary.gender_counts.items():
        kv(f"  {k}", v)

    sub("Occasion split")
    for k, v in sorted(summary.occasion_counts.items(), key=lambda kv: -kv[1]):
        kv(f"  {k}", v)

    sub("Outfit-role split (this project's category -> role mapping)")
    for k, v in sorted(summary.role_counts.items(), key=lambda kv: -kv[1]):
        kv(f"  {k}", v)

    sub("Sample enriched product (note the derived role/color/text_blob fields)")
    row = products.iloc[0]
    kv("name:", row["name"])
    kv("category_label -> role:", f"{row['category_label']} -> {row['role']}")
    kv("derived color:", row["color"])
    kv("occasion / gender:", f"{row['occasion']} / {row['gender']}")

    return products, outfits


# ============================================================ SECTION 2
def section_offline_pipeline(products, outfits):
    banner("STEP 2 / 6", "Offline indexing pipeline (embeddings -> FAISS -> compatibility model)")
    t0 = time.time()

    sub(f"Fitting embedder (backend = '{config.EMBEDDING_BACKEND}')")
    embedder = get_embedder()
    if isinstance(embedder, TfidfEmbedder):
        vectors = embedder.fit_transform(products["text_blob"].tolist())
        embedder.save()
    else:
        vectors = embedder.embed_products(products)
    kv("Embedding matrix shape:", vectors.shape)

    sub("Building the FAISS vector index")
    store = ProductVectorStore().build(vectors, products["id"].tolist())
    store.save()
    kv("Vectors indexed (IndexFlatIP, cosine similarity):", len(products))

    sub("Building the pairwise training table from the 25 curated outfits")
    table = build_training_table(products, outfits)
    n_pos = int(table["label"].sum())
    n_neg = len(table) - n_pos
    kv("Positive pairs (co-occur in a curated outfit):", n_pos)
    kv("Negative pairs (sampled, half same-gender 'hard'):", n_neg)

    sub("Training the compatibility model (Logistic Regression)")
    model = CompatibilityModel().fit(table)
    model.save()
    if model.train_auc is not None:
        kv("Held-out AUC (25% stratified split):", f"{model.train_auc:.3f}")
        sub("Learned feature weights (what the model thinks matters)")
        for feat, weight in sorted(model.feature_importances().items(), key=lambda kv: -abs(kv[1])):
            sign = "+" if weight >= 0 else ""
            kv(f"  {feat}", f"{sign}{weight:.2f}")
    else:
        print("  (fell back to rule-only scoring -- training table too small/degenerate)")

    products.to_csv(config.PRODUCTS_ENRICHED_PATH, index=False)
    kv("\nArtifacts written to:", str(config.ARTIFACT_DIR))
    kv("Pipeline time:", f"{time.time() - t0:.2f}s")
    pause_after()


# ============================================================ SECTION 3
def section_evaluation():
    banner("STEP 3 / 6", "Evaluating the compatibility model (leave-one-out Recall@3)")
    print("  (Sanity check, not a generalization estimate -- see TECHNICAL_DOCS.md section 4)\n")
    run_leave_one_out_evaluation()
    pause_after()


# ============================================================ SECTION 4
def _print_outfit_recommendations(result: dict):
    anchor = result["anchor"]
    print(f"\n  ANCHOR ITEM: {anchor['name']}  "
          f"[{anchor['category_label']}, {anchor.get('color')}, {anchor['occasion']}, "
          f"Rs {int(anchor['price_inr'])}]")
    for role, recs in result["recommendations"].items():
        if not recs:
            continue
        top = recs[0]
        item = top["item"]
        pct = int(round(top["score"] * 100))
        print(f"\n  -> {role.upper()}: {item['name']}  ({pct}% match)")
        print(f"     Stylist's Note: {top['reason']}")


def section_item_anchored_demo(recommender: OutfitRecommender):
    banner("STEP 4 / 6", "Outfit Compatibility Engine (single item -> complete outfit)")
    print("  This is the assignment's example flow: 'Input: White Shirt -> Output: complete outfit'")

    demo_items = [
        ("myntra_28569210", "Men's formal shirt -- the assignment's literal worked example"),
        ("ajio_703182002", "Women's bodycon dress -- a 'onepiece' anchor (no separate top/bottom needed)"),
    ]
    for item_id, label in demo_items:
        sub(label)
        result = recommender.recommend_from_item(item_id, top_k_per_role=1)
        _print_outfit_recommendations(result)
    pause_after()


# ============================================================ SECTION 5
def _print_chat_turn(turn: dict):
    print(f"\n  USER: {turn['query']}")
    print(f"  PROFILE (remembered across turns): {turn['profile']}")
    print(f"  ASSISTANT: {turn['reply']}")
    for outfit in turn["outfits"]:
        pct = int(round(outfit["avg_compat"] * 100))
        item_lines = ", ".join(f"{role}: {item['name']}" for role, item in outfit["items"].items())
        print(f"\n    OUTFIT ({pct}% compatibility, Rs {int(outfit['total_price_inr'])} total)")
        print(f"      {item_lines}")
        print(f"      Stylist's Note: {outfit['reason']}")


def section_conversational_demo(recommender: OutfitRecommender):
    banner("STEP 5 / 6", "Conversational Fashion Assistant (multi-turn, profile memory)")
    print("  Watch the PROFILE line across turns: gender/age/occasion persist until the user")
    print("  changes them -- e.g. gender flips from 'men' to 'women' on turn 3 and then stays")
    print("  'women' on turn 4, even though turn 4 never mentions gender again.")
    chat = ChatAssistant(recommender)

    example_queries = [
        "I am a 22-year-old male looking for a casual summer outfit.",
        "I need an outfit for a business meeting.",  # gender carries over from above; occasion updates
        "I am attending a wedding next weekend, I'm a woman.",  # gender switch -- exercises the
                                                                  # word-boundary fix ("woman" no longer
                                                                  # mis-parsed as "man"), see TECHNICAL_DOCS.md
        "Suggest something stylish for a beach vacation.",  # gender carries over from above
    ]
    for query in example_queries:
        turn = chat.ask(query, top_n_outfits=1)
        _print_chat_turn(turn)
    pause_after()


# ============================================================ SECTION 6
def section_summary():
    banner("STEP 6 / 6", "Summary")
    print("""
  What you just watched, end to end, with no pre-baked output:
    1. Loaded and analyzed the real 68-product / 25-outfit dataset
    2. Built TF-IDF embeddings + a FAISS index + a learned pairwise
       compatibility model (Logistic Regression, ~0.87 held-out AUC)
    3. Sanity-checked the model with a leave-one-out Recall@3 evaluation
    4. Ran the item-anchored "Outfit Compatibility Engine"
    5. Ran the conversational assistant across 4 free-text queries,
       with profile memory persisting gender/age/occasion across turns

  Further reading:
    README.md            -- setup & quickstart
    ARCHITECTURE.md       -- full system design walkthrough
    TECHNICAL_DOCS.md     -- dataset analysis, modeling details, limitations
    docs/architecture_diagram.svg -- the pipeline diagram

  Interactive prototype:
    streamlit run app.py
    uvicorn api:app --reload --port 8000
""")


def main():
    global PAUSE
    parser = argparse.ArgumentParser(description="Run the full Dare XAI fashion assistant pipeline, narrated.")
    parser.add_argument("--pause", action="store_true",
                         help="Wait for Enter between sections (recommended while recording a demo video).")
    args = parser.parse_args()
    PAUSE = args.pause

    products, outfits = section_dataset_analysis()
    pause_after()

    section_offline_pipeline(products, outfits)
    section_evaluation()

    recommender = OutfitRecommender.load_default()
    section_item_anchored_demo(recommender)
    section_conversational_demo(recommender)

    section_summary()


if __name__ == "__main__":
    main()
