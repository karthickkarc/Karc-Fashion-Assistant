"""
Lightweight quantitative check: for every curated outfit in outfits.csv,
hide one of its items and ask the compatibility model to rank all
same-role, same-gender candidates against the remaining items. We report
Recall@3 -- did the held-out (correct, stylist-chosen) item land in the
model's top 3 guesses for that role?

This is a leave-one-out check on the SAME 25 outfits the model was partly
trained on (some of each outfit's pairs were used as positive examples),
so treat the number as a sanity check of "did the model learn the
training signal at all", not as a held-out generalization metric. A
genuine generalization estimate would need outfits the model never saw,
which this 25-outfit dataset is too small to split meaningfully -- see
TECHNICAL_DOCS.md.

    python scripts/evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.compatibility_model import CompatibilityModel, _outfit_item_columns, pair_features
from src.data_loader import load_products, load_outfits
from src.embeddings import TfidfEmbedder


def main():
    products = load_products().set_index("id", drop=False)
    outfits = load_outfits()
    embedder = TfidfEmbedder.load()
    model = CompatibilityModel.load()

    def text_sim(a, b):
        vecs = embedder.transform([products.loc[a, "text_blob"], products.loc[b, "text_blob"]])
        denom = ((vecs[0] @ vecs[0]) ** 0.5) * ((vecs[1] @ vecs[1]) ** 0.5)
        return float(vecs[0] @ vecs[1] / denom) if denom else 0.0

    id_cols = [c for c, _ in _outfit_item_columns()]
    total, hits_at_3 = 0, 0
    per_role_total, per_role_hits = {}, {}

    for _, row in outfits.iterrows():
        ids = [row[c] for c in id_cols if isinstance(row[c], str) and row[c].strip()]
        ids = [i for i in ids if i in products.index]
        if len(ids) < 2:
            continue
        for held_out in ids:
            context = [i for i in ids if i != held_out]
            role = products.loc[held_out, "role"]
            gender = products.loc[held_out, "gender"]
            candidates = products[(products["role"] == role) & (products["gender"] == gender)].index.tolist()
            if held_out not in candidates or len(candidates) < 2:
                continue

            scored = []
            for cand in candidates:
                scores = []
                for ctx_id in context:
                    feats = pair_features(products.loc[ctx_id], products.loc[cand], text_sim(ctx_id, cand))
                    scores.append(model.predict_proba(feats) * feats["role_score"])
                scored.append((cand, sum(scores) / len(scores) if scores else 0.0))
            scored.sort(key=lambda t: t[1], reverse=True)
            top3 = [c for c, _ in scored[:3]]

            total += 1
            per_role_total[role] = per_role_total.get(role, 0) + 1
            if held_out in top3:
                hits_at_3 += 1
                per_role_hits[role] = per_role_hits.get(role, 0) + 1

    print(f"Leave-one-out Recall@3 over {total} held-out items: {hits_at_3 / total:.1%}")
    print("\nPer-role breakdown:")
    for role in sorted(per_role_total):
        hits = per_role_hits.get(role, 0)
        n = per_role_total[role]
        print(f"  {role:12s}: {hits}/{n} = {hits / n:.1%}")


if __name__ == "__main__":
    main()
