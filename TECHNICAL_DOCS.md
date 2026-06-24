# Technical Documentation

## 1. Dataset analysis

Source: `data/raw/products.csv`, `data/raw/outfits.csv`, `data/raw/images/`
(also mirrored in `data/raw/curated25.xlsx`), cloned from the assignment's
dataset repository.

**Products: 68 items, 47 distinct `category_label` values, 64 brands,
price range Rs 270 - Rs 7,799.**

| Dimension | Breakdown |
|---|---|
| Gender | women 41, men 27 |
| Occasion | casual 15, party 13, office 12, festive 9, wedding 6, sports 5, vacation 4, winter 4 |
| Derived role (this project's mapping) | footwear 17, onepiece 13, accessory 13, topwear 12, bottomwear 9, layer 4 |
| Wear type | western 31, ethnic 7 (+ footwear/accessory categories which span both) |
| Color detected in text | 63 of 68 (5 items, mostly watches/clutches/tshirts, have no color word in their text fields) |

**Outfits: 25 stylist-curated combinations.** Each row specifies a hero
item plus up to five complementary slots (second, layer, footwear,
accessory_1, accessory_2), a `theme`, a `palette` description, an
`items_count`, a `total_price_inr`, and a free-text `stylist_rationale` --
this rationale field is effectively a small set of human-written
explanations, and it directly informed the *style* (not the content) of
the template explanations in `src/explainability.py`.

**Key observations that shaped the design:**
- The catalog is small and uneven across role x occasion x gender cells
  (e.g. zero women's footwear tagged "wedding"). Any hard filter-then-fail
  design would frequently return nothing. This is why
  `OutfitRecommender.filter_pool` and `_fill_roles` both implement
  progressive relaxation instead of strict filtering (see
  `ARCHITECTURE.md`).
- Only 4 "layer" items exist for the whole catalog (3 of which are
  gender-specific), so layering recommendations are necessarily thin --
  documented as a limitation below rather than papered over.
- `gender` is the single most decisive feature for compatibility (see the
  learned feature weights below), which matches intuition: the curated
  outfits never mix genders, so the model picked that up immediately.

Run `python -m src.data_loader` to reproduce this summary, or see the
"Dataset Insights" tab in the running app for live charts.

## 2. Category -> role mapping

Every one of the 47 `category_label` values is mapped to exactly one of
six outfit roles in `src/config.py::CATEGORY_TO_ROLE`. The mapping is a
plain dict, not inferred, because (a) with only 47 categories this is more
reliable than a heuristic, and (b) it makes the system's structural
assumptions auditable and trivially extensible -- adding a new category to
a larger catalog later is a one-line addition to that dict, and
`data_loader.load_products()` raises loudly if a category is left
unmapped rather than silently dropping it.

## 3. Color handling

There is no explicit `color` column in `products.csv`. Color is extracted
by regex-matching a curated 38-word lexicon (`config.COLOR_LEXICON`)
against each product's name + description + tags, then bucketed into
`neutral` / `warm` / `cool` families (`config.NEUTRAL_COLORS`,
`WARM_COLORS`, `COOL_COLORS`) for the `color_harmony` feature:
- neutral-involving or unknown-color pairs score 0.8 (neutrals pair with
  almost anything; an undetected color defaults to this same lenient
  score rather than penalizing the pair)
- same-family pairs (warm+warm, cool+cool) score 0.9 (analogous palette)
- cross-family pairs (warm+cool) score 0.6 (deliberate contrast --
  scored lower than analogous, but still a usable outfit, not a penalty)

This is a deliberately simple heuristic, not real color-theory (no actual
hue/saturation extraction from pixels). A production version would pull
the dominant color directly from the product image (e.g. k-means on
non-background pixels, or reading it straight from a CLIP/FashionCLIP
embedding) rather than parsing a text description that might not mention
color at all (5 of 68 products don't).

## 4. The compatibility model

### Method
See `ARCHITECTURE.md` for the pipeline diagram. In short: positive
training pairs are every two items that co-occur in one of the 25 curated
outfits; negative pairs are sampled non-co-occurring pairs (half
same-gender "hard" negatives, half any-gender "easy" negatives, at a
3:1 negative:positive ratio). Eight features per pair feed a
`LogisticRegression(class_weight="balanced")`:

| Feature | What it captures |
|---|---|
| `role_score` | Hard rule from `config.VALID_ROLE_PAIRS` (e.g. topwear+bottomwear=1.0, footwear+accessory=0.5) |
| `same_gender` | Binary |
| `same_wear_type` | western vs ethnic |
| `same_occasion` | Binary |
| `color_harmony` | See section 3 |
| `same_brand` | Binary |
| `text_sim` | TF-IDF cosine similarity between the two items' text_blob |
| `price_tier_diff` | `abs(log10(price_a) - log10(price_b))` |

### Results (reproducible -- re-run `python scripts/build_index.py` to verify)
- Training table: 124 positive pairs, 372 negative pairs (3:1 ratio)
- Held-out AUC: **0.870** (25% stratified split, `random_state=42`)
- Learned feature weights (logistic regression coefficients):

  | Feature | Weight | Interpretation |
  |---|---|---|
  | `same_gender` | **+3.77** | by far the strongest signal -- curated outfits never mix genders |
  | `role_score` | +0.92 | valid role pairings are clearly favored |
  | `same_occasion` | +0.84 | items tagged for the same occasion tend to be paired |
  | `color_harmony` | +0.21 | weak positive -- confirms the heuristic correlates with real stylist choices, though not strongly |
  | `same_brand` | +0.31 | weak positive, plausible but not load-bearing |
  | `text_sim` | +0.15 | weak positive -- text similarity alone is a weak signal here, expected since the role/occasion/gender features already capture most of what makes two items compatible |
  | `same_wear_type` | -0.58 | slightly negative -- a handful of curated outfits intentionally cross western/ethnic (e.g. a Nehru jacket over a western shirt), so exact wear-type match isn't a strong requirement |
  | `price_tier_diff` | -0.56 | items of similar price tier are preferred, as expected |

- Leave-one-out Recall@3 (`python scripts/evaluate.py`): **73.1%** overall
  across 93 held-out role/item slots, with per-role detail:

  | Role | Recall@3 |
  |---|---|
  | bottomwear | 100.0% (12/12) |
  | layer | 100.0% (4/4) |
  | topwear | 76.9% (10/13) |
  | onepiece | 69.2% (9/13) |
  | accessory | 65.4% (17/26) |
  | footwear | 64.0% (16/25) |

  Read this as a sanity check that the model learned the training signal
  at all, **not** as a held-out generalization estimate -- the leave-one-out
  procedure evaluates against the same 25 outfits some of whose pairs
  were used as positive training examples. A true generalization estimate
  would require curated outfits the model never saw during training,
  which a 25-outfit dataset is too small to split meaningfully without
  destroying the (already small) training set.

### Why this is explicitly a proof-of-concept
With ~80 positive pairs from 25 outfits, this model is illustrative, not
production-grade. A real deployment would need:
- Thousands of curated outfits (or large-scale weak supervision from
  co-purchase / co-view data) for a genuine train/val/test split
- Likely a learned embedding space (e.g. a small contrastive/Siamese
  network over CLIP embeddings) instead of 8 hand-engineered features,
  once there's enough data to support it without overfitting
- Calibration analysis -- `predict_proba` here is a relative ranking
  signal, not a calibrated real-world "probability a human would pick
  this pairing"

## 5. NLU: rule-based vs. LLM-assisted

The default rule-based parser (`src/nlu.py::parse_rule_based`) handles
every example query in the assignment brief via keyword/regex matching
against lexicons in `config.py`. It is deliberately simple and fully
auditable. One real bug surfaced and fixed during development is worth
calling out explicitly: a naive substring check (`"man" in "I am a
woman"`) mis-tagged "woman" as gender=`men`, because "man" is literally a
substring of "woman". Fixed by switching to word-boundary regex matching
(`\bman\b` does not match inside "woman"); a regression test for this
specific trap is in `tests/test_recommender.py::
test_gender_parsing_handles_substring_traps`.

The optional LLM-assisted parser (`src/nlu.py::parse_with_llm`) asks the
configured LLM for the same structured JSON schema and falls back to the
rule-based parser on any exception (timeout, malformed JSON, missing key,
etc.) -- the conversational assistant never hard-fails due to an LLM
outage.

## 6. Explainability design

Every recommendation's reasoning (`src/explainability.py`) is built
directly from the same features the compatibility model scored on --
color family match, occasion match, role fit -- rather than being a
free-floating LLM gloss disconnected from the actual scoring. This was a
deliberate choice given the assignment's explicit "Explainability" success
criterion: an explanation should be *faithful* to why the system made
that recommendation, not just plausible-sounding. The optional LLM polish
step (`explainability.py::polish_with_llm`) is constrained by its system
prompt to rewrite the existing reasoning into better prose without
introducing new claims about items or colors.

## 7. Known limitations and future improvements

- **Catalog size and coverage.** 68 products is enough to build and
  validate the full pipeline end-to-end, but several role x occasion x
  gender cells are thin or empty (notably layering items, and
  occasion-specific footwear). The progressive-relaxation fallback logic
  (`ARCHITECTURE.md`, section 2) handles this gracefully for the existing
  data, but the *right* fix is more data, not more fallback logic.
- **Color extraction from text, not pixels.** See section 3 -- a real
  visual color extraction (from the image itself) would be both more
  accurate and would cover the 5 items with no color word in their text.
- **Compatibility model is a small-data proof of concept.** See section 4.
- **No personalization memory across sessions.** `UserProfile` persists
  within a chat session (`ChatAssistant`) but isn't saved across runs of
  the app; a real product would persist this per logged-in user.
- **Single-language, India-centric catalog and lexicon.** Occasion/style
  keyword lists and color vocabulary are tuned to this dataset's English,
  India-market product descriptions (sarees, kurta sets, sherwanis,
  juttis); extending to other markets would mean extending the lexicons
  in `config.py`, not changing any pipeline code.
- **CLIP/FashionCLIP path is implemented but not benchmarked here.** The
  `ClipEmbedder` class is real, working code, but this submission's
  default run (and the numbers in section 4) use the TF-IDF backend, since
  the assignment's evaluation environment may not have reliable access to
  download model weights. A natural next step is benchmarking retrieval
  quality (e.g. via the same leave-one-out procedure, adapted to retrieval
  rather than pairwise scoring) for TF-IDF vs. CLIP vs. FashionCLIP
  embeddings on a larger catalog where the visual signal would matter more
  (this 68-item catalog's text descriptions are detailed enough that the
  two are unlikely to differ hugely here).
- **No outfit diversity / re-ranking beyond category de-duplication.**
  `recommend_from_query` skips a hero candidate if its `category_label`
  was already used in this result set, which is a simple proxy for
  diversity; a more thorough approach (e.g. maximal marginal relevance
  over the full outfit, not just the hero) would help once the catalog is
  large enough for it to matter.
- **Graph-based recommendation, RAG over stylist rationales, and a true
  pairwise ranking network (vs. pointwise logistic regression) are exactly
  the "Advanced Approaches" bonus items the assignment lists** that were
  consciously deferred in favor of a complete, well-tested core system
  within the 48-hour window, rather than a partially-working version of
  everything.
