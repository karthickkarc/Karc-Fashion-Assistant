# Architecture

See `docs/architecture_diagram.svg` for the visual version of everything
described below (source: `docs/architecture.gv`, rendered with Graphviz).

The system is two pipelines sharing one set of modules: an **offline
indexing pipeline** that runs once (or whenever `data/raw/*.csv` changes),
and an **online inference pipeline** that runs on every chat turn or API
call. Splitting it this way means inference never pays the cost of
re-fitting a vectorizer or retraining a model -- it only ever does cheap
vector lookups and a handful of logistic-regression scores.

## 1. Offline indexing pipeline (`scripts/build_index.py`)

```
products.csv, outfits.csv, images/
        |
        v
Data Loader & Enrichment  (src/data_loader.py)
  - maps each of the 47 category_label values to one of 6 outfit roles
    (topwear / bottomwear / onepiece / footwear / layer / accessory)
    via a config table -- this is the hard structural backbone everything
    else builds on
  - extracts a color word from name+description+tags via a curated
    lexicon, and buckets it into neutral/warm/cool for color-harmony scoring
  - builds a combined text_blob per product (name + category + occasion +
    wear_type + description + tags) for embedding
        |
        +----------------------------+
        v                            v
   Embedder                    Training-table builder
 (src/embeddings.py)          (src/compatibility_model.py)
  - default: TF-IDF              - positive pairs: every two items that
    (2048 features, 1-2 grams)     co-occur in the same curated outfit
  - optional: CLIP/FashionCLIP   - negative pairs: sampled non-co-occurring
    (image+text, see below)        pairs, half same-gender ("hard"), half
        |                           any-gender ("easy")
        v                         - 8 features per pair (see below)
   FAISS index                       |
 (src/vector_store.py)               v
  IndexFlatIP over L2-          Logistic Regression
  normalized vectors            (class_weight="balanced")
        |                            |
        +-------------+--------------+
                       v
                 data/artifacts/
       (vectorizer.joblib, product_index.faiss, product_id_map.joblib,
        compatibility_model.joblib, products_enriched.csv)
```

**Why TF-IDF by default, with CLIP/FashionCLIP as an opt-in:** the
assignment's evaluation environment may not have GPU or guaranteed internet
access to download model weights, and the take-home is graded on
understanding, not on having the trendiest stack. TF-IDF over rich text
metadata (name, category, occasion, description, tags) is fast, fully
deterministic, needs zero downloads, and -- on a 68-item catalog with
detailed text fields -- is a perfectly reasonable retrieval signal. The
`ClipEmbedder` class in `src/embeddings.py` is a complete, real
implementation (joint image+text embedding via `transformers`), gated
behind `EMBEDDING_BACKEND=clip|fashionclip` in `.env`, so the "bonus" path
described in the assignment exists and is one config line away, without
forcing it on by default or on the grading environment.

**Why a learned compatibility model and not just rules or an LLM:** the
assignment explicitly calls out "Learning Compatibility Scores" /
"Pairwise Ranking Models" as a deeper-ML path it wants to see, as opposed
to relying solely on prompt engineering. With only 25 curated outfits to
learn from, a deep model would overfit instantly, so this uses the
simplest model that can plausibly generalize: logistic regression over 8
hand-engineered features (`role_score`, `same_gender`, `same_wear_type`,
`same_occasion`, `color_harmony`, `same_brand`, `text_sim`,
`price_tier_diff`). It reaches 0.87 held-out AUC on this dataset (see
`TECHNICAL_DOCS.md` for the full evaluation) and -- importantly for the
explainability requirement -- every one of its inputs is itself a
human-readable fact ("same occasion", "neutral color"), so the model's
score can always be unpacked into the features that produced it.

## 2. Online inference pipeline (every chat turn / API call)

```
User query + optional profile (gender, age, occasion, style)
        |
        v
NLU / Intent Parser  (src/nlu.py)
  - rule-based by default: regex/keyword matching against lexicons in
    config.py for occasion, gender, style, color, age, and a specific
    item mention ("white shirt")
  - if LLM_PROVIDER is set, asks the LLM for the same structured JSON and
    falls back to the rule-based parser on any failure
        |
        v
Profile merge  (src/chat_assistant.py)
  - new explicit fields overwrite the running UserProfile; unspecified
    fields keep whatever was set in earlier turns
        |
        v
Candidate Pool Filter  (OutfitRecommender.filter_pool)
  - filters the catalog by gender / occasion / style, but only applies
    each filter if it doesn't empty the pool -- a small catalog means an
    over-specific filter (e.g. "women + wedding + western") can easily
    have zero matches, so filters are applied defensively, not blindly
        |
        v
Semantic Retrieval  (FAISS search over the TF-IDF embedding of the query)
  - if the user named a specific garment ("white shirt"), search for that
    phrase and anchor on the best match
  - otherwise, search the full query text and take the top onepiece/
    topwear hits that survive the pool filter as hero candidates
        |
        v
Outfit Assembly  (OutfitRecommender.assemble_outfit_around)
  - mandatory roles for the anchor's type (topwear -> [bottomwear,
    footwear]; onepiece -> [footwear]) are filled by scoring every
    same-gender candidate in that role against everything chosen so far
    with the compatibility model, gated by config.VALID_ROLE_PAIRS
  - if the occasion-filtered pool has nothing for a mandatory role (e.g.
    no women's footwear tagged "wedding" in this small catalog), the
    search widens to the full same-gender catalog for that role only --
    better an outfit with a slightly-off-occasion shoe than a missing shoe
  - optional roles (layer, accessory) do NOT get this widening: they're
    only added if an occasion-appropriate candidate actually scores above
    threshold, so a beach outfit never gets an office blazer just because
    it's the only same-gender layer item in the catalog
        |
        v
Ranking -- sort assembled outfit candidates by average pairwise
compatibility across all chosen items
        |
        v
Explainability  (src/explainability.py)
  - template reasoning built directly from the features that produced the
    score (color harmony, occasion match, role fit) -- not a generic LLM
    gloss, so the explanation is faithful to *why the system ranked it
    there*
  - optionally polished into flowing prose by an LLM if configured,
    constrained to not introduce new claims
        |
        v
Output -- Streamlit chat UI (outfit cards: images, role tags, price,
compatibility %, "Stylist's Note") or FastAPI JSON (same data, for any
other client)
```

## Module responsibilities at a glance

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth: paths, category->role map, role-compatibility matrix, NLU lexicons |
| `data_loader.py` | Load + enrich the dataset; dataset-level summary statistics |
| `embeddings.py` | Text/image -> vector, swappable backend |
| `vector_store.py` | FAISS index build/search/persist |
| `compatibility_model.py` | Build pairwise training data from curated outfits; train/score the compatibility model |
| `user_profile.py` | Structured representation of who's shopping |
| `nlu.py` | Free text -> structured intent |
| `llm_client.py` | One abstraction over Gemini/OpenAI/Anthropic, or `None` |
| `explainability.py` | Score/features -> human-readable reasoning |
| `recommender.py` | Orchestrates everything above into `recommend_from_item` / `recommend_from_query` |
| `chat_assistant.py` | Multi-turn state (profile memory) on top of the recommender |
| `json_safe.py` | Pandas/numpy values -> values the strict JSON encoder will accept |

## Why this decomposition

Every box in the diagram is independently testable and independently
swappable:
- Swap the embedding backend (`tfidf` -> `fashionclip`) without touching
  `vector_store.py`, `recommender.py`, or anything downstream -- they only
  depend on "a vector came out", not on how.
- Swap the LLM provider, or turn it off entirely, without touching
  `nlu.py`'s or `explainability.py`'s calling code -- they depend on the
  one-method `LLMClient.complete()` interface, not on any provider SDK.
- Swap the UI (Streamlit today; the included `api.py` shows a second,
  independent client) without touching the recommendation logic at all --
  both just call `OutfitRecommender.recommend_from_item` /
  `recommend_from_query`.

This was a deliberate trade against a faster, more tangled "everything in
one notebook" build: it costs a bit more boilerplate up front, but it is
the same shape you would want if this dataset grew from 68 products to
68,000 and the embedding/LLM choices needed to change independently of the
ranking and assembly logic.
