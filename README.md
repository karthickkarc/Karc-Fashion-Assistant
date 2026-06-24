# Karc-Fashion-Assistant
# Karc -- AI Fashion Outfit Recommendation System

Built for the **Karc XAI Machine Learning & AI Engineer Intern Assignment**.

A conversational, explainable outfit recommendation system over a 68-product /
25-curated-outfit fashion catalog. Given a free-text request ("I need an
outfit for a business meeting") or a single anchor item ("white formal
shirt"), it assembles a complete, role-correct outfit (topwear + bottomwear
+ footwear, or a one-piece + footwear, plus optional layer/accessory),
ranks candidates with a **learned pairwise compatibility model**, retrieves
relevant items with a **FAISS vector index**, and explains every
recommendation in a grounded "Stylist's Note."

The whole pipeline runs **fully offline with zero API keys** by default
(TF-IDF embeddings + rule-based NLU + template explanations), with optional,
clearly-isolated upgrade paths to FashionCLIP embeddings and an LLM
(Gemini / OpenAI / Claude) for richer conversational parsing and prose.

---

## What's implemented, mapped to the assignment brief

| Requirement | Where |
|---|---|
| Dataset analysis | `src/data_loader.py::summarize_dataset`, "Dataset Insights" tab in the app, `TECHNICAL_DOCS.md` |
| Outfit Compatibility Engine | `src/compatibility_model.py` (learned) + `src/recommender.py::recommend_from_item` |
| Ranking by compatibility/relevance | `CompatibilityModel.predict_proba` x role-gate, sorted in `recommend_from_query` |
| User-aware recommendations (gender/age/occasion/style) | `src/user_profile.py`, `OutfitRecommender.filter_pool` |
| Conversational Fashion Assistant | `src/nlu.py`, `src/chat_assistant.py`, chat tab in `app.py` |
| Explainability on every recommendation | `src/explainability.py` (template, grounded in real features; optional LLM polish) |
| Computer Vision / image embeddings (bonus) | `src/embeddings.py::ClipEmbedder` (FashionCLIP-ready, opt-in) |
| Vector database / similarity search | `src/vector_store.py` (FAISS `IndexFlatIP`) |
| Learning compatibility scores / pairwise ranking (bonus) | `src/compatibility_model.py` -- logistic regression trained on the 25 curated outfits |
| LLM integration | `src/llm_client.py` -- pluggable Gemini / OpenAI / Anthropic, all optional |

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd dare-xai-fashion-assistant

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Build the offline artifacts (embeddings, FAISS index, compatibility model)
python scripts/build_index.py

# 4. Run the working prototype
streamlit run app.py
```

That's it -- no API key, no GPU, no internet access required for the steps
above (the dataset and images ship inside `data/raw/`).

### Recording the demo video? Run `demo.py`

```bash
python demo.py --pause
```

This runs the **entire pipeline end-to-end in one script**, in the same
order as the architecture diagram: dataset analysis -> offline indexing
(embeddings, FAISS, compatibility model training) -> evaluation ->
item-anchored recommendations -> conversational assistant -> summary.
Every number it prints is computed live, not pre-baked. `--pause` waits
for Enter between sections so you can narrate over each one without the
output racing ahead while recording.

Optional: run the REST API instead of / alongside Streamlit:
```bash
uvicorn api:app --reload --port 8000   # docs at http://localhost:8000/docs
```

Optional: enable an LLM for richer conversational parsing and prose
explanations (see `.env.example`):
```bash
cp .env.example .env
# edit .env: LLM_PROVIDER=gemini  and  GEMINI_API_KEY=...
pip install google-generativeai
```

Run the tests and the quantitative sanity-check evaluation:
```bash
pytest tests/ -v
python scripts/evaluate.py
```

---

## Project structure

```
karc-xai-fashion-assistant/
|-- app.py                     # Streamlit working prototype (chat + item explorer + dataset insights)
|-- api.py                     # Optional FastAPI REST interface over the same recommender
|-- requirements.txt
|-- requirements-clip.txt      # optional: enables the FashionCLIP/CLIP embedding backend
|-- .env.example
|-- src/
|   |-- config.py              # paths, category->role map, role-compatibility rules, lexicons
|   |-- data_loader.py         # load + enrich products.csv/outfits.csv, dataset summary stats
|   |-- embeddings.py          # TF-IDF (default) and CLIP/FashionCLIP (optional) embedders
|   |-- vector_store.py        # FAISS IndexFlatIP wrapper
|   |-- compatibility_model.py # pairwise training-table construction + LogisticRegression model
|   |-- user_profile.py        # UserProfile dataclass
|   |-- nlu.py                 # rule-based intent parser + optional LLM-assisted parser
|   |-- llm_client.py          # pluggable Gemini / OpenAI / Anthropic client, or None
|   |-- explainability.py      # template reasoning + optional LLM polish
|   |-- recommender.py         # OutfitRecommender: the main orchestration class
|   |-- chat_assistant.py      # multi-turn conversational wrapper (profile memory)
|   `-- json_safe.py           # NaN/numpy -> JSON-safe dict conversion
|-- scripts/
|   |-- build_index.py         # offline pipeline: enrich -> embed -> index -> train -> persist
|   `-- evaluate.py            # leave-one-out Recall@3 sanity check against curated outfits
|-- tests/
|   `-- test_recommender.py    # 17 tests: data integrity, model behavior, NLU, end-to-end recs
|-- data/
|   |-- raw/                   # products.csv, outfits.csv, curated25.xlsx, images/ (provided dataset)
|   `-- artifacts/             # generated by build_index.py (vectorizer, FAISS index, model)
`-- docs/
    |-- architecture.gv             # Graphviz source for the architecture diagram
    `-- architecture_diagram.svg    # rendered diagram
```

See **`ARCHITECTURE.md`** for the system design walkthrough and
**`TECHNICAL_DOCS.md`** for dataset analysis, modeling decisions, evaluation
results, and known limitations.

---

## Example interactions

**Conversational:**
> "I need an outfit for a business meeting."
> "I am a 22-year-old male looking for a casual summer outfit."
> "I am attending a wedding next weekend, I'm a woman."

**Item-anchored** (`recommend_from_item`, the "Outfit Compatibility Engine"
example from the brief):
> Input: *Cotton Slim Fit Formal Shirt* (white, men's, office)
> Output: Slim Fit Formal Trousers + Slip-On Formal Shoes, each with a
> reason such as *"White formal shirt pairs well with brown formal shoes
> because a neutral tone keeps the palette balanced; both fit the same
> occasion."*

---

## Configuration reference

All configuration is environment-variable based (`.env`, see
`.env.example`); every variable has a working, key-free default.

| Variable | Default | Effect |
|---|---|---|
| `EMBEDDING_BACKEND` | `tfidf` | `tfidf` (offline) / `clip` / `fashionclip` (bonus, needs `requirements-clip.txt` + internet on first run) |
| `LLM_PROVIDER` | `none` | `none` (rule-based NLU + templates) / `gemini` / `openai` / `anthropic` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | empty | API key for the selected provider |
