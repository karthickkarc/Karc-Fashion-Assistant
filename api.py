"""
Optional REST API over the same OutfitRecommender used by app.py --
demonstrates the engine is not tied to one UI; a mobile app, a Slack bot,
or another service could call this instead of Streamlit.

Run with:   uvicorn api:app --reload --port 8000
Docs at:    http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.json_safe import safe_dict
from src.recommender import OutfitRecommender
from src.user_profile import UserProfile

app = FastAPI(title="Dare XAI Fashion Assistant API", version="1.0")
_recommender: OutfitRecommender | None = None


def get_recommender() -> OutfitRecommender:
    global _recommender
    if _recommender is None:
        _recommender = OutfitRecommender.load_default()
    return _recommender


class ProfileIn(BaseModel):
    gender: str | None = None
    age: int | None = None
    occasion: str | None = None
    style: str | None = None


class QueryIn(BaseModel):
    query: str
    profile: ProfileIn | None = None
    top_n_outfits: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/products")
def list_products(gender: str | None = None, role: str | None = None, occasion: str | None = None):
    rec = get_recommender()
    df = rec.products
    if gender:
        df = df[df["gender"] == gender]
    if role:
        df = df[df["role"] == role]
    if occasion:
        df = df[df["occasion"] == occasion]
    cols = ["id", "name", "category_label", "role", "gender", "color", "occasion", "price_inr"]
    return [safe_dict(r) for _, r in df[cols].iterrows()]


@app.get("/recommend/item/{item_id}")
def recommend_from_item(item_id: str, top_k_per_role: int = 3):
    rec = get_recommender()
    try:
        return rec.recommend_from_item(item_id, top_k_per_role=top_k_per_role)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/recommend/query")
def recommend_from_query(payload: QueryIn):
    rec = get_recommender()
    profile = UserProfile(**payload.profile.dict()) if payload.profile else UserProfile()
    outfits = rec.recommend_from_query(payload.query, profile, top_n_outfits=payload.top_n_outfits)
    return {"query": payload.query, "outfits": outfits}
