"""
Main recommendation orchestration.

Two entry points map directly onto the assignment's two example flows:

  * `recommend_from_item(item_id)`      -- "Outfit Compatibility Engine":
        given a single item, return ranked compatible items per role.
  * `recommend_from_query(query, profile)` -- "Conversational Fashion
        Assistant": parse free text + user profile, retrieve relevant
        candidates, assemble 1-3 complete outfits, rank, and explain.

Both reuse the same building blocks: FAISS retrieval for "what's
semantically relevant to this text", the learned CompatibilityModel for
"do these two specific items go together", and config.VALID_ROLE_PAIRS as
the hard structural gate ("a complete outfit needs a top, a bottom, and
shoes").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .compatibility_model import CompatibilityModel, pair_features
from .data_loader import load_products, load_outfits
from .embeddings import TfidfEmbedder
from .explainability import explain_outfit, explain_pair, polish_with_llm
from .json_safe import safe_dict
from .nlu import ParsedIntent, parse_query
from .user_profile import UserProfile
from .vector_store import ProductVectorStore

MANDATORY_BY_ANCHOR_ROLE = {
    config.ROLE_TOPWEAR: [config.ROLE_BOTTOMWEAR, config.ROLE_FOOTWEAR],
    config.ROLE_BOTTOMWEAR: [config.ROLE_TOPWEAR, config.ROLE_FOOTWEAR],
    config.ROLE_ONEPIECE: [config.ROLE_FOOTWEAR],
}
OPTIONAL_ROLES = [config.ROLE_LAYER, config.ROLE_ACCESSORY]


class OutfitRecommender:
    def __init__(self, products: pd.DataFrame, outfits: pd.DataFrame,
                 embedder: TfidfEmbedder, vector_store: ProductVectorStore,
                 compat_model: CompatibilityModel, llm_client=None):
        self.products = products.set_index("id", drop=False)
        self.outfits = outfits
        self.embedder = embedder
        self.vector_store = vector_store
        self.compat_model = compat_model
        self.llm_client = llm_client

    # ---------------------------------------------------------- factory
    @classmethod
    def load_default(cls) -> "OutfitRecommender":
        from .llm_client import get_llm_client
        products = load_products()
        outfits = load_outfits()
        embedder = TfidfEmbedder.load()
        vector_store = ProductVectorStore.load()
        compat_model = CompatibilityModel.load()
        return cls(products, outfits, embedder, vector_store, compat_model,
                    llm_client=get_llm_client())

    # ---------------------------------------------------------- helpers
    def _text_sim(self, a_id: str, b_id: str) -> float:
        vecs = self.embedder.transform(
            [self.products.loc[a_id, "text_blob"], self.products.loc[b_id, "text_blob"]]
        )
        denom = np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        return float(np.dot(vecs[0], vecs[1]) / denom) if denom else 0.0

    def _score_pair(self, a_id: str, b_id: str) -> tuple[float, dict]:
        feats = pair_features(self.products.loc[a_id], self.products.loc[b_id],
                               self._text_sim(a_id, b_id))
        prob = self.compat_model.predict_proba(feats)
        return prob * feats["role_score"], feats

    def filter_pool(self, gender: str | None = None, occasion: str | None = None,
                     style: str | None = None) -> pd.DataFrame:
        pool = self.products
        if gender:
            pool = pool[pool["gender"] == gender]
        if occasion and (pool["occasion"] == occasion).any():
            pool = pool[pool["occasion"] == occasion]  # only apply if it doesn't empty the pool
        if style and (pool["wear_type"] == style).any():
            pool = pool[pool["wear_type"] == style]
        return pool

    # --------------------------------------------------- item -> outfit
    def recommend_from_item(self, item_id: str, top_k_per_role: int = 3) -> dict:
        if item_id not in self.products.index:
            raise KeyError(f"Unknown product id: {item_id}")
        anchor = self.products.loc[item_id]
        same_gender_pool = self.products[
            (self.products["gender"] == anchor["gender"]) & (self.products.index != item_id)
        ]

        anchor_role = anchor["role"]
        target_roles = MANDATORY_BY_ANCHOR_ROLE.get(anchor_role, [r for r in config.ALL_ROLES if r != anchor_role])
        target_roles = target_roles + [r for r in OPTIONAL_ROLES if r not in target_roles]

        recommendations = {}
        for role in target_roles:
            role_pool = same_gender_pool[same_gender_pool["role"] == role]
            scored = []
            for cand_id in role_pool.index:
                score, feats = self._score_pair(item_id, cand_id)
                scored.append((cand_id, score, feats))
            scored.sort(key=lambda t: t[1], reverse=True)
            top = scored[:top_k_per_role]
            recommendations[role] = [
                {
                    "item": safe_dict(self.products.loc[cid]),
                    "score": round(score, 3),
                    "reason": explain_pair(anchor, self.products.loc[cid], feats),
                }
                for cid, score, feats in top
            ]
        return {"anchor": safe_dict(anchor), "recommendations": recommendations}

    # ------------------------------------------------ outfit assembly
    def _fill_roles(self, anchor_id: str, roles_needed: list[str], pool: pd.DataFrame,
                     fallback_pool: pd.DataFrame | None = None) -> dict:
        chosen = {self.products.loc[anchor_id, "role"]: anchor_id}
        used = {anchor_id}
        for role in roles_needed:
            role_pool = pool[(pool["role"] == role) & (~pool.index.isin(used))]
            if role_pool.empty and fallback_pool is not None:
                # The occasion-restricted pool has nothing for this role
                # (e.g. no women's footwear tagged "wedding" in a small
                # catalog) -- widen to the full same-gender catalog rather
                # than silently shipping an outfit with a role missing.
                role_pool = fallback_pool[(fallback_pool["role"] == role) & (~fallback_pool.index.isin(used))]
            if role_pool.empty:
                continue
            best_id, best_score = None, -1.0
            for cand_id in role_pool.index:
                scores = [self._score_pair(existing_id, cand_id)[0] for existing_id in chosen.values()]
                avg_score = float(np.mean(scores)) if scores else 0.0
                if avg_score > best_score:
                    best_id, best_score = cand_id, avg_score
            if best_id is not None:
                chosen[role] = best_id
                used.add(best_id)
        return chosen

    def assemble_outfit_around(self, anchor_id: str, pool: pd.DataFrame) -> dict:
        anchor = self.products.loc[anchor_id]
        gendered_pool = pool[pool["gender"] == anchor["gender"]]
        fallback_pool = self.products[self.products["gender"] == anchor["gender"]]

        mandatory = MANDATORY_BY_ANCHOR_ROLE.get(anchor["role"], [])
        chosen_ids = self._fill_roles(anchor_id, mandatory, gendered_pool, fallback_pool=fallback_pool)

        # Optional layer/accessory: only add if an occasion-appropriate
        # candidate exists. Unlike mandatory roles, we deliberately do NOT
        # fall back to the full catalog here -- better to ship a 3-piece
        # outfit than bolt on an office blazer just because it's the only
        # same-gender layer item in a small catalog.
        optional_ids = self._fill_roles(anchor_id, OPTIONAL_ROLES, gendered_pool, fallback_pool=None)
        for role, pid in optional_ids.items():
            if role not in chosen_ids:
                score, _ = self._score_pair(anchor_id, pid)
                if score >= 0.35:
                    chosen_ids[role] = pid

        items = {role: self.products.loc[pid] for role, pid in chosen_ids.items()}
        pair_scores = []
        for role_a, id_a in chosen_ids.items():
            for role_b, id_b in chosen_ids.items():
                if id_a < id_b:
                    score, _ = self._score_pair(id_a, id_b)
                    pair_scores.append(score)
        avg_compat = float(np.mean(pair_scores)) if pair_scores else 0.0
        total_price = float(sum(self.products.loc[pid, "price_inr"] for pid in chosen_ids.values()))

        return {
            "items": items,
            "item_ids": chosen_ids,
            "avg_compat": avg_compat,
            "total_price_inr": total_price,
        }

    # --------------------------------------------------- query -> outfits
    def recommend_from_query(self, query: str, profile: UserProfile | None = None,
                              top_n_outfits: int = 3) -> list[dict]:
        profile = profile or UserProfile()
        intent = parse_query(query, llm_client=self.llm_client if config.LLM_PROVIDER != "none" else None)

        gender = intent.gender or profile.gender
        occasion = intent.occasion or profile.occasion or "casual"
        style = intent.style or profile.style

        pool = self.filter_pool(gender=gender, occasion=occasion, style=style)
        if pool.empty:  # progressively relax if filters over-constrain
            pool = self.filter_pool(gender=gender, occasion=occasion)
        if pool.empty:
            pool = self.filter_pool(gender=gender)
        if pool.empty:
            pool = self.products

        # Semantic retrieval over the *whole* catalog via FAISS, then
        # intersect with the filtered pool to bias hero selection toward
        # what the user actually asked for in free text.
        query_vec = self.embedder.transform([query])[0]
        retrieved = self.vector_store.search(query_vec, top_k=30)
        retrieved_ids = [pid for pid, _ in retrieved if pid in pool.index]

        if intent.item_mentioned:
            # User named a specific garment -- anchor on the best textual
            # match for that phrase rather than a generic hero search.
            item_vec = self.embedder.transform([intent.item_mentioned])[0]
            item_matches = self.vector_store.search(item_vec, top_k=30)
            hero_candidates = [pid for pid, _ in item_matches if pid in pool.index][:top_n_outfits + 2]
        else:
            hero_roles = {config.ROLE_ONEPIECE, config.ROLE_TOPWEAR}
            hero_candidates = [pid for pid in retrieved_ids if pool.loc[pid, "role"] in hero_roles]
            if not hero_candidates:
                hero_candidates = pool[pool["role"].isin(hero_roles)].index.tolist()

        outfits = []
        seen_categories = set()
        for hero_id in hero_candidates:
            if len(outfits) >= top_n_outfits:
                break
            category = pool.loc[hero_id, "category_label"]
            if category in seen_categories:
                continue  # keep results varied rather than 3x the same shirt
            seen_categories.add(category)

            assembled = self.assemble_outfit_around(hero_id, pool)
            template_reason = explain_outfit(assembled["items"], occasion, assembled["avg_compat"])
            reason = polish_with_llm(template_reason, self.llm_client, occasion) \
                if self.llm_client else template_reason

            outfits.append({
                "hero_id": hero_id,
                "items": {role: safe_dict(row) for role, row in assembled["items"].items()},
                "avg_compat": round(assembled["avg_compat"], 3),
                "total_price_inr": assembled["total_price_inr"],
                "reason": reason,
                "intent": intent.__dict__,
            })

        outfits.sort(key=lambda o: o["avg_compat"], reverse=True)
        return outfits
