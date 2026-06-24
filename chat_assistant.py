"""
Thin conversational layer on top of OutfitRecommender: keeps a running
UserProfile across turns (so "I'm a 24 year old man" in turn 1 still
applies in turn 3) and formats results as chat-friendly text + structured
data for the UI to render as outfit cards.
"""
from __future__ import annotations

from .recommender import OutfitRecommender
from .user_profile import UserProfile
from .nlu import parse_query


class ChatAssistant:
    def __init__(self, recommender: OutfitRecommender):
        self.recommender = recommender
        self.profile = UserProfile()
        self.history: list[dict] = []

    def _merge_profile(self, query: str):
        intent = parse_query(query, llm_client=None)  # cheap rule pass just to update profile memory
        if intent.gender:
            self.profile.gender = intent.gender
        if intent.age:
            self.profile.age = intent.age
        if intent.occasion:
            self.profile.occasion = intent.occasion
        if intent.style:
            self.profile.style = intent.style
        if intent.color:
            self.profile.color_preference = intent.color

    def ask(self, query: str, top_n_outfits: int = 3) -> dict:
        self._merge_profile(query)
        outfits = self.recommender.recommend_from_query(query, self.profile, top_n_outfits=top_n_outfits)

        if not outfits:
            reply = ("I couldn't find a complete outfit for that request in the current "
                      "catalog -- could you tell me the occasion or gender you're shopping for?")
        else:
            reply = f"Here are {len(outfits)} outfit option(s) for you:"

        turn = {"query": query, "profile": self.profile.to_dict(), "outfits": outfits, "reply": reply}
        self.history.append(turn)
        return turn
