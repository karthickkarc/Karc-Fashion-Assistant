"""User profile representation used to bias recommendations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserProfile:
    gender: str | None = None       # "men" | "women" | None
    age: int | None = None
    occasion: str | None = None
    style: str | None = None        # "western" | "ethnic" | None
    color_preference: str | None = None

    def age_band(self) -> str | None:
        if self.age is None:
            return None
        if self.age < 25:
            return "youth"        # leans bolder colors, trend pieces
        if self.age < 40:
            return "young_adult"  # smart casual default
        return "mature"           # leans classic, neutral, tailored

    def style_bias(self) -> dict:
        """Small, transparent nudges applied as tie-breakers during
        ranking -- not hard filters, so the system never *hides* an
        otherwise great match just because of age, it only nudges order."""
        band = self.age_band()
        if band == "youth":
            return {"prefer_neutral": False, "prefer_classic_brand_tier": False}
        if band == "mature":
            return {"prefer_neutral": True, "prefer_classic_brand_tier": True}
        return {"prefer_neutral": False, "prefer_classic_brand_tier": False}

    def to_dict(self) -> dict:
        return {
            "gender": self.gender,
            "age": self.age,
            "occasion": self.occasion,
            "style": self.style,
            "color_preference": self.color_preference,
        }
