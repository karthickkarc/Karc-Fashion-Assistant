"""
Explainability: turns a chosen pair/outfit + its compatibility features
into a human-readable reason, in the style shown in the assignment
("Beige chinos pair well with a navy blazer because...").

`explain_pair()` is fully template-based (no LLM needed) and grounded in
the actual features the compatibility model used -- color harmony, role
fit, occasion match -- so the explanation is faithful to *why the system
scored it that way*, not a generic LLM gloss. `polish_with_llm()` is an
optional pass that asks an LLM to rewrite the templated bullets into
flowing prose, used only when an LLM client is configured.
"""
from __future__ import annotations

import pandas as pd


def _color_phrase(row: pd.Series) -> str:
    return f"{row['color']} " if isinstance(row.get("color"), str) else ""


def explain_pair(item_a: pd.Series, item_b: pd.Series, feats: dict) -> str:
    name_a = f"{_color_phrase(item_a)}{item_a['category_label'].lower()}"
    name_b = f"{_color_phrase(item_b)}{item_b['category_label'].lower()}"

    reasons = []
    if feats["color_harmony"] >= 0.85:
        reasons.append("their colors sit in the same easy-to-pair family")
    elif feats["color_harmony"] >= 0.7:
        reasons.append("a neutral tone keeps the palette balanced")
    else:
        reasons.append("the contrast in tone keeps the look from feeling flat")

    if feats["same_occasion"]:
        reasons.append("both fit the same occasion")
    if feats["role_score"] >= 0.9:
        reasons.append("the silhouettes complete a full head-to-toe look")

    reason_text = "; ".join(reasons)
    return f"{name_a.capitalize()} pairs well with {name_b} because {reason_text}."


def explain_outfit(items: dict[str, pd.Series], occasion: str | None,
                    avg_compat: float) -> str:
    """`items` maps role -> product row for the assembled outfit."""
    parts = []
    ordered_roles = ["onepiece", "topwear", "bottomwear", "layer", "footwear", "accessory"]
    descriptors = []
    for role in ordered_roles:
        if role in items:
            row = items[role]
            descriptors.append(f"{_color_phrase(row)}{row['category_label'].lower()}")

    occasion_phrase = f"for a {occasion} occasion" if occasion else "for your occasion"
    body = ", ".join(descriptors[:-1]) + (f" and {descriptors[-1]}" if len(descriptors) > 1 else descriptors[0])
    parts.append(f"This look combines {body} {occasion_phrase}.")

    if avg_compat >= 0.75:
        parts.append("The pieces share a coherent palette and matching formality, so they read as a deliberate, put-together outfit.")
    elif avg_compat >= 0.5:
        parts.append("The pieces are compatible in formality and occasion, with a deliberate color contrast for visual interest.")
    else:
        parts.append("This is a more experimental pairing -- it satisfies the occasion and role requirements, but the color match is looser than the system's top picks.")

    return " ".join(parts)


def polish_with_llm(template_explanation: str, llm_client, occasion: str | None) -> str:
    if llm_client is None:
        return template_explanation
    system = (
        "You are a fashion stylist. Rewrite the given outfit reasoning in 2-3 warm, "
        "confident sentences for a customer. Keep every factual claim from the input "
        "(items, colors, occasion) -- do not invent new items or colors. No markdown."
    )
    user = f"Occasion: {occasion or 'general'}\nReasoning notes: {template_explanation}"
    try:
        return llm_client.complete(system, user).strip()
    except Exception:
        return template_explanation
