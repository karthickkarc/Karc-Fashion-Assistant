"""
Working prototype: a chat-based fashion assistant + a single-item
compatibility explorer + a dataset insights view, in one Streamlit app.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.chat_assistant import ChatAssistant
from src.data_loader import load_products, load_outfits, summarize_dataset
from src.recommender import OutfitRecommender
from src.user_profile import UserProfile

st.set_page_config(page_title="Atelier -- AI Outfit Assistant", page_icon="\U0001F9F5", layout="wide")

# --------------------------------------------------------------------------
# Visual identity: deep charcoal-plum surface, warm ivory text, a single
# gold "stylist's thread" accent. Editorial serif (Fraunces) for the voice
# of the stylist, monospace (IBM Plex Mono) for the data the model
# produces (scores, prices, role tags) -- the contrast is the point: this
# product is a machine-learned ranking wearing a human stylist's voice.
# --------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #221F26;
  --surface: #2C2832;
  --surface-2: #34303C;
  --ink: #F3EFE6;
  --ink-soft: #B8AFC2;
  --gold: #C9A227;
  --teal: #5C9EA0;
  --rose: #A6635B;
}
.stApp { background-color: var(--bg); color: var(--ink); }
h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink) !important; letter-spacing: 0.2px; }
body, .stMarkdown, p, label, .stSelectbox, .stTextInput, .stSlider { font-family: 'Inter', sans-serif; }
.role-chip {
  display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--bg);
  background: var(--gold); padding: 2px 8px; border-radius: 3px; margin-bottom: 6px;
}
.outfit-card {
  background: var(--surface); border: 1px solid #443f4d; border-radius: 10px;
  padding: 16px; margin-bottom: 14px;
}
.stylists-note {
  border-left: 3px solid var(--gold); padding: 10px 16px; margin-top: 10px;
  background: var(--surface-2); border-radius: 0 6px 6px 0;
}
.stylists-note .label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.15em;
  color: var(--gold); text-transform: uppercase;
}
.stylists-note p {
  font-family: 'Fraunces', serif; font-style: italic; color: var(--ink); margin: 4px 0 0 0;
  font-size: 1.02rem; line-height: 1.5;
}
.score-track { height: 6px; background: #443f4d; border-radius: 3px; margin-top: 4px; overflow: hidden; }
.score-fill { height: 6px; background: var(--gold); }
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--ink-soft); }
.item-name { font-size: 0.92rem; color: var(--ink); margin: 4px 0 2px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_recommender() -> OutfitRecommender:
    if not config.FAISS_INDEX_PATH.exists():
        st.error("Index artifacts not found. Run `python scripts/build_index.py` first.")
        st.stop()
    return OutfitRecommender.load_default()


def image_path_for(row) -> str | None:
    p = config.RAW_DIR / row["image"]
    return str(p) if p.exists() else None


def render_outfit_card(outfit: dict):
    role_order = ["onepiece", "topwear", "bottomwear", "layer", "footwear", "accessory"]
    items = outfit["items"]
    cols = st.columns(min(len(items), 5) or 1)
    for i, role in enumerate([r for r in role_order if r in items]):
        item = items[role]
        with cols[i % len(cols)]:
            st.markdown(f"<span class='role-chip'>{role}</span>", unsafe_allow_html=True)
            img = image_path_for(item)
            if img:
                st.image(img, use_container_width=True)
            st.markdown(f"<div class='item-name'>{item['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='mono'>Rs {int(item['price_inr'])}</div>", unsafe_allow_html=True)

    pct = int(round(outfit["avg_compat"] * 100))
    st.markdown(
        f"<div class='mono'>Compatibility match &mdash; {pct}%</div>"
        f"<div class='score-track'><div class='score-fill' style='width:{pct}%;'></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="stylists-note"><span class="label">Stylist's Note</span>
        <p>{outfit['reason']}</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='mono' style='margin-top:6px;'>Total: Rs {int(outfit['total_price_inr'])}</div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
st.markdown("## Atelier")
st.markdown(
    "<span class='mono'>AI OUTFIT RECOMMENDATION SYSTEM &mdash; DARE XAI ML/AI ENGINEER ASSIGNMENT</span>",
    unsafe_allow_html=True,
)
st.write("")

recommender = get_recommender()
tab_chat, tab_explorer, tab_insights = st.tabs(
    ["\U0001F4AC Chat Assistant", "\U0001F9E9 Compatibility Explorer", "\U0001F4CA Dataset Insights"]
)

# --------------------------------------------------------------- Chat tab
with tab_chat:
    if "chat" not in st.session_state:
        st.session_state.chat = ChatAssistant(recommender)
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    with st.sidebar:
        st.markdown("### Your profile")
        st.caption("Optional -- the assistant also picks these up from what you type.")
        g = st.selectbox("Gender", ["(unspecified)", "men", "women"], key="profile_gender")
        age = st.number_input("Age", min_value=0, max_value=100, value=0, key="profile_age")
        occ = st.selectbox(
            "Occasion",
            ["(unspecified)", "office", "party", "wedding", "festive", "casual", "sports", "vacation", "winter"],
            key="profile_occasion",
        )
        if st.button("Apply profile to chat"):
            p = st.session_state.chat.profile
            if g != "(unspecified)":
                p.gender = g
            if age:
                p.age = int(age)
            if occ != "(unspecified)":
                p.occasion = occ
            st.success("Profile updated.")
        if st.button("Reset conversation"):
            st.session_state.chat = ChatAssistant(recommender)
            st.session_state.chat_log = []
            st.rerun()

    st.caption(
        "Try: \"I need an outfit for a business meeting\" / \"I am a 22-year-old male looking for a "
        "casual summer outfit\" / \"Suggest something stylish for a beach vacation, I'm a woman\""
    )

    for turn in st.session_state.chat_log:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["reply"])
            for outfit in turn["outfits"]:
                with st.container(border=False):
                    st.markdown("<div class='outfit-card'>", unsafe_allow_html=True)
                    render_outfit_card(outfit)
                    st.markdown("</div>", unsafe_allow_html=True)

    query = st.chat_input("Tell the assistant what you're dressing for...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        turn = st.session_state.chat.ask(query, top_n_outfits=3)
        st.session_state.chat_log.append(turn)
        with st.chat_message("assistant"):
            st.write(turn["reply"])
            for outfit in turn["outfits"]:
                st.markdown("<div class='outfit-card'>", unsafe_allow_html=True)
                render_outfit_card(outfit)
                st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------- Explorer tab
with tab_explorer:
    st.markdown("#### Pick one item, see what the compatibility model builds around it")
    products = load_products()
    products_display = products.copy()
    products_display["label"] = products_display["name"] + "  --  " + products_display["id"]
    choice = st.selectbox("Anchor item", products_display["label"].tolist())
    anchor_id = choice.split("--")[-1].strip()

    if st.button("Build outfit around this item", type="primary"):
        result = recommender.recommend_from_item(anchor_id, top_k_per_role=3)
        anchor = result["anchor"]
        c1, c2 = st.columns([1, 3])
        with c1:
            img = config.RAW_DIR / anchor["image"]
            if img.exists():
                st.image(str(img), use_container_width=True)
        with c2:
            st.markdown(f"**{anchor['name']}**")
            st.markdown(f"<span class='mono'>{anchor['category_label']} | {anchor['color']} | "
                        f"{anchor['occasion']} | Rs {int(anchor['price_inr'])}</span>", unsafe_allow_html=True)

        for role, recs in result["recommendations"].items():
            st.markdown(f"<span class='role-chip'>{role}</span>", unsafe_allow_html=True)
            cols = st.columns(len(recs) or 1)
            for i, r in enumerate(recs):
                with cols[i]:
                    item = r["item"]
                    img = config.RAW_DIR / item["image"]
                    if img.exists():
                        st.image(str(img), use_container_width=True)
                    st.markdown(f"<div class='item-name'>{item['name']}</div>", unsafe_allow_html=True)
                    pct = int(round(r["score"] * 100))
                    st.markdown(
                        f"<div class='mono'>{pct}% match</div>"
                        f"<div class='score-track'><div class='score-fill' style='width:{pct}%;'></div></div>",
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f"""<div class="stylists-note"><span class="label">Stylist's Note</span>
                <p>{recs[0]['reason']}</p></div>""",
                unsafe_allow_html=True,
            )
            st.write("")

# ----------------------------------------------------------- Insights tab
with tab_insights:
    st.markdown("#### Dataset analysis")
    products = load_products()
    outfits = load_outfits()
    summary = summarize_dataset(products, outfits)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", summary.n_products)
    c2.metric("Curated outfits", summary.n_outfits)
    c3.metric("Brands", products["brand"].nunique())
    c4.metric("Missing color signal", summary.products_missing_color)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Role distribution**")
        st.bar_chart(pd.Series(summary.role_counts))
        st.markdown("**Gender split**")
        st.bar_chart(pd.Series(summary.gender_counts))
    with cc2:
        st.markdown("**Occasion distribution**")
        st.bar_chart(pd.Series(summary.occasion_counts))
        st.markdown("**Wear-type split**")
        st.bar_chart(pd.Series(summary.wear_type_counts))

    st.markdown("**Sample of enriched product metadata**")
    st.dataframe(
        products[["id", "name", "gender", "role", "color", "occasion", "price_inr"]].head(15),
        use_container_width=True,
    )
