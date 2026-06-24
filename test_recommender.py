"""
Test suite. Run with:  pytest tests/ -v

These tests run fully offline against the artifacts produced by
`scripts/build_index.py` (the tests assume that script has been run --
see the Makefile target `make test` / README setup instructions, which
run build_index.py before pytest).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import load_products, load_outfits
from src.compatibility_model import CompatibilityModel, build_training_table, pair_features
from src.embeddings import TfidfEmbedder
from src.vector_store import ProductVectorStore
from src.nlu import parse_rule_based
from src.recommender import OutfitRecommender
from src.user_profile import UserProfile
from src.chat_assistant import ChatAssistant


@pytest.fixture(scope="module")
def products():
    return load_products()


@pytest.fixture(scope="module")
def outfits():
    return load_outfits()


@pytest.fixture(scope="module")
def recommender():
    return OutfitRecommender.load_default()


# --------------------------------------------------------------- data
def test_products_load_and_have_expected_columns(products):
    assert len(products) == 68
    for col in ("role", "color", "color_family", "text_blob"):
        assert col in products.columns


def test_every_category_maps_to_a_role(products):
    assert products["role"].isna().sum() == 0
    assert set(products["role"].unique()) <= set(config.ALL_ROLES)


def test_outfits_load(outfits):
    assert len(outfits) == 25
    assert "stylist_rationale" in outfits.columns


# ------------------------------------------------------ compatibility model
def test_training_table_has_positive_and_negative_examples(products, outfits):
    table = build_training_table(products, outfits)
    assert table["label"].sum() > 0
    assert (table["label"] == 0).sum() > 0


def test_compatibility_model_scores_in_unit_interval(products, outfits):
    table = build_training_table(products, outfits)
    model = CompatibilityModel().fit(table)
    sample = table.iloc[0].to_dict()
    score = model.predict_proba(sample)
    assert 0.0 <= score <= 1.0


def test_known_good_pair_scores_higher_than_known_bad_pair(products):
    products = products.set_index("id", drop=False)
    model = CompatibilityModel.load()
    # Formal shirt + formal trousers (same office occasion, neutral colors)
    # should clearly outscore formal shirt + running shoes (mismatched
    # formality), validating the model learned *something* sensible.
    shirt = products.loc["myntra_28569210"]
    trousers = products.loc["myntra_23237806"]
    running_shoes = products.loc["ajio_469763526"]

    good_feats = pair_features(shirt, trousers, text_sim=0.1)
    bad_feats = pair_features(shirt, running_shoes, text_sim=0.1)
    assert model.predict_proba(good_feats) >= model.predict_proba(bad_feats)


# ----------------------------------------------------------------- NLU
@pytest.mark.parametrize("query,expected_gender", [
    ("I am a woman attending a wedding", "women"),
    ("I am a 22-year-old male looking for a casual outfit", "men"),
    ("My girlfriend needs a party dress", "women"),
])
def test_gender_parsing_handles_substring_traps(query, expected_gender):
    """Regression test for the 'man' substring inside 'woman' bug."""
    intent = parse_rule_based(query)
    assert intent.gender == expected_gender


def test_occasion_parsing():
    assert parse_rule_based("I need an outfit for a business meeting").occasion == "office"
    assert parse_rule_based("attending a wedding next weekend").occasion == "wedding"
    assert parse_rule_based("beach vacation please").occasion == "vacation"


def test_age_parsing():
    assert parse_rule_based("I am a 22-year-old male").age == 22
    assert parse_rule_based("no age mentioned here").age is None


# ------------------------------------------------------------ vector store
def test_vector_store_round_trip(products):
    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform(products["text_blob"].tolist())
    store = ProductVectorStore().build(vectors, products["id"].tolist())
    query_vec = embedder.transform([products.iloc[0]["text_blob"]])[0]
    results = store.search(query_vec, top_k=5)
    assert len(results) == 5
    assert results[0][0] == products.iloc[0]["id"]  # nearest neighbor of itself is itself


# ------------------------------------------------------------- recommender
def test_recommend_from_item_returns_all_mandatory_roles(recommender):
    result = recommender.recommend_from_item("myntra_28569210")  # white formal shirt, men
    assert "bottomwear" in result["recommendations"]
    assert "footwear" in result["recommendations"]
    assert len(result["recommendations"]["bottomwear"]) > 0


def test_recommend_from_item_unknown_id_raises(recommender):
    with pytest.raises(KeyError):
        recommender.recommend_from_item("not_a_real_id")


def test_recommend_from_query_returns_complete_outfits(recommender):
    profile = UserProfile(gender="men", occasion="office")
    outfits = recommender.recommend_from_query("business meeting outfit", profile, top_n_outfits=2)
    assert len(outfits) >= 1
    for outfit in outfits:
        roles = set(outfit["items"].keys())
        has_bottom_coverage = "onepiece" in roles or {"topwear", "bottomwear"} <= roles
        assert has_bottom_coverage
        assert "footwear" in roles


def test_recommend_from_query_respects_gender_consistency(recommender):
    profile = UserProfile(gender="women")
    outfits = recommender.recommend_from_query("wedding outfit", profile, top_n_outfits=2)
    for outfit in outfits:
        for item in outfit["items"].values():
            assert item["gender"] == "women"


# --------------------------------------------------------- chat assistant
def test_chat_assistant_persists_profile_across_turns(recommender):
    chat = ChatAssistant(recommender)
    chat.ask("I am a 24 year old man")
    assert chat.profile.gender == "men"
    assert chat.profile.age == 24
    chat.ask("I need something for a party")
    assert chat.profile.gender == "men"  # still remembered
    assert chat.profile.occasion == "party"
