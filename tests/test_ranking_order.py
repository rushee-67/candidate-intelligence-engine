"""
Ranking Order Integration Tests.

Runs the full precompute + scoring pipeline on the 6 synthetic fixtures and
asserts the expected ordering:

    ideal  >  tier5_fit  >  keyword_stuffer  >  title_chaser
    honeypot.is_honeypot == True  →  score == 0
    consulting_only.soft_negative_modifier < 1.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when pytest runs from any directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jd_anchor import JD_ANCHOR_TEXT
from src.precompute_features import features_from_list
from src.honeypot_audit import add_honeypot_column
from src.disqualifier_filters import add_disqualifier_columns
from src.scoring import compute_scores

from tests.fixtures import (
    IDEAL,
    KEYWORD_STUFFER,
    HONEYPOT,
    TIER5_FIT,
    TITLE_CHASER,
    CONSULTING_ONLY,
    ALL_FIXTURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_weights() -> dict:
    with open(PROJECT_ROOT / "config" / "weights.yaml", "rb") as fh:
        return yaml.safe_load(fh)


def _run_pipeline(candidates: list[dict], weights: dict):
    """
    Run the full precompute + audit + filter + embedding + scoring pipeline
    on an in-memory list of candidate dicts.

    Returns
    -------
    scored_df : pd.DataFrame
        Scored and sorted DataFrame.
    enriched_df : pd.DataFrame
        The enriched feature DataFrame (before scoring, includes audit cols).
    """
    from sentence_transformers import SentenceTransformer

    # --- Feature extraction ---
    df = features_from_list(candidates, weights)

    # --- Honeypot audit ---
    df = add_honeypot_column(df, candidates, weights)

    # --- Disqualifier filters ---
    df = add_disqualifier_columns(df, candidates, weights)

    # --- Embeddings ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    blobs = df["candidate_text_blob"].fillna("").tolist()
    cand_embs = model.encode(blobs, convert_to_numpy=True, show_progress_bar=False)
    jd_emb = model.encode(JD_ANCHOR_TEXT, convert_to_numpy=True, show_progress_bar=False)

    # --- Scoring ---
    scored = compute_scores(df, cand_embs, jd_emb, weights)

    return scored, df


# ---------------------------------------------------------------------------
# Shared fixture (pytest) — runs pipeline once for the entire test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pipeline_results():
    weights = _load_weights()
    scored, enriched = _run_pipeline(ALL_FIXTURES, weights)
    return scored, enriched


def _score(scored_df, candidate_id: str) -> float:
    rows = scored_df.loc[scored_df["candidate_id"] == candidate_id, "final_score"]
    assert len(rows) == 1, f"candidate_id {candidate_id!r} not found in scored_df"
    return float(rows.iloc[0])


def _enriched_col(enriched_df, candidate_id: str, col: str):
    rows = enriched_df.loc[enriched_df["candidate_id"] == candidate_id, col]
    assert len(rows) == 1
    return rows.iloc[0]


def _scored_col(scored_df, candidate_id: str, col: str):
    rows = scored_df.loc[scored_df["candidate_id"] == candidate_id, col]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHoneypotExclusion:
    def test_honeypot_is_flagged(self, pipeline_results):
        scored, enriched = pipeline_results
        assert _enriched_col(enriched, "CAND_9000003", "is_honeypot") is True or \
               bool(_enriched_col(enriched, "CAND_9000003", "is_honeypot")) is True

    def test_honeypot_score_is_zero(self, pipeline_results):
        scored, _ = pipeline_results
        assert _score(scored, "CAND_9000003") == 0.0, (
            f"Honeypot score should be 0.0, got {_score(scored, 'CAND_9000003')}"
        )

    def test_honeypot_expert_no_time_flag(self, pipeline_results):
        _, enriched = pipeline_results
        assert bool(_enriched_col(enriched, "CAND_9000003", "flag_expert_no_time")) is True

    def test_honeypot_career_mismatch_flag(self, pipeline_results):
        _, enriched = pipeline_results
        assert bool(_enriched_col(enriched, "CAND_9000003", "flag_career_mismatch")) is True


class TestRankingOrder:
    def test_ideal_beats_tier5_fit(self, pipeline_results):
        scored, _ = pipeline_results
        ideal_score = _score(scored, "CAND_9000001")
        tier5_score = _score(scored, "CAND_9000004")
        assert ideal_score > tier5_score, (
            f"ideal ({ideal_score:.4f}) should beat tier5_fit ({tier5_score:.4f})"
        )

    def test_tier5_fit_beats_keyword_stuffer(self, pipeline_results):
        scored, _ = pipeline_results
        tier5_score  = _score(scored, "CAND_9000004")
        kw_score     = _score(scored, "CAND_9000002")
        assert tier5_score > kw_score, (
            f"tier5_fit ({tier5_score:.4f}) should beat keyword_stuffer ({kw_score:.4f})"
        )

    def test_keyword_stuffer_beats_title_chaser(self, pipeline_results):
        scored, _ = pipeline_results
        kw_score     = _score(scored, "CAND_9000002")
        chaser_score = _score(scored, "CAND_9000005")
        assert kw_score > chaser_score, (
            f"keyword_stuffer ({kw_score:.4f}) should beat title_chaser ({chaser_score:.4f})"
        )


class TestSoftNegatives:
    def test_consulting_only_modifier_lt_1(self, pipeline_results):
        scored, enriched = pipeline_results
        # Check in the enriched df (disqualifier output)
        modifier = float(_enriched_col(enriched, "CAND_9000006", "soft_negative_modifier"))
        assert modifier < 1.0, (
            f"consulting_only soft_negative_modifier should be < 1.0, got {modifier}"
        )

    def test_consulting_only_flag_set(self, pipeline_results):
        _, enriched = pipeline_results
        assert bool(_enriched_col(enriched, "CAND_9000006", "soft_consulting_only")) is True

    def test_title_chaser_flag_set(self, pipeline_results):
        _, enriched = pipeline_results
        assert bool(_enriched_col(enriched, "CAND_9000005", "soft_title_chaser")) is True


class TestIdealProperties:
    def test_ideal_not_excluded(self, pipeline_results):
        scored, _ = pipeline_results
        assert _score(scored, "CAND_9000001") > 0.0

    def test_ideal_not_honeypot(self, pipeline_results):
        _, enriched = pipeline_results
        assert bool(_enriched_col(enriched, "CAND_9000001", "is_honeypot")) is False

    def test_ideal_not_hard_disqualified(self, pipeline_results):
        _, enriched = pipeline_results
        assert bool(_enriched_col(enriched, "CAND_9000001", "is_hard_disqualified")) is False

    def test_ideal_has_relevant_experience(self, pipeline_results):
        _, enriched = pipeline_results
        rel_months = int(_enriched_col(enriched, "CAND_9000001", "relevant_exp_months"))
        assert rel_months > 0, f"ideal should have relevant_exp_months > 0, got {rel_months}"


class TestDiagnosticPrint:
    """Non-asserting test that prints a score breakdown — always passes."""

    def test_print_score_breakdown(self, pipeline_results):
        scored, enriched = pipeline_results
        fixture_map = {
            "CAND_9000001": "ideal",
            "CAND_9000002": "keyword_stuffer",
            "CAND_9000003": "honeypot",
            "CAND_9000004": "tier5_fit",
            "CAND_9000005": "title_chaser",
            "CAND_9000006": "consulting_only",
        }
        print("\n\n" + "=" * 72)
        print("SCORE BREAKDOWN BY FIXTURE")
        print("=" * 72)
        cols = ["candidate_id", "final_score", "title_score", "experience_score",
                "skill_trust_norm", "location_score", "soft_negative_modifier",
                "availability_multiplier", "base_fit", "is_honeypot", "is_hard_disqualified"]
        subset = scored[scored["candidate_id"].isin(fixture_map)].copy()
        subset["fixture"] = subset["candidate_id"].map(fixture_map)
        for _, row in subset.sort_values("final_score", ascending=False).iterrows():
            print(f"\n  [{row['fixture']:18s}] (id={row['candidate_id']})")
            print(f"    final_score           = {row['final_score']:.4f}")
            print(f"    base_fit              = {row['base_fit']:.4f}")
            print(f"    title_score           = {row['title_score']:.4f}")
            print(f"    experience_score      = {row['experience_score']:.4f}")
            print(f"    skill_trust_norm      = {row['skill_trust_norm']:.4f}")
            print(f"    location_score        = {row['location_score']:.4f}")
            print(f"    soft_negative_mod     = {row['soft_negative_modifier']:.4f}")
            print(f"    availability_mult     = {row['availability_multiplier']:.4f}")
            print(f"    excluded              = {bool(row['is_honeypot']) or bool(row['is_hard_disqualified'])}")

        # Also print enriched audit columns
        print("\n  Audit flags:")
        audit_cols = ["candidate_id", "flag_expert_no_time", "flag_career_mismatch",
                      "total_flags", "is_honeypot",
                      "hard_pure_research", "hard_langchain_only", "hard_architect_no_code",
                      "is_hard_disqualified",
                      "soft_consulting_only", "soft_title_chaser", "soft_cv_speech_robotics",
                      "num_soft_flags", "soft_negative_modifier"]
        enr = enriched[enriched["candidate_id"].isin(fixture_map)][audit_cols]
        print(enr.to_string(index=False))
        print("=" * 72)
