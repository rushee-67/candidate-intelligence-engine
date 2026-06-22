"""
Scoring Module.

Loads the precomputed feature parquet and embedding .npy files produced by
precompute_features.py, then computes a final_score for every candidate.

Score formula
-------------
All weights and thresholds are read from config/weights.yaml.

    cosine_sim  = normalised dot product of candidate text-blob embedding vs JD embedding
    title_score = (cosine_sim + 1) / 2            -- maps [-1,1] to [0,1]

    experience_score = min(1.0, relevant_exp_months / experience_relevance_cap_months)

    skill_trust_norm = min(1.0, skill_trust_score / 2.0)
        (theoretical max of skill_trust_score is 2.0; divide to keep in [0,1])

    location_score  ← lookup from config/weights.yaml location_scores map

    soft_negative_modifier ← already in parquet from disqualifier_filters

    base_fit = (title_score_weight   * title_score
              + skill_trust_weight   * skill_trust_norm
              + experience_score_weight * experience_score
              + soft_negative_weight * soft_negative_modifier
              + location_score_weight * location_score)

    availability_multiplier:
        recency   = 1.0 if days_since_active < 30
                    max(0, 1 − (days−30) / (decay_days−30))  otherwise
        engagement = mean([recruiter_response_rate,
                           interview_completion_rate,
                           1.0 if open_to_work else 0.3,
                           offer_acceptance_rate  ← skipped if == -1])
        multiplier = availability_floor + availability_range * mean([recency, engagement])

    final_score = 0  if is_honeypot or is_hard_disqualified
                  else base_fit * availability_multiplier
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "weights.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUT_PARQUET = OUTPUT_DIR / "candidate_features.parquet"
JD_EMBEDDING_NPY = OUTPUT_DIR / "jd_embedding.npy"
CANDIDATE_EMBEDDINGS_NPY = OUTPUT_DIR / "candidate_embeddings.npy"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_weights() -> dict:
    with open(CONFIG_PATH, "rb") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Location score
# ---------------------------------------------------------------------------

# Cities that map to the "ncr" bucket even though they don't contain "ncr"
_NCR_CITIES = frozenset({"gurgaon", "gurugram", "faridabad", "new delhi"})
_INDIA_COUNTRIES = frozenset({"india", "in"})


def _location_score(location: str, country: str, weights: dict) -> float:
    """
    Map a candidate's location/country pair to a score in [0, 1] using the
    location_scores map in weights.yaml.
    """
    loc_scores: dict[str, float] = weights["location_scores"]
    outside = loc_scores.get("outside_india", 0.1)
    other_india = loc_scores.get("other_india", 0.5)

    if (country or "").strip().lower() not in _INDIA_COUNTRIES:
        return outside

    loc_lower = (location or "").lower()

    # Check NCR fringe cities first
    for ncr_city in _NCR_CITIES:
        if ncr_city in loc_lower:
            return loc_scores.get("ncr", 0.8)

    # Check each config key as a substring of the location string
    for key, score in loc_scores.items():
        if key in ("other_india", "outside_india"):
            continue
        if key in loc_lower:
            return score

    return other_india


# ---------------------------------------------------------------------------
# Availability multiplier helpers
# ---------------------------------------------------------------------------

def _recency_score(last_active_date, decay_days: float) -> float:
    """
    1.0 if candidate was active within the last 30 days;
    linear decay to 0.0 at decay_days; clamped at 0.
    """
    if last_active_date is None or (isinstance(last_active_date, float) and np.isnan(last_active_date)):
        return 0.0
    try:
        if isinstance(last_active_date, (date, datetime)):
            d = last_active_date if isinstance(last_active_date, date) else last_active_date.date()
        else:
            d = date.fromisoformat(str(last_active_date)[:10])
        days = (date.today() - d).days
    except (ValueError, TypeError):
        return 0.0

    if days < 30:
        return 1.0
    denom = max(1.0, decay_days - 30.0)
    return max(0.0, 1.0 - (days - 30.0) / denom)


def _engagement_score(row: pd.Series) -> float:
    """
    Mean of: recruiter_response_rate, interview_completion_rate,
             1.0 if open_to_work else 0.3,
             offer_acceptance_rate (skipped when == -1).
    """
    components: list[float] = [
        float(row.get("recruiter_response_rate") or 0.0),
        float(row.get("interview_completion_rate") or 0.0),
        1.0 if row.get("open_to_work_flag") else 0.3,
    ]
    oar = row.get("offer_acceptance_rate", -1)
    if oar is not None and float(oar) != -1.0:
        components.append(float(oar))
    return float(np.mean(components))


def _availability_multiplier(row: pd.Series, weights: dict) -> float:
    """
    Combines recency and engagement into a single multiplier in
    [availability_floor, availability_floor + availability_range].
    """
    floor: float = weights["availability_floor"]
    rng: float = weights["availability_range"]
    decay: float = weights["recency_decay_days"]

    recency = _recency_score(row.get("last_active_date"), decay)
    engagement = _engagement_score(row)
    return floor + rng * float(np.mean([recency, engagement]))


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def compute_scores(
    df: pd.DataFrame,
    candidate_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Compute final_score for every row in *df* and return a new DataFrame
    sorted by final_score descending.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with all columns produced by the precompute pipeline
        (including is_honeypot, is_hard_disqualified, soft_negative_modifier).
    candidate_embeddings : np.ndarray, shape (N, 384)
        One row per candidate, same order as df.
    jd_embedding : np.ndarray, shape (384,)
        The JD anchor embedding.
    weights : dict, optional
        Parsed weights.yaml; loaded from disk if None.

    Returns
    -------
    pd.DataFrame
        Input columns plus: cosine_sim, title_score, experience_score,
        skill_trust_norm, location_score, availability_multiplier, base_fit,
        final_score.  Sorted by final_score descending.
    """
    if weights is None:
        weights = _load_weights()

    n = len(df)
    assert len(candidate_embeddings) == n, (
        f"Embeddings length {len(candidate_embeddings)} != DataFrame length {n}"
    )

    # ------------------------------------------------------------------ #
    # 1. Cosine similarity                                                 #
    # ------------------------------------------------------------------ #
    jd_norm = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-8)
    cand_norms = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
    )
    cosine_sim: np.ndarray = cand_norms @ jd_norm  # shape (N,)

    # ------------------------------------------------------------------ #
    # 2. Title score                                                       #
    # ------------------------------------------------------------------ #
    title_score: np.ndarray = (cosine_sim + 1.0) / 2.0

    # ------------------------------------------------------------------ #
    # 3. Experience score                                                  #
    # ------------------------------------------------------------------ #
    cap: float = float(weights["experience_relevance_cap_months"])
    experience_score: np.ndarray = np.minimum(
        1.0,
        df["relevant_exp_months"].to_numpy(dtype=float) / cap,
    )

    # ------------------------------------------------------------------ #
    # 4. Skill trust (normalised to [0, 1]; theoretical max is 2.0)       #
    # ------------------------------------------------------------------ #
    skill_trust_norm: np.ndarray = np.minimum(
        1.0,
        df["skill_trust_score"].to_numpy(dtype=float) / 2.0,
    )

    # ------------------------------------------------------------------ #
    # 5. Location score                                                    #
    # ------------------------------------------------------------------ #
    location_score: np.ndarray = np.array([
        _location_score(row["location"], row["country"], weights)
        for _, row in df.iterrows()
    ], dtype=float)

    # ------------------------------------------------------------------ #
    # 6. Soft-negative modifier (already in the DataFrame)                #
    # ------------------------------------------------------------------ #
    soft_modifier: np.ndarray = df.get(
        "soft_negative_modifier",
        pd.Series(np.ones(n)),
    ).to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 7. Availability multiplier                                           #
    # ------------------------------------------------------------------ #
    availability: np.ndarray = np.array([
        _availability_multiplier(row, weights)
        for _, row in df.iterrows()
    ], dtype=float)

    # ------------------------------------------------------------------ #
    # 8. Base fit                                                          #
    # ------------------------------------------------------------------ #
    w_title: float = weights["title_score_weight"]
    w_skill: float = weights["skill_trust_weight"]
    w_exp: float = weights["experience_score_weight"]
    w_soft: float = weights["soft_negative_weight"]
    w_loc: float = weights["location_score_weight"]

    base_fit: np.ndarray = (
        w_title * title_score
        + w_skill * skill_trust_norm
        + w_exp  * experience_score
        + w_soft * soft_modifier
        + w_loc  * location_score
    )

    # ------------------------------------------------------------------ #
    # 9. Exclusions → score = 0                                           #
    # ------------------------------------------------------------------ #
    is_honeypot: np.ndarray = df.get(
        "is_honeypot", pd.Series(np.zeros(n, dtype=bool))
    ).to_numpy(dtype=bool)
    is_hard_disq: np.ndarray = df.get(
        "is_hard_disqualified", pd.Series(np.zeros(n, dtype=bool))
    ).to_numpy(dtype=bool)
    excluded: np.ndarray = is_honeypot | is_hard_disq

    final_score: np.ndarray = np.where(excluded, 0.0, base_fit * availability)

    # ------------------------------------------------------------------ #
    # 10. Assemble result DataFrame                                        #
    # ------------------------------------------------------------------ #
    result = df[["candidate_id"]].copy().reset_index(drop=True)
    result["cosine_sim"] = cosine_sim
    result["title_score"] = title_score
    result["experience_score"] = experience_score
    result["skill_trust_norm"] = skill_trust_norm
    result["location_score"] = location_score
    result["soft_negative_modifier"] = soft_modifier
    result["availability_multiplier"] = availability
    result["base_fit"] = base_fit
    result["final_score"] = final_score
    result["is_honeypot"] = is_honeypot
    result["is_hard_disqualified"] = is_hard_disq

    return result.sort_values("final_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience loader — read everything from disk
# ---------------------------------------------------------------------------

def load_and_score(
    parquet_path: Path = OUTPUT_PARQUET,
    embeddings_path: Path = CANDIDATE_EMBEDDINGS_NPY,
    jd_emb_path: Path = JD_EMBEDDING_NPY,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Load precomputed artifacts from disk and return a scored DataFrame.
    """
    if weights is None:
        weights = _load_weights()
    df = pd.read_parquet(parquet_path)
    cand_embs = np.load(embeddings_path)
    jd_emb = np.load(jd_emb_path)
    return compute_scores(df, cand_embs, jd_emb, weights)


if __name__ == "__main__":
    scored = load_and_score()
    print(scored[["candidate_id", "final_score", "title_score",
                  "experience_score", "skill_trust_norm",
                  "location_score", "base_fit",
                  "availability_multiplier"]].head(20).to_string(index=False))
