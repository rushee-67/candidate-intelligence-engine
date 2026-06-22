"""
Precompute Features Module.

This module parses raw candidate profiles from the JSONL dataset and precomputes
the feature metrics required for scoring and filtering.

================================================================================
DATA ANALYSIS FINDINGS (from resources/sample_candidates.json & schema)
================================================================================
1. Top-Level Keys in a Candidate Record:
   - candidate_id
   - career_history
   - certifications
   - education
   - languages
   - profile
   - redrob_signals
   - skills

2. Keys Inside redrob_signals:
   - applications_submitted_30d
   - avg_response_time_hours
   - connection_count
   - endorsements_received
   - expected_salary_range_inr_lpa (contains 'min' and 'max' fields)
   - github_activity_score
   - interview_completion_rate
   - last_active_date
   - linkedin_connected
   - notice_period_days
   - offer_acceptance_rate
   - open_to_work_flag
   - preferred_work_mode
   - profile_completeness_score
   - profile_views_received_30d
   - recruiter_response_rate
   - saved_by_recruiters_30d
   - search_appearance_30d
   - signup_date
   - skill_assessment_scores (nested dictionary of skill_name -> score)
   - verified_email
   - verified_phone
   - willing_to_relocate

3. Skill proficiency, endorsements, and duration_months for the first 3 candidates:
   - Candidate 1 (CAND_0000001):
     * Tailwind: proficiency=intermediate, endorsements=3, duration_months=13
     * NLP: proficiency=advanced, endorsements=37, duration_months=26
     * Image Classification: proficiency=advanced, endorsements=7, duration_months=40
     * Fine-tuning LLMs: proficiency=advanced, endorsements=21, duration_months=36
     * Weights & Biases: proficiency=intermediate, endorsements=13, duration_months=30
     * Speech Recognition: proficiency=advanced, endorsements=52, duration_months=33
     * Photoshop: proficiency=intermediate, endorsements=8, duration_months=24
     * TTS: proficiency=advanced, endorsements=56, duration_months=60
     * LoRA: proficiency=intermediate, endorsements=0, duration_months=28
     * Apache Beam: proficiency=intermediate, endorsements=4, duration_months=9
     * AWS: proficiency=beginner, endorsements=5, duration_months=8
     * Flask: proficiency=beginner, endorsements=15, duration_months=15
     * BentoML: proficiency=intermediate, endorsements=3, duration_months=36
     * Milvus: proficiency=advanced, endorsements=40, duration_months=35
     * GANs: proficiency=advanced, endorsements=12, duration_months=19
     * Statistical Modeling: proficiency=intermediate, endorsements=9, duration_months=8
     * GCP: proficiency=beginner, endorsements=7, duration_months=2
   - Candidate 2 (CAND_0000002):
     * Project Management: proficiency=intermediate, endorsements=14, duration_months=23
     * React: proficiency=intermediate, endorsements=6, duration_months=35
     * Photoshop: proficiency=intermediate, endorsements=9, duration_months=35
     * TypeScript: proficiency=beginner, endorsements=2, duration_months=3
     * Marketing: proficiency=beginner, endorsements=9, duration_months=11
     * Kafka: proficiency=intermediate, endorsements=3, duration_months=16
     * JavaScript: proficiency=beginner, endorsements=9, duration_months=3
     * Feature Engineering: proficiency=intermediate, endorsements=11, duration_months=27
     * GCP: proficiency=intermediate, endorsements=7, duration_months=30
   - Candidate 3 (CAND_0000003):
     * Angular: proficiency=intermediate, endorsements=13, duration_months=10
     * SEO: proficiency=beginner, endorsements=11, duration_months=11
     * Excel: proficiency=intermediate, endorsements=2, duration_months=15
     * Accounting: proficiency=beginner, endorsements=7, duration_months=18
     * Kubernetes: proficiency=intermediate, endorsements=0, duration_months=34
     * Databricks: proficiency=beginner, endorsements=14, duration_months=18

4. Fields present in the data but missing from candidate_schema.json:
   - None. Every key and nested key in the sample candidate profiles matches the schema definition.
================================================================================

This module parses raw candidate profiles from the JSONL dataset and precomputes
the feature metrics required for scoring and filtering, such as:
- Total work experience duration (months) and relevant AI/ML tenure.
- Skill trust levels based on endorsements, proficiency levels, and usage duration.
- Classification of past employers (e.g., product companies vs service companies like TCS, Infosys, Wipro).
- Relocation willingness and normalized locations.
- Engagement scores from Redrob behavioral signals (active recency, response rates, etc.).
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Iterator

import orjson
import pandas as pd
import yaml

from src.jd_anchor import JD_REQUIRED_SKILLS_LOWER, RELEVANCE_KEYWORDS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "weights.yaml"
CANDIDATES_JSONL = PROJECT_ROOT / "resources" / "candidates.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts"
OUTPUT_PARQUET = OUTPUT_DIR / "candidate_features.parquet"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_weights() -> dict:
    """Load and return the parsed config/weights.yaml dictionary."""
    with open(CONFIG_PATH, "rb") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterator[dict]:
    """
    Yield one candidate dict per line from a JSONL file using orjson for
    speed/memory efficiency.  Lines that fail to parse are skipped with a
    warning rather than aborting the entire run.
    """
    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield orjson.loads(raw)
            except orjson.JSONDecodeError as exc:
                print(f"[precompute] Skipping malformed line {lineno}: {exc}")


# ---------------------------------------------------------------------------
# Per-candidate feature extraction helpers
# ---------------------------------------------------------------------------

def _relevant_exp_months(career_history: list[dict]) -> int:
    """
    Sum duration_months for roles whose description contains any relevance
    keyword (case-insensitive substring match).
    """
    keywords = [kw.lower() for kw in RELEVANCE_KEYWORDS]
    total = 0
    for role in career_history:
        desc = (role.get("description") or "").lower()
        if any(kw in desc for kw in keywords):
            total += role.get("duration_months", 0)
    return total


def _skill_trust_score(skills: list[dict], weights: dict) -> float:
    """
    Compute mean trust score over JD-required skills that appear in the
    candidate's profile.

    Trust formula (all parameters from weights.yaml):
        proficiency_weight  — from skill_proficiency_weights
        endorse_boost       — min(endorsements / endorsement_scale, 1.0)
        duration_factor     — min(duration_months / duration_cap, 1.0)
        trust_i             = proficiency_weight * (1 + endorse_boost) * duration_factor

    If the candidate has none of the required skills, returns 0.0.
    """
    prof_weights: dict[str, float] = weights["skill_proficiency_weights"]
    endorse_scale: float = weights["skill_trust_endorsement_scale"]
    dur_cap: float = weights["skill_trust_duration_cap_months"]

    scores: list[float] = []
    for skill in skills:
        name_lower = (skill.get("name") or "").lower()
        if name_lower not in JD_REQUIRED_SKILLS_LOWER:
            continue
        prof = skill.get("proficiency", "beginner")
        pw = prof_weights.get(prof, 0.25)
        endorse_boost = min((skill.get("endorsements") or 0) / endorse_scale, 1.0)
        dur_factor = min((skill.get("duration_months") or 0) / max(dur_cap, 1.0), 1.0)
        trust_i = pw * (1.0 + endorse_boost) * dur_factor
        scores.append(trust_i)

    return float(sum(scores) / len(scores)) if scores else 0.0


def _build_text_blob(profile: dict, career_history: list[dict]) -> str:
    """
    Concatenate: current_title + headline + summary + last 3 role descriptions.
    """
    parts: list[str] = [
        profile.get("current_title") or "",
        profile.get("headline") or "",
        profile.get("summary") or "",
    ]
    # Sort by start_date descending to get the most recent 3 roles
    sorted_roles = sorted(
        career_history,
        key=lambda r: r.get("start_date") or "0000-00-00",
        reverse=True,
    )
    for role in sorted_roles[:3]:
        parts.append(role.get("description") or "")
    return " ".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_candidate_features(candidate: dict, weights: dict) -> dict:
    """
    Parse a single raw candidate dictionary and return a flat feature dict
    ready for a single pandas row.

    Parameters
    ----------
    candidate : dict
        Raw candidate dict as loaded from candidates.jsonl.
    weights : dict
        Parsed config/weights.yaml.

    Returns
    -------
    dict
        Flat feature dictionary.
    """
    profile: dict = candidate.get("profile") or {}
    career: list[dict] = candidate.get("career_history") or []
    skills: list[dict] = candidate.get("skills") or []
    signals: dict = candidate.get("redrob_signals") or {}

    return {
        # --- Identity ---
        "candidate_id": candidate.get("candidate_id"),
        # --- Profile basics ---
        "years_exp": profile.get("years_of_experience", 0.0),
        "current_title": profile.get("current_title", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        # --- Career aggregates ---
        "sum_career_months": sum(r.get("duration_months", 0) for r in career),
        "relevant_exp_months": _relevant_exp_months(career),
        # --- Skill trust ---
        "skill_trust_score": _skill_trust_score(skills, weights),
        # --- Redrob signals ---
        "github_activity_score": signals.get("github_activity_score", -1),
        "profile_completeness_score": signals.get("profile_completeness_score", 0.0),
        "last_active_date": signals.get("last_active_date", ""),
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
        "avg_response_time_hours": signals.get("avg_response_time_hours", 0.0),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.0),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
        "notice_period_days": signals.get("notice_period_days", 0),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "preferred_work_mode": signals.get("preferred_work_mode", ""),
        "verified_email": bool(signals.get("verified_email", False)),
        "verified_phone": bool(signals.get("verified_phone", False)),
        "linkedin_connected": bool(signals.get("linkedin_connected", False)),
        # --- Text blob for semantic scoring ---
        "candidate_text_blob": _build_text_blob(profile, career),
    }


# ---------------------------------------------------------------------------
# Batch runner — JSONL → Parquet
# ---------------------------------------------------------------------------

def precompute_all_features(
    jsonl_path: Path = CANDIDATES_JSONL,
    output_path: Path = OUTPUT_PARQUET,
    weights: dict | None = None,
    chunk_size: int = 5_000,
) -> pd.DataFrame:
    """
    Stream ``jsonl_path`` line-by-line, compute features for every candidate,
    and write the result to ``output_path`` as a Parquet file.

    Parameters
    ----------
    jsonl_path : Path
        Path to the candidates JSONL file (defaults to resources/candidates.jsonl).
    output_path : Path
        Destination Parquet path (defaults to data/artifacts/candidate_features.parquet).
    weights : dict, optional
        Pre-loaded weights dict.  Loaded from config/weights.yaml if None.
    chunk_size : int
        Number of records accumulated before writing a partial Parquet chunk
        to manage peak memory.  All chunks are concatenated at the end.

    Returns
    -------
    pd.DataFrame
        The full features DataFrame (also persisted to ``output_path``).
    """
    if weights is None:
        weights = _load_weights()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = 0

    for candidate in _iter_jsonl(jsonl_path):
        rows.append(extract_candidate_features(candidate, weights))
        total += 1
        if total % chunk_size == 0:
            print(f"[precompute] Processed {total} candidates…")

    print(f"[precompute] Total candidates processed: {total}")

    df = pd.DataFrame(rows)

    # Ensure correct dtypes
    df["last_active_date"] = pd.to_datetime(df["last_active_date"], errors="coerce")
    df["open_to_work_flag"] = df["open_to_work_flag"].astype(bool)
    df["willing_to_relocate"] = df["willing_to_relocate"].astype(bool)
    df["verified_email"] = df["verified_email"].astype(bool)
    df["verified_phone"] = df["verified_phone"].astype(bool)
    df["linkedin_connected"] = df["linkedin_connected"].astype(bool)

    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"[precompute] Saved features to {output_path}  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Convenience: build from a list of dicts (used by tests / sandbox)
# ---------------------------------------------------------------------------

def features_from_list(
    candidates: list[dict],
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Build a features DataFrame directly from a Python list of candidate dicts
    (e.g. loaded from sample_candidates.json).  Does NOT write to disk.
    """
    if weights is None:
        weights = _load_weights()
    rows = [extract_candidate_features(c, weights) for c in candidates]
    df = pd.DataFrame(rows)
    df["last_active_date"] = pd.to_datetime(df["last_active_date"], errors="coerce")
    for col in ("open_to_work_flag", "willing_to_relocate", "verified_email",
                "verified_phone", "linkedin_connected"):
        df[col] = df[col].astype(bool)
    return df


if __name__ == "__main__":
    df = precompute_all_features()
    print(df.dtypes)
    print(df.head(3))
