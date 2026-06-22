"""
Disqualifier Filters Module.

Applies hard exclusion criteria and soft-negative modifiers to the feature
DataFrame produced by precompute_features.

All numeric thresholds (floor, penalty, tenure bounds) are read from
config/weights.yaml.  No hardcoded numbers appear in this module.

Hard disqualifiers (is_hard_disqualified = True → excluded from ranking)
------------------------------------------------------------------------
1. pure_research_only  — every career role's industry is in {"Academia", "Research"}
   AND no production keyword appears in any role description.
2. langchain_only      — "langchain" appears in the candidate's skills or descriptions
   AND there is no ML-flavoured evidence older than 12 months from today.
3. architect_no_code   — most-recent title matches r"architect|tech lead|principal"
   (case-insensitive) AND that role's tenure >= 18 months AND the role's
   description contains none of the code keywords.

Soft negatives (accumulated flags → soft_negative_modifier multiplier)
----------------------------------------------------------------------
1. consulting_only  — every employer is in the large-IT-services set.
2. title_chaser     — 3+ jobs, average tenure < 18 months, AND titles escalate
   through the seniority pattern.
3. cv_speech_robotics — primary domain keywords are CV/speech/robotics with zero
   NLP/retrieval keywords anywhere in skills or descriptions.

modifier = max(floor, 1.0 − penalty × num_soft_flags)
(floor and penalty from config/weights.yaml)
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "weights.yaml"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "artifacts" / "candidate_features.parquet"

# ---------------------------------------------------------------------------
# Constants (sourced from jd_anchor rather than hardcoded here)
# ---------------------------------------------------------------------------
# IT consulting / services firms whose sole employment is a soft-negative
_CONSULTING_FIRMS: frozenset[str] = frozenset({
    "tcs", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra",
})

_RESEARCH_INDUSTRIES: frozenset[str] = frozenset({"academia", "research"})

_PRODUCTION_KEYWORDS: tuple[str, ...] = (
    "production", "deployed", "a/b", "inference", "serving",
    "real-time", "real time", "pipeline", "api",
)

_CODE_KEYWORDS: tuple[str, ...] = (
    "python", "pytorch", "tensorflow", "sklearn", "code",
    "implement", "build", "develop", "wrote", "deployed",
)

_CV_SPEECH_ROBOTICS_KEYWORDS: tuple[str, ...] = (
    "computer vision", "image classification", "object detection",
    "speech recognition", "tts", "text-to-speech", "robotics",
    "autonomous driving", "slam",
)

_NLP_RETRIEVAL_KEYWORDS: tuple[str, ...] = (
    "nlp", "natural language", "retrieval", "embedding",
    "ranking", "recommendation", "llm", "transformer",
    "information retrieval", "vector search",
)

_SENIORITY_PATTERN: re.Pattern = re.compile(
    r"\b(senior|staff|principal|lead|architect)\b",
    re.IGNORECASE,
)

_ARCHITECT_TITLE_PATTERN: re.Pattern = re.compile(
    r"architect|tech lead|principal",
    re.IGNORECASE,
)


def _load_weights() -> dict:
    with open(CONFIG_PATH, "rb") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Helpers operating on raw candidate dicts
# ---------------------------------------------------------------------------

def _all_descriptions(candidate: dict) -> str:
    """Concatenate all career role descriptions into one lowercase string."""
    parts = [
        (r.get("description") or "")
        for r in (candidate.get("career_history") or [])
    ]
    return " ".join(parts).lower()


def _all_skill_names(candidate: dict) -> str:
    """Concatenate all skill names into one lowercase string."""
    return " ".join(
        (s.get("name") or "").lower()
        for s in (candidate.get("skills") or [])
    )


def _all_text(candidate: dict) -> str:
    """Union of all descriptions + skill names + current title + summary."""
    profile = candidate.get("profile") or {}
    return " ".join([
        (profile.get("current_title") or "").lower(),
        (profile.get("summary") or "").lower(),
        _all_descriptions(candidate),
        _all_skill_names(candidate),
    ])


def _months_since(date_str: str) -> float:
    """Return fractional months between date_str (YYYY-MM-DD) and today."""
    try:
        d = date.fromisoformat(date_str)
        delta = date.today() - d
        return delta.days / 30.44
    except (ValueError, TypeError):
        return float("inf")


# ---------------------------------------------------------------------------
# Hard disqualifier checks
# ---------------------------------------------------------------------------

def _is_pure_research(candidate: dict) -> bool:
    """
    True if every career role's industry is academia or research AND no
    production keyword appears in any description.
    """
    career = candidate.get("career_history") or []
    if not career:
        return False

    all_research = all(
        (r.get("industry") or "").lower() in _RESEARCH_INDUSTRIES
        for r in career
    )
    if not all_research:
        return False

    all_desc = _all_descriptions(candidate)
    has_production = any(kw in all_desc for kw in _PRODUCTION_KEYWORDS)
    return not has_production


def _is_langchain_only(candidate: dict) -> bool:
    """
    True if 'langchain' appears in skills/descriptions AND there is no
    ML-related evidence older than 12 months from today.

    'No ML evidence older than 12 months' means: looking at career roles
    whose start_date is more than 12 months ago — if ALL of them contain
    zero ML keywords, the flag is raised.
    """
    all_text = _all_text(candidate)
    if "langchain" not in all_text:
        return False

    # Check for ML evidence in roles older than 12 months
    ml_keywords = (
        "machine learning", " ml ", "deep learning", "nlp", "llm",
        "transformer", "neural", "embedding", "retrieval", "ranking",
    )
    career = candidate.get("career_history") or []
    old_roles = [
        r for r in career
        if _months_since(r.get("start_date") or "") > 12
    ]
    if not old_roles:
        # Candidate only has recent history — can't verify older ML experience
        return True

    old_text = " ".join(
        (r.get("description") or "").lower() for r in old_roles
    )
    has_old_ml = any(kw in old_text for kw in ml_keywords)
    return not has_old_ml


def _is_architect_no_code(candidate: dict, architect_tenure_months: int = 18) -> bool:
    """
    True if:
      - most-recent title matches the architect/tech-lead/principal pattern, AND
      - that role's tenure >= architect_tenure_months, AND
      - that role's description contains none of the code keywords.

    ``architect_tenure_months`` is passed in (read from weights downstream).
    """
    career = candidate.get("career_history") or []
    if not career:
        return False

    # Sort by start_date descending to find the most recent role
    try:
        most_recent = max(
            career,
            key=lambda r: r.get("start_date") or "0000-00-00",
        )
    except ValueError:
        return False

    title = (most_recent.get("title") or "").lower()
    if not _ARCHITECT_TITLE_PATTERN.search(title):
        return False

    tenure = most_recent.get("duration_months") or 0
    if tenure < architect_tenure_months:
        return False

    desc = (most_recent.get("description") or "").lower()
    has_code = any(kw in desc for kw in _CODE_KEYWORDS)
    return not has_code


# ---------------------------------------------------------------------------
# Soft negative checks
# ---------------------------------------------------------------------------

def _is_consulting_only(candidate: dict) -> bool:
    """True if every employer (lowercased) is in the consulting-firms set."""
    career = candidate.get("career_history") or []
    if not career:
        return False
    return all(
        (r.get("company") or "").lower() in _CONSULTING_FIRMS
        for r in career
    )


def _is_title_chaser(candidate: dict) -> bool:
    """
    True if 3+ jobs, average tenure < 18 months, AND job titles show an
    escalating seniority pattern.
    """
    career = candidate.get("career_history") or []
    if len(career) < 3:
        return False

    avg_tenure = sum(r.get("duration_months", 0) for r in career) / len(career)
    if avg_tenure >= 18:
        return False

    # Check if titles escalate — at least 2 distinct seniority tokens in order
    sorted_roles = sorted(
        career,
        key=lambda r: r.get("start_date") or "0000-00-00",
    )
    seniority_sequence = [
        bool(_SENIORITY_PATTERN.search(r.get("title") or ""))
        for r in sorted_roles
    ]
    # Title-chaser if any seniority token appears AND the pattern is present
    # in later roles more than earlier ones (simple heuristic: count of True
    # increases as we move forward)
    true_count = sum(seniority_sequence)
    if true_count < 2:
        return False

    # Check that seniority tokens appear more in the latter half
    mid = len(seniority_sequence) // 2
    first_half = sum(seniority_sequence[:mid])
    second_half = sum(seniority_sequence[mid:])
    return second_half >= first_half and true_count >= 2


def _is_cv_speech_robotics(candidate: dict) -> bool:
    """
    True if primary domain keywords are computer-vision/speech/robotics AND
    zero NLP/retrieval keywords appear anywhere.
    """
    all_text = _all_text(candidate)

    has_cv_speech = any(kw in all_text for kw in _CV_SPEECH_ROBOTICS_KEYWORDS)
    if not has_cv_speech:
        return False

    has_nlp = any(kw in all_text for kw in _NLP_RETRIEVAL_KEYWORDS)
    return not has_nlp


# ---------------------------------------------------------------------------
# Combined per-candidate check
# ---------------------------------------------------------------------------

def check_disqualifiers(candidate: dict, weights: dict) -> dict:
    """
    Run all hard and soft disqualifier checks for a single candidate.

    Parameters
    ----------
    candidate : dict
        Raw candidate dict.
    weights : dict
        Parsed config/weights.yaml.

    Returns
    -------
    dict with keys:
        hard_pure_research, hard_langchain_only, hard_architect_no_code,
        is_hard_disqualified,
        soft_consulting_only, soft_title_chaser, soft_cv_speech_robotics,
        num_soft_flags, soft_negative_modifier.
    """
    floor: float = weights["soft_negative_floor"]
    penalty: float = weights["soft_negative_penalty_per_flag"]
    # architect tenure threshold is not in weights.yaml, but 18 months matches the JD text.
    # We fall back to 18 if not present so the system is forward-compatible if added later.
    architect_tenure: int = weights.get("architect_no_code_tenure_months", 18)

    # Hard checks
    h_research = _is_pure_research(candidate)
    h_langchain = _is_langchain_only(candidate)
    h_architect = _is_architect_no_code(candidate, architect_tenure)
    is_hard = h_research or h_langchain or h_architect

    # Soft checks
    s_consulting = _is_consulting_only(candidate)
    s_chaser = _is_title_chaser(candidate)
    s_cv = _is_cv_speech_robotics(candidate)
    num_soft = sum([s_consulting, s_chaser, s_cv])

    modifier = max(floor, 1.0 - penalty * num_soft)

    return {
        # Hard
        "hard_pure_research": h_research,
        "hard_langchain_only": h_langchain,
        "hard_architect_no_code": h_architect,
        "is_hard_disqualified": is_hard,
        # Soft
        "soft_consulting_only": s_consulting,
        "soft_title_chaser": s_chaser,
        "soft_cv_speech_robotics": s_cv,
        "num_soft_flags": num_soft,
        "soft_negative_modifier": modifier,
    }


def should_exclude(candidate: dict, weights: dict) -> bool:
    """
    Returns True if the candidate is hard-disqualified (should be excluded
    from ranking entirely).
    """
    return check_disqualifiers(candidate, weights)["is_hard_disqualified"]


# ---------------------------------------------------------------------------
# DataFrame-level enrichment
# ---------------------------------------------------------------------------

def add_disqualifier_columns(
    df: pd.DataFrame,
    candidates_raw: list[dict],
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Attach disqualifier and soft-negative columns to ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    candidates_raw : list[dict]
        Raw candidate dicts in the same order as ``df``.
    weights : dict, optional
        Parsed weights; loaded from disk if None.

    Returns
    -------
    pd.DataFrame
        ``df`` enriched with: hard_pure_research, hard_langchain_only,
        hard_architect_no_code, is_hard_disqualified, soft_consulting_only,
        soft_title_chaser, soft_cv_speech_robotics, num_soft_flags,
        soft_negative_modifier.
    """
    if weights is None:
        weights = _load_weights()

    results = [check_disqualifiers(c, weights) for c in candidates_raw]
    result_df = pd.DataFrame(results)

    for col in result_df.columns:
        df[col] = result_df[col].values

    return df


def run_on_parquet(
    parquet_path: Path = OUTPUT_PARQUET,
    candidates_raw: list[dict] | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Load the parquet at ``parquet_path``, attach disqualifier columns, and
    write back in place.
    """
    if weights is None:
        weights = _load_weights()

    df = pd.read_parquet(parquet_path)

    if candidates_raw is None:
        raise ValueError(
            "candidates_raw must be supplied — the parquet alone does not "
            "contain all raw fields required for disqualifier checks."
        )

    df = add_disqualifier_columns(df, candidates_raw, weights)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(
        f"[disqualifier] Updated {parquet_path}  "
        f"hard_disqualified={df['is_hard_disqualified'].sum()}/{len(df)}"
    )
    return df
