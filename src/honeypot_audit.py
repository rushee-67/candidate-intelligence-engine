"""
Honeypot Audit Module.

Detects and flags honeypot candidates by looking for logical anomalies in their
profiles.  The submission specification warns that ~80 honeypot profiles exist
in the full dataset (impossible career durations, inflated skill levels with
minimal usage, etc.).

A candidate is classified as a honeypot when it accumulates honeypot_min_flags_to_exclude
or more individual anomaly flags (threshold read from config/weights.yaml).

Anomaly checks implemented
--------------------------
a) Expert-with-no-time  — any skill has proficiency=expert AND duration_months
   is below honeypot_expert_duration_threshold_months.
b) Career-years mismatch — |sum_career_months - years_exp*12| exceeds
   honeypot_career_months_tolerance.
c) Duration vs date-span — any role's stated duration_months > actual calendar
   span of that role (start_date .. end_date/today) + 1 month tolerance.
d) Education-before-employment — any education end_year > the start year of
   the candidate's very first job.

All thresholds are read from config/weights.yaml — no hardcoded numbers.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "weights.yaml"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "artifacts" / "candidate_features.parquet"


def _load_weights() -> dict:
    with open(CONFIG_PATH, "rb") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Anomaly check helpers  (operate on the raw candidate dict)
# ---------------------------------------------------------------------------

def _check_expert_no_time(candidate: dict, threshold_months: float) -> bool:
    """
    Flag True if ANY skill has proficiency=expert AND duration_months < threshold.
    """
    for skill in candidate.get("skills") or []:
        if skill.get("proficiency") == "expert":
            dur = skill.get("duration_months") or 0
            if dur < threshold_months:
                return True
    return False


def _check_career_years_mismatch(
    sum_career_months: int,
    years_exp: float,
    tolerance: float,
) -> bool:
    """
    Flag True if the stated years_of_experience is inconsistent with the sum
    of all role duration_months by more than tolerance.
    """
    expected_months = years_exp * 12.0
    return abs(sum_career_months - expected_months) > tolerance


def _check_duration_vs_datespan(candidate: dict) -> bool:
    """
    Flag True if ANY role's stated duration_months exceeds the actual calendar
    span (start_date → end_date or today) by more than 1 month.
    """
    today = date.today()
    for role in candidate.get("career_history") or []:
        try:
            start = date.fromisoformat(role["start_date"])
        except (KeyError, ValueError, TypeError):
            continue

        end_raw = role.get("end_date")
        try:
            end = date.fromisoformat(end_raw) if end_raw else today
        except (ValueError, TypeError):
            end = today

        actual_months = (end.year - start.year) * 12 + (end.month - start.month)
        stated = role.get("duration_months") or 0
        if stated > actual_months + 1:
            return True
    return False


def _check_edu_before_work(candidate: dict) -> bool:
    """
    Flag True if any education entry ended AFTER the candidate's earliest job
    start year — i.e. they appear to have been working before they graduated.
    This is a light check; common for part-time study, so it only contributes
    one flag rather than being decisive on its own.
    """
    career = candidate.get("career_history") or []
    education = candidate.get("education") or []

    if not career or not education:
        return False

    try:
        first_job_year = min(
            int(r["start_date"][:4])
            for r in career
            if r.get("start_date")
        )
    except (ValueError, TypeError):
        return False

    edu_end_years = [e.get("end_year") for e in education if e.get("end_year")]
    if not edu_end_years:
        return False

    # Only flag if education finished MORE than 1 year after the first job
    # (>1 year is unusual; <=1 yr overlap is plausible for final-year / part-time)
    return any(y > first_job_year + 1 for y in edu_end_years)


# ---------------------------------------------------------------------------
# Per-candidate audit
# ---------------------------------------------------------------------------

def audit_candidate(candidate: dict, weights: dict) -> dict:
    """
    Run all four anomaly checks and return a dict with individual check results
    and the total flag count.

    Parameters
    ----------
    candidate : dict
        Raw candidate dict.
    weights : dict
        Parsed config/weights.yaml.

    Returns
    -------
    dict
        Keys: flag_expert_no_time, flag_career_mismatch, flag_duration_span,
              flag_edu_before_work, total_flags, is_honeypot.
    """
    expert_thresh = weights["honeypot_expert_duration_threshold_months"]
    career_tol = weights["honeypot_career_months_tolerance"]
    min_flags = weights["honeypot_min_flags_to_exclude"]

    years_exp = (candidate.get("profile") or {}).get("years_of_experience", 0.0)
    sum_months = sum(
        r.get("duration_months", 0)
        for r in (candidate.get("career_history") or [])
    )

    flag_a = _check_expert_no_time(candidate, expert_thresh)
    flag_b = _check_career_years_mismatch(sum_months, years_exp, career_tol)
    flag_c = _check_duration_vs_datespan(candidate)
    flag_d = _check_edu_before_work(candidate)

    total = sum([flag_a, flag_b, flag_c, flag_d])
    return {
        "flag_expert_no_time": flag_a,
        "flag_career_mismatch": flag_b,
        "flag_duration_span": flag_c,
        "flag_edu_before_work": flag_d,
        "total_flags": total,
        "is_honeypot": total >= min_flags,
    }


def is_honeypot(candidate: dict, weights: dict) -> bool:
    """
    Convenience wrapper — returns True if the candidate should be treated as a
    honeypot.
    """
    return audit_candidate(candidate, weights)["is_honeypot"]


# ---------------------------------------------------------------------------
# DataFrame-level enrichment
# ---------------------------------------------------------------------------

def add_honeypot_column(
    df: pd.DataFrame,
    candidates_raw: list[dict],
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Attach an ``is_honeypot`` boolean column (and individual flag columns) to
    ``df`` by running audit_candidate() on each raw record.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (output of precompute_features).
    candidates_raw : list[dict]
        The original raw candidate dicts in the same order as ``df``.
    weights : dict, optional
        Loaded weights.  Loaded from disk if None.

    Returns
    -------
    pd.DataFrame
        ``df`` with additional columns: flag_expert_no_time, flag_career_mismatch,
        flag_duration_span, flag_edu_before_work, total_flags, is_honeypot.
    """
    if weights is None:
        weights = _load_weights()

    audit_results = [audit_candidate(c, weights) for c in candidates_raw]
    audit_df = pd.DataFrame(audit_results)

    for col in audit_df.columns:
        df[col] = audit_df[col].values

    return df


def run_on_parquet(
    parquet_path: Path = OUTPUT_PARQUET,
    candidates_raw: list[dict] | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Load the parquet at ``parquet_path``, attach honeypot columns, and write
    the updated DataFrame back in place.

    ``candidates_raw`` must be provided because the parquet does not store every
    raw field needed for the checks (e.g. skill proficiency, exact dates).
    """
    if weights is None:
        weights = _load_weights()

    df = pd.read_parquet(parquet_path)

    if candidates_raw is None:
        raise ValueError(
            "candidates_raw must be supplied — the parquet alone does not "
            "contain all raw fields required for honeypot checks."
        )

    df = add_honeypot_column(df, candidates_raw, weights)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(
        f"[honeypot] Updated {parquet_path}  "
        f"honeypots={df['is_honeypot'].sum()}/{len(df)}"
    )
    return df
