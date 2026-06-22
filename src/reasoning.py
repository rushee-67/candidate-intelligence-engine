"""
Reasoning Module.

Generates a 1-2 sentence human-readable justification string for each ranked
candidate, using ONLY field values extracted from the candidate's data — no
assumptions, no generated text.

Sentence structure varies by the dominant scoring component:

  title-dominant:
    "{years}y as {current_title}; career history shows {relevant_exp_months}mo
     in retrieval/ML-adjacent roles. {concern}"

  skill-dominant:
    "{skill_count} relevant skills (top: {top_skill} at {proficiency},
     {duration}mo); {years}y total experience. {concern}"

  experience-dominant:
    "{relevant_exp_months}mo hands-on ML/retrieval experience across
     {companies}. {concern}"

Concern clause is appended when any of these is true:
  - any soft flag (consulting_only / title_chaser / cv_speech_robotics)
  - location outside India  → "location may require relocation discussion"
  - notice_period_days > 60 → "notice period N days"
  - offer_acceptance_rate < 0.5 and != -1 → "low offer acceptance rate"
"""

from __future__ import annotations

from src.jd_anchor import JD_REQUIRED_SKILLS_LOWER

# Map of soft-flag column names → human-readable concern text
_SOFT_FLAG_LABELS: dict[str, str] = {
    "soft_consulting_only":     "entire career at IT services firms",
    "soft_title_chaser":        "short average tenure suggests title-chasing",
    "soft_cv_speech_robotics":  "primary domain is CV/speech/robotics, not NLP/IR",
}

_INDIA_COUNTRIES = frozenset({"india", "in"})


# ---------------------------------------------------------------------------
# Public helpers used by rank.py to prepare inputs for generate_reasoning()
# ---------------------------------------------------------------------------

def dominant_component(score_row: dict) -> str:
    """
    Identify which score component contributes the most to base_fit.

    Parameters
    ----------
    score_row : dict
        A single row from the scored DataFrame (as a dict), containing at
        minimum: title_score, skill_trust_norm, experience_score.

    Returns
    -------
    str  — one of "title", "skill", "experience"
    """
    # Weighted contributions  (mirrors scoring.py base_fit formula)
    components = {
        "title":      0.40 * float(score_row.get("title_score", 0.0)),
        "skill":      0.25 * float(score_row.get("skill_trust_norm", 0.0)),
        "experience": 0.20 * float(score_row.get("experience_score", 0.0)),
    }
    return max(components, key=lambda k: components[k])


def _top_jd_skill(candidate: dict) -> tuple[str, str, int]:
    """
    Return (name, proficiency, duration_months) of the highest-trust JD-
    relevant skill possessed by the candidate.

    Trust is: proficiency_rank * duration_months  (endorsements not needed
    for the one-line summary).
    """
    prof_rank = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    best: tuple[str, str, int] | None = None
    best_score = -1

    for skill in candidate.get("skills") or []:
        name = (skill.get("name") or "").strip()
        if name.lower() not in JD_REQUIRED_SKILLS_LOWER:
            continue
        prof = skill.get("proficiency", "beginner")
        dur = int(skill.get("duration_months") or 0)
        score = prof_rank.get(prof, 1) * dur
        if score > best_score:
            best_score = score
            best = (name, prof, dur)

    return best or ("", "", 0)


def _unique_companies(candidate: dict) -> str:
    """Comma-joined list of unique employers (preserving order)."""
    seen: set[str] = set()
    companies: list[str] = []
    for role in candidate.get("career_history") or []:
        co = (role.get("company") or "").strip()
        if co and co not in seen:
            seen.add(co)
            companies.append(co)
    return ", ".join(companies) if companies else "multiple companies"


def _concern_clause(candidate: dict, feature_row: dict) -> str:
    """
    Build the concern suffix string based on soft flags, location, notice,
    and offer-acceptance rate.  Returns "" when no concerns apply.
    """
    signals = candidate.get("redrob_signals") or {}
    profile = candidate.get("profile") or {}
    concerns: list[str] = []

    # --- Soft flags from the feature/scored row ---
    for col, label in _SOFT_FLAG_LABELS.items():
        if feature_row.get(col):
            concerns.append(label)

    # --- Location outside India ---
    country = (profile.get("country") or "").strip().lower()
    if country not in _INDIA_COUNTRIES:
        concerns.append("location may require relocation discussion")

    # --- Notice period ---
    notice = int(signals.get("notice_period_days") or
                 feature_row.get("notice_period_days") or 0)
    if notice > 60:
        concerns.append(f"notice period {notice} days")

    # --- Low offer acceptance rate ---
    oar = float(signals.get("offer_acceptance_rate",
                             feature_row.get("offer_acceptance_rate", -1)) or -1)
    if oar != -1.0 and oar < 0.5:
        concerns.append("low offer acceptance rate")

    if not concerns:
        return ""
    # Capitalise the first concern, join the rest with semicolons
    text = "; ".join(concerns)
    return text[0].upper() + text[1:] + "."


# ---------------------------------------------------------------------------
# Primary public function
# ---------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict,
    feature_row: dict,
    score_row: dict,
) -> str:
    """
    Generate a 1-2 sentence ranking justification for one candidate.

    Parameters
    ----------
    candidate : dict
        Raw candidate dict (as loaded from the JSONL / JSON source).
    feature_row : dict
        The enriched feature row (from the parquet / DataFrame.to_dict()).
        Must contain at minimum: relevant_exp_months, soft_* flag columns,
        notice_period_days, offer_acceptance_rate.
    score_row : dict
        The scored row (output of compute_scores).  Must contain at minimum:
        title_score, skill_trust_norm, experience_score.

    Returns
    -------
    str
        A 1-2 sentence string built exclusively from data field values.
    """
    profile = candidate.get("profile") or {}

    years = float(profile.get("years_of_experience") or
                  feature_row.get("years_exp") or 0)
    years_str = f"{years:.1f}y" if years != int(years) else f"{int(years)}y"

    current_title = (profile.get("current_title") or
                     feature_row.get("current_title") or "").strip()
    rel_months = int(feature_row.get("relevant_exp_months") or 0)

    dom = dominant_component(score_row)
    concern = _concern_clause(candidate, feature_row)

    # ---- Template selection ------------------------------------------------

    if dom == "title":
        sentence = (
            f"{years_str} as {current_title}; "
            f"career history shows {rel_months}mo in retrieval/ML-adjacent roles."
        )

    elif dom == "skill":
        top_name, top_prof, top_dur = _top_jd_skill(candidate)
        skill_count = sum(
            1 for s in (candidate.get("skills") or [])
            if (s.get("name") or "").lower() in JD_REQUIRED_SKILLS_LOWER
            and int(s.get("duration_months") or 0) > 0
        )
        if top_name:
            sentence = (
                f"{skill_count} relevant skill{'s' if skill_count != 1 else ''} "
                f"(top: {top_name} at {top_prof}, {top_dur}mo); "
                f"{years_str} total experience."
            )
        else:
            # Fallback to title template if no matched JD skills found
            sentence = (
                f"{years_str} as {current_title}; "
                f"career history shows {rel_months}mo in retrieval/ML-adjacent roles."
            )

    else:  # experience-dominant
        companies = _unique_companies(candidate)
        sentence = (
            f"{rel_months}mo hands-on ML/retrieval experience across {companies}."
        )

    # ---- Append concern clause ---------------------------------------------
    if concern:
        return f"{sentence} {concern}"
    return sentence


# ---------------------------------------------------------------------------
# Batch helper — generates reasoning for many candidates at once
# ---------------------------------------------------------------------------

def generate_reasoning_batch(
    candidates: list[dict],
    feature_df,          # pd.DataFrame — will not be imported at module level
    scored_df,           # pd.DataFrame
) -> list[str]:
    """
    Vectorised wrapper.  Matches candidates to their feature/score rows by
    candidate_id and calls generate_reasoning() for each.

    Parameters
    ----------
    candidates : list[dict]
        Raw candidate dicts in any order.
    feature_df : pd.DataFrame
        Full enriched feature DataFrame (includes soft_* columns).
    scored_df : pd.DataFrame
        Output of compute_scores().

    Returns
    -------
    list[str]
        Reasoning strings aligned with *scored_df* row order.
    """
    # Index raw candidates by id for O(1) lookup
    cand_index: dict[str, dict] = {
        c["candidate_id"]: c for c in candidates
    }

    feature_index: dict[str, dict] = {
        row["candidate_id"]: row
        for row in feature_df.to_dict(orient="records")
    }

    reasons: list[str] = []
    for score_row in scored_df.to_dict(orient="records"):
        cid = score_row["candidate_id"]
        candidate = cand_index.get(cid, {})
        feature_row = feature_index.get(cid, {})
        reasons.append(generate_reasoning(candidate, feature_row, score_row))

    return reasons
