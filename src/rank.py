"""
Rank Pipeline — main entry point for the Redrob candidate ranking challenge.

Usage
-----
    python src/rank.py --candidates resources/candidates.jsonl --out submission.csv

What it does
------------
1.  Loads precomputed artifacts (parquet + embeddings) from data/artifacts/.
    If candidates.jsonl is provided and the parquet is missing/stale, it runs
    the full precompute pipeline first.
2.  Runs scoring via src/scoring.py.
3.  Filters out honeypots and hard-disqualified candidates (score forced to 0
    in compute_scores, but they are also excluded from the top-100 list).
4.  Sorts by final_score DESC, tie-breaks by candidate_id ASC.
5.  Takes the top 100, assigns ranks 1-100.
6.  Generates a reasoning string for each top-100 via src/reasoning.py.
7.  Writes: candidate_id,rank,score,reasoning  (CSV, UTF-8).
8.  Runs resources/validate_submission.py on the output and prints result.
9.  Prints total wall-clock time.

Performance
-----------
Designed to complete the full pipeline (precompute + embed + score + rank)
in under 5 minutes on 100K candidates on a 16 GB RAM CPU machine:
  - Streaming JSONL ingest (orjson, never fully in RAM).
  - Vectorised numpy cosine similarity (one matrix multiply for all N candidates).
  - SentenceTransformer encode with batch_size=256, progress bar.
  - All per-row loops avoided in scoring via numpy broadcasting.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root on sys.path so that "python src/rank.py" works from root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import orjson
import pandas as pd
import yaml

from src.scoring import compute_scores, _load_weights
from src.reasoning import generate_reasoning, dominant_component

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_CANDIDATES = PROJECT_ROOT / "resources" / "candidates.jsonl"
DEFAULT_OUTPUT     = PROJECT_ROOT / "submission.csv"
ARTIFACTS_DIR      = PROJECT_ROOT / "data" / "artifacts"
PARQUET_PATH       = ARTIFACTS_DIR / "candidate_features.parquet"
CAND_EMBEDDINGS    = ARTIFACTS_DIR / "candidate_embeddings.npy"
JD_EMBEDDING       = ARTIFACTS_DIR / "jd_embedding.npy"
VALIDATE_SCRIPT    = PROJECT_ROOT / "resources" / "validate_submission.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts using orjson."""
    candidates: list[dict] = []
    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                candidates.append(orjson.loads(raw))
            except orjson.JSONDecodeError as exc:
                print(f"[rank] Skipping malformed line {lineno}: {exc}", file=sys.stderr)
    return candidates


def _ensure_artifacts(candidates_path: Path, weights: dict) -> list[dict]:
    """
    If any precomputed artifact is missing, run the full precompute pipeline.
    Returns the raw candidate list (needed for reasoning).
    """
    missing = not PARQUET_PATH.exists() or \
              not CAND_EMBEDDINGS.exists() or \
              not JD_EMBEDDING.exists()

    if missing:
        print("[rank] Precomputed artifacts missing — running full pipeline…")
        t = time.perf_counter()

        from src.precompute_features import (
            _iter_jsonl, extract_candidate_features,
            OUTPUT_PARQUET, OUTPUT_DIR,
        )
        from src.honeypot_audit import add_honeypot_column
        from src.disqualifier_filters import add_disqualifier_columns
        from src.precompute_features import run_embeddings

        raw: list[dict] = []
        rows: list[dict] = []
        total = 0
        for candidate in _iter_jsonl(candidates_path):
            raw.append(candidate)
            rows.append(extract_candidate_features(candidate, weights))
            total += 1
            if total % 10_000 == 0:
                print(f"  … flattened {total:,} candidates")

        print(f"  Flattening done: {total:,} candidates in {time.perf_counter()-t:.1f}s")

        df = pd.DataFrame(rows)
        df["last_active_date"] = pd.to_datetime(df["last_active_date"], errors="coerce")
        for col in ("open_to_work_flag", "willing_to_relocate",
                    "verified_email", "verified_phone", "linkedin_connected"):
            df[col] = df[col].astype(bool)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")

        df = add_honeypot_column(df, raw, weights)
        df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
        print(f"  Honeypots: {df['is_honeypot'].sum()}/{len(df)}")

        df = add_disqualifier_columns(df, raw, weights)
        df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
        print(f"  Hard-disqualified: {df['is_hard_disqualified'].sum()}/{len(df)}")

        run_embeddings(parquet_path=OUTPUT_PARQUET)
        return raw

    # Artifacts exist — still need raw candidates for reasoning
    print("[rank] Loading raw candidates for reasoning…")
    return _load_jsonl(candidates_path)


def _write_csv(output_path: Path, rows: list[dict]) -> None:
    """Write the submission CSV in the format expected by validate_submission.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                f"{row['score']:.6f}",
                row["reasoning"],
            ])


def _validate(output_path: Path) -> bool:
    """Run validate_submission.py and return True if valid."""
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), str(output_path)],
        capture_output=True, text=True,
    )
    print()
    print("─" * 60)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    print("─" * 60)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def run_ranking(
    candidates_path: Path = DEFAULT_CANDIDATES,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """
    Full ranking pipeline.  Returns the output CSV path.
    """
    wall_start = time.perf_counter()

    weights = _load_weights()

    # ---- Step 1: Ensure artifacts / load raw candidates -------------------
    t = time.perf_counter()
    raw_candidates = _ensure_artifacts(candidates_path, weights)
    cand_index = {c["candidate_id"]: c for c in raw_candidates}
    print(f"[rank] Raw candidates loaded: {len(raw_candidates):,}  ({time.perf_counter()-t:.1f}s)")

    # ---- Step 2: Load artifacts -------------------------------------------
    t = time.perf_counter()
    df = pd.read_parquet(PARQUET_PATH)
    cand_embs = np.load(CAND_EMBEDDINGS)
    jd_emb    = np.load(JD_EMBEDDING)
    print(f"[rank] Artifacts loaded: parquet={df.shape}, emb={cand_embs.shape}  ({time.perf_counter()-t:.1f}s)")

    # ---- Step 3: Score all candidates -------------------------------------
    t = time.perf_counter()
    scored = compute_scores(df, cand_embs, jd_emb, weights)
    print(f"[rank] Scoring done: {len(scored):,} candidates  ({time.perf_counter()-t:.1f}s)")

    # ---- Step 4: Build feature index for reasoning ------------------------
    feature_index: dict[str, dict] = {
        row["candidate_id"]: row
        for row in df.to_dict(orient="records")
    }

    # ---- Step 5: Exclude, sort, take top 100 ------------------------------
    # Honeypots and hard-disqualified already have score=0, but exclude
    # them explicitly so they never appear in the ranked list.
    eligible = scored[
        (~scored["is_honeypot"].astype(bool)) &
        (~scored["is_hard_disqualified"].astype(bool))
    ].copy()

    eligible = eligible.sort_values(
        by=["final_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    top100 = eligible.head(100).reset_index(drop=True)
    print(f"[rank] Eligible candidates: {len(eligible):,}  →  top-100 selected")
    if len(top100) < 100:
        print(
            f"[rank] WARNING: only {len(top100)} eligible candidates; "
            "submission will fail validation (need exactly 100).",
            file=sys.stderr,
        )

    # ---- Step 6: Generate reasoning ---------------------------------------
    t = time.perf_counter()
    output_rows: list[dict] = []
    reasonings_seen: set[str] = set()

    for rank, score_row in enumerate(top100.to_dict(orient="records"), start=1):
        cid = score_row["candidate_id"]
        candidate    = cand_index.get(cid, {})
        feature_row  = feature_index.get(cid, {})
        # Merge soft-flag columns into feature_row from the scored row
        for col in ("soft_consulting_only", "soft_title_chaser", "soft_cv_speech_robotics",
                    "notice_period_days", "offer_acceptance_rate"):
            if col not in feature_row and col in score_row:
                feature_row[col] = score_row[col]

        reasoning = generate_reasoning(candidate, feature_row, score_row, reasonings_seen)
        reasonings_seen.add(reasoning)

        output_rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        score_row["final_score"],
            "reasoning":    reasoning,
        })

    unique_ratio = len(reasonings_seen) / len(output_rows) if output_rows else 0
    print(
        f"[rank] Reasoning generated: {len(output_rows)} rows, "
        f"{len(reasonings_seen)} unique ({unique_ratio:.0%})  "
        f"({time.perf_counter()-t:.1f}s)"
    )

    # ---- Step 7: Write CSV ------------------------------------------------
    _write_csv(output_path, output_rows)
    print(f"[rank] Submission written → {output_path}")

    # ---- Step 8: Validate -------------------------------------------------
    valid = _validate(output_path)

    # ---- Step 9: Print totals ---------------------------------------------
    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n{'✓' if valid else '✗'} Total wall-clock time: {wall_elapsed:.1f}s")

    # Print top-10 preview
    print("\nTop-10 by score:")
    for row in output_rows[:10]:
        print(f"  Rank {row['rank']:3d}  {row['candidate_id']}  "
              f"score={row['score']:.4f}  {row['reasoning']}")

    if not valid:
        sys.exit(1)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redrob candidate ranking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Path to candidates JSONL file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ranking(candidates_path=args.candidates, output_path=args.out)
