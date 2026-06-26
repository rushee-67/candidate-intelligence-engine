from __future__ import annotations

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
    "sentence-transformers", "torch", "pandas", 
    "pyarrow", "numpy", "orjson", "pyyaml"], 
    capture_output=True)

"""
Streamlit Sandbox Application.

This file implements the interactive web application used to:
- Load candidate files (JSON) and the job description.
- Trigger the ranking pipeline and visualize the top candidates.
- Display score breakdowns (title, skill trust, experience, location, etc.) for debugging.
- Render candidate details alongside their generated reasoning.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so 'src.*' imports work
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import compute_scores, _load_weights
from src.reasoning import generate_reasoning

def extract_features_on_the_fly(candidates: list[dict], weights: dict) -> pd.DataFrame:
    from src.precompute_features import extract_candidate_features
    from src.honeypot_audit import add_honeypot_column
    from src.disqualifier_filters import add_disqualifier_columns

    rows = []
    for c in candidates:
        rows.append(extract_candidate_features(c, weights))

    df = pd.DataFrame(rows)
    df["last_active_date"] = pd.to_datetime(df["last_active_date"], errors="coerce")
    for col in ("open_to_work_flag", "willing_to_relocate",
                "verified_email", "verified_phone", "linkedin_connected"):
        if col in df.columns:
            df[col] = df[col].astype(bool)

    df = add_honeypot_column(df, candidates, weights)
    df = add_disqualifier_columns(df, candidates, weights)
    return df

@st.cache_resource
def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def encode_on_the_fly(candidate_blobs: list[str]):
    model = load_sentence_transformer()
    from src.jd_anchor import JD_ANCHOR_TEXT
    jd_emb = model.encode(JD_ANCHOR_TEXT, convert_to_numpy=True)
    cand_embs = model.encode(candidate_blobs, convert_to_numpy=True, batch_size=256, show_progress_bar=False)
    return jd_emb, cand_embs

def main():
    # Page setup
    st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
    st.title("Redrob Candidate Ranker")

    # Sidebar
    st.sidebar.title("Configuration")
    
    # 1. Check if precomputed artifacts exist
    parquet_path = PROJECT_ROOT / "data" / "artifacts" / "candidate_features.parquet"
    cand_emb_path = PROJECT_ROOT / "data" / "artifacts" / "candidate_embeddings.npy"
    jd_emb_path = PROJECT_ROOT / "data" / "artifacts" / "jd_embedding.npy"

    artifacts_exist = parquet_path.exists() and cand_emb_path.exists() and jd_emb_path.exists()

    if artifacts_exist:
        st.sidebar.success("Precomputed database loaded (Fast Lookup Mode)")
        st.sidebar.markdown(
            "**Model:** `BAAI/bge-base-en-v1.5` (768-dim) / Stage 2 `MiniLM-L-12`\n\n"
            "**Scoring:** hybrid embedding + skill-trust + experience + availability"
        )
        @st.cache_data
        def load_artifacts():
            df_features = pd.read_parquet(parquet_path)
            cand_embs = np.load(cand_emb_path)
            jd_emb = np.load(jd_emb_path)
            return df_features, cand_embs, jd_emb

        df_features, cand_embs, jd_emb = load_artifacts()
        id_to_index = {cid: idx for idx, cid in enumerate(df_features["candidate_id"])}
    else:
        st.sidebar.warning("Demo Mode: Running on-the-fly extraction & encoding using `all-MiniLM-L6-v2`")
        st.sidebar.markdown(
            "**Model:** `all-MiniLM-L6-v2` (384-dim) on-the-fly\n\n"
            "**Scoring:** hybrid embedding + skill-trust + experience + availability"
        )

    # 2. File uploader
    uploaded_file = st.file_uploader(
        "Upload a candidate sample (.json format)",
        type=["json"]
    )

    if uploaded_file is not None:
        try:
            candidates = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing JSON file: {e}")
            return

        if not isinstance(candidates, list):
            st.error("JSON file must contain a list of candidate records.")
            return

        weights = _load_weights()

        if artifacts_exist:
            # Find matching candidate profiles in precomputed features and embeddings
            uploaded_ids = [c.get("candidate_id") for c in candidates if c.get("candidate_id")]
            indices = [id_to_index[cid] for cid in uploaded_ids if cid in id_to_index]

            if not indices:
                st.warning("None of the uploaded candidate IDs match the precomputed database.")
                return

            # Prepare matching subsets
            uploaded_df = df_features.iloc[indices].copy().reset_index(drop=True)
            uploaded_embs = cand_embs[indices]
        else:
            # Compute everything on-the-fly
            with st.spinner("Extracting candidate features..."):
                uploaded_df = extract_features_on_the_fly(candidates, weights)

            with st.spinner("Encoding profiles on-the-fly..."):
                candidate_blobs = uploaded_df["candidate_text_blob"].fillna("").tolist()
                try:
                    jd_emb, uploaded_embs = encode_on_the_fly(candidate_blobs)
                except Exception as e:
                    st.error(f"Error encoding embeddings on the fly: {e}")
                    return

        # 3. Run Pipeline
        with st.spinner("Calculating fit scores..."):
            scored = compute_scores(uploaded_df, uploaded_embs, jd_emb, weights)

        # Exclude honeypots and hard disqualified from the ranked display list
        eligible = scored[
            (~scored["is_honeypot"].astype(bool)) &
            (~scored["is_hard_disqualified"].astype(bool))
        ].copy()

        # Sort by final score DESC, tie-break by candidate ID ASC
        eligible = eligible.sort_values(
            by=["final_score", "candidate_id"],
            ascending=[False, True]
        ).reset_index(drop=True)

        st.subheader(f"Ranked Candidates ({len(eligible)} eligible out of {len(candidates)} uploaded)")

        if len(eligible) == 0:
            st.info("No candidates match the eligibility criteria.")
            return

        # 4. Generate Reasoning
        cand_index = {c["candidate_id"]: c for c in candidates if "candidate_id" in c}
        feature_index = {
            row["candidate_id"]: row
            for row in uploaded_df.to_dict(orient="records")
        }

        output_rows = []
        reasonings_seen = set()
        for rank, score_row in enumerate(eligible.to_dict(orient="records"), start=1):
            cid = score_row["candidate_id"]
            candidate = cand_index.get(cid, {})
            feature_row = feature_index.get(cid, {})

            # Merge soft-flag columns into feature_row from the scored row
            for col in ("soft_consulting_only", "soft_title_chaser", "soft_cv_speech_robotics",
                        "notice_period_days", "offer_acceptance_rate"):
                if col not in feature_row and col in score_row:
                    feature_row[col] = score_row[col]

            reasoning = generate_reasoning(candidate, feature_row, score_row, reasonings_seen)
            reasonings_seen.add(reasoning)

            output_rows.append({
                "Rank": rank,
                "Candidate ID": cid,
                "Score": round(score_row["final_score"], 6),
                "Reasoning": reasoning
            })

        # Display results in table
        df_ranked = pd.DataFrame(output_rows)
        st.dataframe(df_ranked, use_container_width=True)

if __name__ == '__main__':
    main()
