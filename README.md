# Redrob Candidate Ranker

Ranks 100,000 candidates against a job description for a Senior AI Engineer role. Filters honeypots and disqualified profiles, then scores eligible candidates using semantic embeddings + skill trust + experience + availability signals. Outputs the top 100 as a validated CSV with one unique reasoning string per candidate.

---

## 1. Project Structure

* **`config/weights.yaml`**: Holds all scoring weights, threshold constants, location mapping weights, and honeypot flags.
* **`sandbox/app.py`**: Streamlit application providing a web interface to upload small candidate JSON files and visually inspect scored and ranked results.
* **`src/__init__.py`**: Initializes the `src` package.
* **`src/disqualifier_filters.py`**: Implementation of hard exclusions (pure research, LangChain-only, non-coding architects) and soft negative flags (consulting-only, title chasing, domain mismatch).
* **`src/honeypot_audit.py`**: Audits candidates for profile logical anomalies (expert skill with no duration, years of experience mismatches, impossible career durations, study-work timeline overlaps).
* **`src/jd_anchor.py`**: Defines the target job description (JD) responsibility text, normalized required skills, acceptable locations, and relevance keywords.
* **`src/precompute_features.py`**: Handles streaming JSONL candidate ingestion, parses features, runs honeypot and disqualifier audits, and batch-encodes candidate text profiles using a SentenceTransformer.
* **`src/rank.py`**: Primary entry point to load precomputed features and embeddings, run ranking scoring, select top 100 candidates, resolve duplicate reasonings, and output `submission.csv`.
* **`src/reasoning.py`**: Formulates unique, data-backed 1-2 sentence summaries justifying candidate ranks based strictly on their profile fields (preventing hallucinations).
* **`src/scoring.py`**: Computes final scores using vector similarity of candidate profiles vs the JD, normalized experience durations, skill trust scores, location suitability, and engagement signals.
* **`submission_metadata.yaml`**: Metadata file declaring reproduce commands, compute settings, approach methodology, and declarations.
* **`tests/__init__.py`**: Initializer for the `tests` package.
* **`tests/fixtures.py`**: Contains 6 synthetic candidate profiles mimicking specific test archetypes (ideal, keyword stuffer, honeypot, tier 5 fit, title chaser, consulting only).
* **`tests/test_ranking_order.py`**: Integration tests confirming that candidate ordering, honeypot exclusion, and soft penalties function correctly against mock profiles.

---

## 2. Setup

Install dependencies on a CPU-only environment from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 3. How to Run

Execute the pipeline commands in order:

1. **Precompute Features and Embeddings**
   ```bash
   python src/precompute_features.py
   ```
   * *Produces:* `data/artifacts/candidate_features.parquet`, `data/artifacts/candidate_embeddings.npy`, and `data/artifacts/jd_embedding.npy`. (Embeds the JD and candidate profiles using `all-MiniLM-L6-v2`).

2. **Run Scoring and Ranking**
   ```bash
   python src/rank.py --candidates resources/candidates.jsonl --out submission.csv
   ```
   * *Produces:* `submission.csv` (contains the top 100 ranked candidates with `candidate_id`, `rank`, `score`, and `reasoning`).

3. **Verify Submission Formatting**
   ```bash
   python resources/validate_submission.py submission.csv
   ```
   * *Produces:* Prints formatting confirmation outputs.

4. **Launch Local Streamlit Sandbox**
   ```bash
   streamlit run sandbox/app.py
   ```
   * *Produces:* Opens a local port in the web browser to upload candidate JSON files and view ranked tables interactively.

---

## 4. How the Scoring Works

The final score is calculated using the following formulas and parameters:

$$\text{base-fit} = 0.40 \cdot \text{title-score} + 0.25 \cdot \text{skill-trust-norm} + 0.20 \cdot \text{experience-score} + 0.10 \cdot \text{soft-modifier} + 0.05 \cdot \text{location-score}$$

### Scoring Component Details:
1. **Title Score (`weight: 0.40`):** Map cosine similarity of candidate text-blob vs JD responsibility text from $[-1, 1]$ to $[0, 1]$:
   $$\text{title-score} = \frac{\text{cosine-sim} + 1.0}{2.0}$$
2. **Skill Trust (`weight: 0.25`):** Normalizes total skill trust score:
   $$\text{skill-trust-norm} = \min\left(1.0, \frac{\text{skill-trust-score}}{2.0}\right)$$
   Where each JD-matched skill $i$ is scored in `src/precompute_features.py` as:
   $$\text{trust}_i = \text{proficiency-weight} \cdot \left(1.0 + \text{endorse-boost}\right) \cdot \text{dur-factor}$$
   The exact lines used in `src/precompute_features.py` to compute `endorse_boost` and the resulting `trust_i` are:
   ```python
   endorse_boost = min((skill.get("endorsements") or 0) / endorse_scale, 1.0)
   dur_factor = min((skill.get("duration_months") or 0) / max(dur_cap, 1.0), 1.0)
   trust_i = pw * (1.0 + endorse_boost) * dur_factor
   ```
   *(proficiency weights: beginner = 0.25, intermediate = 0.50, advanced = 0.75, expert = 1.00)*
3. **Experience Score (`weight: 0.20`):** Linear cap at 60 months of relevant experience:
   $$\text{experience-score} = \min\left(1.0, \frac{\text{relevant-exp-months}}{60}\right)$$
4. **Soft-Negative Modifier (`weight: 0.10`):** Penalty multiplier applied from disqualifier flags:
   $$\text{soft-modifier} = \max\left(0.4, 1.0 - 0.15 \cdot \text{num-soft-flags}\right)$$
5. **Location Score (`weight: 0.05`):** Suitability score mapped by city (Noida/Pune = 1.0; Delhi/NCR/Hyderabad/Mumbai = 0.8; Other India = 0.5; Outside India = 0.1).

### Availability Multiplier:
Candidates are scaled by active and behavioral engagement signals, floored at `0.5` and capped at `1.0`:
$$\text{availability-multiplier} = 0.5 + 0.5 \cdot \text{mean}(\text{recency}, \text{engagement})$$
* **Recency:** `1.0` if candidate active within last 30 days, linear decay to `0.0` at 180 days.
* **Engagement:** Mean of recruiter response rate, interview completion rate, candidate open to work flag (`1.0` if True, `0.3` if False), and offer acceptance rate (omitted if `-1.0`).

### Final Score:
$$\text{final-score} = \begin{cases} 0.0 & \text{if } \text{is-honeypot} \text{ or } \text{is-hard-disqualified} \\ \text{base-fit} \cdot \text{availability-multiplier} & \text{otherwise} \end{cases}$$

---

## 5. Honeypot Detection Rules

Candidates are excluded (score forced to 0.0) if they trigger **2 or more** of the following anomaly rules:
1. **Expert with No Time:** Candidate claims a skill at `expert` proficiency but has usage `duration_months < 6`.
2. **Stated Years Mismatch:** Stated `years_of_experience` multiplied by 12 differs from the sum of all career history role durations by more than `18` months.
3. **Stated Duration vs Date-Span:** A career role's stated `duration_months` exceeds the actual calendar span of that role (from start to end date/today) by more than `1` month.
4. **Education Timeline Overlap:** Stated education `end_year` is greater than the start year of their very first job by more than `1` year.

---

## 6. Disqualification Filters

### Hard Disqualifiers (Forced Exclusions):
* **Pure Research Only:** Every career role is in `academia` or `research` and no production keywords (`production`, `deployed`, `a/b`, `inference`, `serving`, `real-time`, `pipeline`, `api`) are mentioned in descriptions.
* **LangChain Only:** The profile text mentions `langchain` but no ML evidence (`machine learning`, `nlp`, `llm`, `transformer`, etc.) older than `12` months.
* **Architect with No Code:** Most recent job title matches `architect`, `tech lead`, or `principal` with tenure `>= 18` months, and description contains zero coding keywords (`python`, `pytorch`, `tensorflow`, `sklearn`, `code`, `implement`, `build`, etc.).

### Soft Negatives (Penalty Modifiers):
* **Consulting Only:** Every job company listed is a known large IT services firm (`TCS`, `Infosys`, `Wipro`, `Accenture`, `Cognizant`, `Capgemini`, `HCL`, `Tech Mahindra`).
* **Title Chaser:** Candidate has `3+` jobs, average role tenure `< 18` months, and job titles escalate in seniority tokens (`senior`, `staff`, `principal`, `lead`, `architect`).
* **Computer Vision / Speech Focus:** Profile contains CV/speech/robotics keywords with zero NLP/retrieval keywords.

---

## 7. Test Suite

The integration test suite ([`tests/test_ranking_order.py`](file:///home/user/Documents/redrob-ranker/tests/test_ranking_order.py)) runs the pipeline against 6 synthetic candidates representing distinct archetypes:
1. **`CAND_9000001` (Ideal):** An ML Engineer with strong relevant experience and engagement. Asserts score is `> 0.0`, is not flagged, and matches relevant months.
2. **`CAND_9000002` (Keyword Stuffer):** Stuffed skills but with `duration_months = 0`. Asserts they get `skill_trust_score = 0.0` and rank lower than high-experience candidates.
3. **`CAND_9000003` (Honeypot):** Confirms they trigger `expert_no_time` and `career_mismatch`, are classified as a honeypot, and receive a score of `0.0`.
4. **`CAND_9000004` (Tier 5 Fit):** Data Engineer with relevant keywords in role descriptions but none in skills. Asserts they score highly on experience and rank correctly in order: `ideal` > `tier5_fit` > `keyword_stuffer`.
5. **`CAND_9000005` (Title Chaser):** Asserts they are flagged for `soft_title_chaser`.
6. **`CAND_9000006` (Consulting Only):** Asserts they are flagged for `soft_consulting_only` and their `soft_negative_modifier < 1.0`.

---

## 8. JD Anchor Text

* **Anchor Definition:** A 6-sentence paragraph in [`src/jd_anchor.py`](file:///home/user/Documents/redrob-ranker/src/jd_anchor.py) extracted from the actual responsibilities ("What you'd actually be doing") section of the job description.
* **Why it matters:** It serves as the target reference vector for similarity mapping. By comparing candidate profile text blocks against actual job responsibilities—rather than isolated skill keywords—the system scores candidates based on actual role suitability and filters out keyword stuffers who have the terminology but lack relevant project history.

---

## 9. Compute Profile

- Stage 1: BAAI/bge-base-en-v1.5 bi-encoder (768-dim), all 100K
- Stage 2: ms-marco-MiniLM-L-12-v2 cross-encoder, top 300
- Online Ranking Pipeline: 2m58s (~3 minutes) on CPU
- Memory: <1.5GB peak
