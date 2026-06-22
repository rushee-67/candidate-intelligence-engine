"""
JD Anchor Extraction Module.

This module is responsible for defining and extracting target criteria (anchors)
from the Job Description. The anchors include:
- Required experience range (e.g., 5 to 9 years).
- Targeted technical skills (dense retrieval, embeddings, vector databases, ranking systems, and eval frameworks).
- Highly preferred locations (Pune, Noida) and acceptable relocation hubs (Delhi NCR, Hyderabad, Mumbai).
- Critical preferences (product-company experience, active platform activity, hands-on production code, and writing focus).
"""

# ---------------------------------------------------------------------------
# JD_ANCHOR_TEXT — 5-6 sentences extracted from the "What you'd actually be
# doing" / responsibilities section of job_description.docx.
# This is the semantic anchor that candidate text blobs are compared against.
# ---------------------------------------------------------------------------
JD_ANCHOR_TEXT: str = (
    "Own the intelligence layer of Redrob's product: the ranking, retrieval, "
    "and matching systems that decide what recruiters see when they search for "
    "candidates and what candidates see when they search for roles. "
    "Ship a v2 ranking system that demonstrably improves recruiter-engagement "
    "metrics using embeddings, hybrid retrieval, and LLM-based re-ranking. "
    "Set up evaluation infrastructure including offline benchmarks, online A/B "
    "testing, and recruiter-feedback loops. "
    "Drive the long-term architecture of candidate-JD matching at scale, "
    "mentoring the next round of hires and working closely with the "
    "recruiter-experience PM. "
    "Audit the current BM25 plus rule-based scoring system and identify the "
    "highest-leverage improvements. "
    "Deep technical depth in modern ML systems including embeddings, retrieval, "
    "ranking, LLMs, and fine-tuning combined with a scrappy product-engineering "
    "attitude to ship fast and iterate with real users."
)

# ---------------------------------------------------------------------------
# JD_REQUIRED_SKILLS — skill names from the JD's "Things you absolutely need"
# and "Things we'd like you to have" sections, normalized to lowercase.
# These are matched case-insensitively against candidate skills[*].name.
# ---------------------------------------------------------------------------
JD_REQUIRED_SKILLS: list[str] = [
    # Core retrieval / embedding systems
    "embeddings",
    "sentence transformers",
    "vector search",
    "information retrieval",
    "semantic search",
    "bm25",
    "hybrid search",
    # Vector DBs / indexes
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "elasticsearch",
    "opensearch",
    # Ranking / recommendation
    "ranking",
    "recommendation systems",
    "learning to rank",
    "xgboost",
    # LLM / NLP
    "nlp",
    "llm",
    "hugging face transformers",
    "fine-tuning llms",
    "lora",
    "qlora",
    "peft",
    "rag",
    "retrieval augmented generation",
    # Evaluation
    "ndcg",
    "a/b testing",
    # Core platform / infra
    "python",
    "pytorch",
    "tensorflow",
    "scikit-learn",
]

# Set version for O(1) membership tests (already lowercase)
JD_REQUIRED_SKILLS_LOWER: set[str] = set(JD_REQUIRED_SKILLS)

# Keywords that appear in the JD as experience signals used by precompute_features
RELEVANCE_KEYWORDS: list[str] = [
    "retrieval",
    "ranking",
    "recommendation",
    "embedding",
    "vector",
    "search",
    "machine learning",
    " ml ",
    "deep learning",
    "nlp",
    "llm",
    "transformer",
    "fine-tun",
    "production",
    "deployed",
    "a/b",
    "inference",
]


def load_jd_anchors() -> dict:
    """
    Returns a structured dictionary of job description anchors used downstream
    in scoring and filtering modules.

    Returns
    -------
    dict
        Keys:
        - jd_anchor_text (str): Semantic anchor text describing the role responsibilities.
        - required_skills (list[str]): Canonical JD required/preferred skill names (lowercase).
        - required_skills_lower (set[str]): Same as required_skills, as a set for fast lookup.
        - relevance_keywords (list[str]): Keywords that signal relevant ML/IR experience.
        - experience_min_years (int): Minimum years of experience considered in range.
        - experience_max_years (int): Maximum years of experience considered in range.
        - preferred_locations (list[str]): City tokens that map to location_scores in weights.yaml.
        - consulting_firms (set[str]): Employer names flagged as large IT service firms.
        - title_escalation_pattern (str): Regex pattern for title-chaser detection.
        - code_keywords (list[str]): Keywords indicating hands-on production coding.
        - relevance_industry_keywords (list[str]): Industry names considered relevant.
    """
    return {
        "jd_anchor_text": JD_ANCHOR_TEXT,
        "required_skills": JD_REQUIRED_SKILLS,
        "required_skills_lower": JD_REQUIRED_SKILLS_LOWER,
        "relevance_keywords": RELEVANCE_KEYWORDS,
        "experience_min_years": 5,
        "experience_max_years": 9,
        "preferred_locations": ["pune", "noida"],
        "consulting_firms": {
            "tcs", "infosys", "wipro", "accenture",
            "cognizant", "capgemini", "hcl", "tech mahindra",
        },
        "title_escalation_pattern": r"senior|staff|principal|lead|architect",
        "code_keywords": [
            "python", "pytorch", "tensorflow", "sklearn", "code",
            "implement", "build", "develop", "wrote", "deployed",
        ],
        "relevance_industry_keywords": [
            "ai/ml", "software", "e-commerce", "fintech",
            "food delivery", "transportation",
        ],
    }
