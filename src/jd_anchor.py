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
# Canonical skill names that correspond to the JD's "Things you absolutely need"
# and "Things we'd like you to have" sections.  The names here must be
# case-insensitively comparable against a candidate's skills[*].name field.
# ---------------------------------------------------------------------------
JD_REQUIRED_SKILLS: list[str] = [
    # Core retrieval / embedding systems
    "Embeddings",
    "Sentence Transformers",
    "Vector Search",
    "Information Retrieval",
    "Semantic Search",
    "BM25",
    "Hybrid Search",
    # Vector DBs / indexes
    "Pinecone",
    "Weaviate",
    "Qdrant",
    "Milvus",
    "FAISS",
    "Elasticsearch",
    "OpenSearch",
    # Ranking / recommendation
    "Ranking",
    "Recommendation Systems",
    "Learning to Rank",
    "XGBoost",
    # LLM / NLP
    "NLP",
    "LLM",
    "Hugging Face Transformers",
    "Fine-tuning LLMs",
    "LoRA",
    "QLoRA",
    "PEFT",
    "RAG",
    "Retrieval Augmented Generation",
    # Evaluation
    "NDCG",
    "A/B Testing",
    # Core platform / infra
    "Python",
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
]

# Set version for O(1) membership tests (names lower-cased for matching)
JD_REQUIRED_SKILLS_LOWER: set[str] = {s.lower() for s in JD_REQUIRED_SKILLS}

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
        - required_skills (list[str]): Canonical JD required/preferred skill names.
        - required_skills_lower (set[str]): Lower-cased version for fast lookup.
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
