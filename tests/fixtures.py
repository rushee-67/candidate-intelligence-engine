"""
Test Fixtures Module.

Provides 6 synthetic candidate dicts that are fully compliant with
resources/candidate_schema.json and are designed to stress-test the ranking
pipeline's ability to distinguish between different candidate archetypes.

Fixture design goals
--------------------
CAND_9000001  ideal              — 7y ML Engineer in Pune, strong relevant experience,
                                   JD-required skills with real durations, highly engaged.
CAND_9000002  keyword_stuffer    — HR Manager who listed Python/PyTorch/LangChain/FAISS
                                   as skills but ALL with duration_months=0; career
                                   descriptions are pure HR (no ML keywords).
CAND_9000003  honeypot           — Claims 10y experience but career history totals only
                                   24 months (flag b); also has expert Python with
                                   duration_months=0 (flag a) → 2 flags → is_honeypot=True.
CAND_9000004  tier5_fit          — Data Engineer with no AI keywords in title or skill
                                   names, but every role description mentions
                                   "recommendation system", "ranking model",
                                   "retrieval pipeline" → relevant_exp_months is high.
CAND_9000005  title_chaser       — 3 jobs averaging 10 months each, escalating titles
                                   (Senior → Staff → Principal Sales Manager) at product
                                   companies, NOT open-to-work, last active 150 days ago.
CAND_9000006  consulting_only    — 4 consecutive jobs at TCS / Infosys / Wipro / Accenture
                                   → soft_negative_modifier < 1.0.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helper to build a minimal-but-valid redrob_signals block
# ---------------------------------------------------------------------------

def _signals(
    *,
    last_active: str,
    open_to_work: bool,
    response_rate: float,
    interview_rate: float,
    offer_rate: float = -1,
    completeness: float = 80.0,
    notice: int = 30,
    willing_relocate: bool = True,
    work_mode: str = "hybrid",
    github: float = -1,
    verified_email: bool = True,
    verified_phone: bool = True,
    linkedin: bool = True,
) -> dict:
    return {
        "profile_completeness_score": completeness,
        "signup_date": "2024-01-01",
        "last_active_date": last_active,
        "open_to_work_flag": open_to_work,
        "profile_views_received_30d": 10,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": response_rate,
        "avg_response_time_hours": 4.0,
        "skill_assessment_scores": {},
        "connection_count": 200,
        "endorsements_received": 20,
        "notice_period_days": notice,
        "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
        "preferred_work_mode": work_mode,
        "willing_to_relocate": willing_relocate,
        "github_activity_score": github,
        "search_appearance_30d": 50,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": interview_rate,
        "offer_acceptance_rate": offer_rate,
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        "linkedin_connected": linkedin,
    }


# ---------------------------------------------------------------------------
# 1. IDEAL  —  ML Engineer, Pune, 7 years, high engagement
# ---------------------------------------------------------------------------
IDEAL: dict = {
    "candidate_id": "CAND_9000001",
    "profile": {
        "anonymized_name": "Arjun Mehta",
        "headline": "Senior ML Engineer | Embeddings, Retrieval, Ranking, NLP",
        "summary": (
            "7 years building production ML systems at product companies. "
            "Deep expertise in embedding-based retrieval, hybrid search, "
            "vector databases, and LLM-based re-ranking pipelines. "
            "Shipped end-to-end ranking systems used by millions. "
            "Strong Python and PyTorch. Active open-source contributor."
        ),
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "Senior ML Engineer",
        "current_company": "Pied Piper",
        "current_company_size": "51-200",
        "current_industry": "AI/ML",
    },
    "career_history": [
        {
            "company": "Pied Piper",
            "title": "Senior ML Engineer",
            "start_date": "2024-07-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "51-200",
            "description": (
                "Built a production embedding-based retrieval system using "
                "FAISS and sentence-transformers serving 5M queries/day. "
                "Designed the hybrid search (BM25 + dense) re-ranking pipeline "
                "using a cross-encoder. Set up offline NDCG benchmarks and online "
                "A/B testing framework. Fine-tuned LoRA adapters for domain "
                "adaptation. Deployed inference endpoints on GCP at p99 < 50ms."
            ),
        },
        {
            "company": "CRED",
            "title": "ML Engineer",
            "start_date": "2022-06-01",
            "end_date": "2024-06-30",
            "duration_months": 25,
            "is_current": False,
            "industry": "Fintech",
            "company_size": "1001-5000",
            "description": (
                "Owned the recommendation system for credit-card offers. "
                "Built and deployed a two-tower embedding model for candidate "
                "retrieval, followed by a LightGBM learning-to-rank model. "
                "Used Milvus as the vector store. Implemented A/B testing to "
                "measure recommendation quality (NDCG, MAP). Ran inference "
                "pipeline serving real-time ranking results."
            ),
        },
        {
            "company": "Razorpay",
            "title": "Data Scientist",
            "start_date": "2019-06-01",
            "end_date": "2022-05-31",
            "duration_months": 36,
            "is_current": False,
            "industry": "Fintech",
            "company_size": "1001-5000",
            "description": (
                "Built NLP-based transaction categorisation using transformer "
                "models (BERT fine-tuning). Developed search retrieval pipeline "
                "for merchant discovery. Deployed ML models to production using "
                "FastAPI and Docker. Established evaluation framework for ranking "
                "quality including offline MRR and online click-through metrics."
            ),
        },
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2014,
            "end_year": 2018,
            "grade": "8.9 CGPA",
            "tier": "tier_1",
        }
    ],
    "skills": [
        {"name": "NLP",                        "proficiency": "advanced",      "endorsements": 30, "duration_months": 48},
        {"name": "Python",                     "proficiency": "advanced",      "endorsements": 25, "duration_months": 84},
        {"name": "PyTorch",                    "proficiency": "advanced",      "endorsements": 20, "duration_months": 48},
        {"name": "Embeddings",                 "proficiency": "advanced",      "endorsements": 15, "duration_months": 36},
        {"name": "Milvus",                     "proficiency": "intermediate",  "endorsements": 10, "duration_months": 25},
        {"name": "FAISS",                      "proficiency": "advanced",      "endorsements": 12, "duration_months": 24},
        {"name": "Fine-tuning LLMs",           "proficiency": "advanced",      "endorsements": 18, "duration_months": 24},
        {"name": "Information Retrieval",      "proficiency": "advanced",      "endorsements": 14, "duration_months": 48},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": _signals(
        last_active="2026-06-21",
        open_to_work=True,
        response_rate=0.90,
        interview_rate=0.90,
        offer_rate=0.80,
        completeness=95.0,
        notice=30,
        willing_relocate=True,
        github=75.0,
        linkedin=True,
    ),
}


# ---------------------------------------------------------------------------
# 2. KEYWORD_STUFFER  —  HR Manager with AI skill keywords, all duration=0
# ---------------------------------------------------------------------------
KEYWORD_STUFFER: dict = {
    "candidate_id": "CAND_9000002",
    "profile": {
        "anonymized_name": "Priya Sharma",
        "headline": "HR Manager | AI Tools Enthusiast | Talent Acquisition",
        "summary": (
            "Experienced HR professional with a growing interest in AI tools "
            "for workflow automation. Primary expertise is in talent acquisition, "
            "performance management, and employee relations. Self-taught in AI "
            "tools like ChatGPT for productivity enhancement."
        ),
        "location": "Hyderabad, Telangana",
        "country": "India",
        "years_of_experience": 5.0,
        "current_title": "HR Manager",
        "current_company": "Stark Industries",
        "current_company_size": "1001-5000",
        "current_industry": "Manufacturing",
    },
    "career_history": [
        {
            "company": "Stark Industries",
            "title": "HR Manager",
            "start_date": "2024-07-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": "Manufacturing",
            "company_size": "1001-5000",
            "description": (
                "Managing talent acquisition pipelines and employee experience "
                "programs. Overseeing performance review cycles and compensation "
                "benchmarking. Used neural network-based resume screening tools "
                "to shortlist candidates. Maintained HRIS and compliance records."
            ),
        },
        {
            "company": "Wayne Enterprises",
            "title": "HR Manager",
            "start_date": "2021-06-01",
            "end_date": "2024-06-30",
            "duration_months": 37,
            "is_current": False,
            "industry": "Conglomerate",
            "company_size": "10001+",
            "description": (
                "Led HR operations for a 400-person business unit. Managed "
                "onboarding, L&D programs, and exit interviews. Ran engagement "
                "surveys and interpreted results using basic analytics. Partnered "
                "with business leaders on headcount planning."
            ),
        },
    ],
    "education": [
        {
            "institution": "Symbiosis Institute of Management",
            "degree": "MBA",
            "field_of_study": "Human Resources",
            "start_year": 2018,
            "end_year": 2020,
            "grade": "78%",
            "tier": "tier_2",
        }
    ],
    "skills": [
        # All JD-relevant skills listed with proficiency=expert, duration_months=0
        # → skill_trust_score = 0 (duration_factor = 0)
        {"name": "Python",   "proficiency": "expert", "endorsements": 5, "duration_months": 0},
        {"name": "PyTorch",  "proficiency": "expert", "endorsements": 3, "duration_months": 0},
        {"name": "FAISS",    "proficiency": "expert", "endorsements": 2, "duration_months": 0},
        {"name": "NLP",      "proficiency": "expert", "endorsements": 4, "duration_months": 0},
        {"name": "LangChain","proficiency": "expert", "endorsements": 6, "duration_months": 0},
        {"name": "Talent Acquisition", "proficiency": "expert", "endorsements": 20, "duration_months": 60},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": _signals(
        last_active="2026-06-20",
        open_to_work=True,
        response_rate=0.70,
        interview_rate=0.70,
        completeness=78.0,
        notice=45,
        willing_relocate=False,
        github=-1,
    ),
}


# ---------------------------------------------------------------------------
# 3. HONEYPOT  —  impossible profile: 10y claimed, 2-year career, expert skills with 0 months
# ---------------------------------------------------------------------------
HONEYPOT: dict = {
    "candidate_id": "CAND_9000003",
    "profile": {
        "anonymized_name": "Fake Expert",
        "headline": "Expert AI Researcher | 10 years | Deep Learning | LLMs",
        "summary": (
            "10 years of expert-level experience in machine learning, "
            "deep learning, NLP, and large language models. "
            "Published 50+ papers. Expert in every framework."
        ),
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 10.0,  # claims 10 years
        "current_title": "Principal AI Researcher",
        "current_company": "Hooli",
        "current_company_size": "1001-5000",
        "current_industry": "AI/ML",
    },
    "career_history": [
        {
            # Only 24 months of career → |24 - 10*12| = |24-120| = 96 > 18 → flag_b
            "company": "Hooli",
            "title": "Principal AI Researcher",
            "start_date": "2024-07-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "1001-5000",
            "description": "Advanced AI research and development.",
        },
    ],
    "education": [
        {
            "institution": "IIT Delhi",
            "degree": "Ph.D",
            "field_of_study": "Machine Learning",
            "start_year": 2014,
            "end_year": 2020,
            "grade": "9.5 CGPA",
            "tier": "tier_1",
        }
    ],
    "skills": [
        # Expert with duration_months=0 → flag_a
        {"name": "Python",     "proficiency": "expert", "endorsements": 50, "duration_months": 0},
        {"name": "PyTorch",    "proficiency": "expert", "endorsements": 50, "duration_months": 0},
        {"name": "NLP",        "proficiency": "expert", "endorsements": 50, "duration_months": 0},
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 50, "duration_months": 0},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "native"}],
    "redrob_signals": _signals(
        last_active="2026-06-21",
        open_to_work=True,
        response_rate=0.95,
        interview_rate=0.95,
        offer_rate=0.95,
        completeness=99.0,
        notice=0,
        github=99.0,
    ),
}


# ---------------------------------------------------------------------------
# 4. TIER5_FIT  —  Data Engineer, no AI in title/skills, but ML in descriptions
# ---------------------------------------------------------------------------
TIER5_FIT: dict = {
    "candidate_id": "CAND_9000004",
    "profile": {
        "anonymized_name": "Divya Krishnan",
        "headline": "Data Engineer | Pipelines, ETL, Platform",
        "summary": (
            "Data Engineer with 5 years building large-scale data pipelines and "
            "platform infrastructure. Background in Spark, Airflow, and SQL "
            "warehouses. Have shipped platform components that ML teams depend on "
            "for feature serving and retrieval."
        ),
        "location": "Chennai, Tamil Nadu",
        "country": "India",
        "years_of_experience": 5.0,
        "current_title": "Data Engineer",
        "current_company": "Flipkart",
        "current_company_size": "5001-10000",
        "current_industry": "E-commerce",
    },
    "career_history": [
        {
            "company": "Flipkart",
            "title": "Data Engineer",
            "start_date": "2023-06-01",
            "end_date": None,
            "duration_months": 37,
            "is_current": True,
            "industry": "E-commerce",
            "company_size": "5001-10000",
            "description": (
                "Built and deployed the recommendation system retrieval layer "
                "that powers product search for 200M users. Designed the ranking "
                "pipeline that merges candidate sets from multiple retrieval "
                "sources. Implemented the vector embedding index refresh and "
                "production monitoring. Ran A/B tests to validate retrieval "
                "quality improvements using NDCG and MRR metrics."
            ),
        },
        {
            "company": "Swiggy",
            "title": "Data Engineer",
            "start_date": "2021-06-01",
            "end_date": "2023-05-31",
            "duration_months": 24,
            "is_current": False,
            "industry": "Food Delivery",
            "company_size": "1001-5000",
            "description": (
                "Owned ETL pipelines ingesting 50GB/day of order and restaurant "
                "data into Redshift. Built Spark jobs for feature engineering "
                "used by the ML team. Maintained Airflow DAGs and monitored "
                "pipeline SLAs."
            ),
        },
    ],
    "education": [
        {
            "institution": "NIT Trichy",
            "degree": "B.Tech",
            "field_of_study": "Information Technology",
            "start_year": 2016,
            "end_year": 2020,
            "grade": "8.1 CGPA",
            "tier": "tier_2",
        }
    ],
    "skills": [
        # Python IS in JD_REQUIRED_SKILLS → contributes to skill_trust
        {"name": "Python",  "proficiency": "intermediate", "endorsements": 18, "duration_months": 48},
        {"name": "Spark",   "proficiency": "advanced",     "endorsements": 12, "duration_months": 36},
        {"name": "Airflow", "proficiency": "intermediate", "endorsements": 8,  "duration_months": 30},
        {"name": "SQL",     "proficiency": "advanced",     "endorsements": 15, "duration_months": 48},
        {"name": "Redshift","proficiency": "intermediate", "endorsements": 6,  "duration_months": 24},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": _signals(
        last_active="2026-06-20",
        open_to_work=True,
        response_rate=0.75,
        interview_rate=0.75,
        completeness=85.0,
        notice=30,
        willing_relocate=True,
        github=30.0,
    ),
}


# ---------------------------------------------------------------------------
# 5. TITLE_CHASER  —  3 jobs ~10 months avg, Senior→Staff→Principal, low engagement
# ---------------------------------------------------------------------------
TITLE_CHASER: dict = {
    "candidate_id": "CAND_9000005",
    "profile": {
        "anonymized_name": "Rohan Kapoor",
        "headline": "Principal Sales Manager | Growth | Revenue",
        "summary": (
            "Seasoned sales and business-development leader with a track record "
            "of building outbound pipelines and closing enterprise deals. "
            "Progressive career from individual contributor to Principal. "
            "Looking for the next challenge."
        ),
        "location": "Bangalore, Karnataka",
        "country": "India",
        "years_of_experience": 2.5,
        "current_title": "Principal Sales Manager",
        "current_company": "Globex Inc",
        "current_company_size": "501-1000",
        "current_industry": "Software",
    },
    "career_history": [
        {
            "company": "Globex Inc",
            "title": "Principal Sales Manager",
            "start_date": "2025-09-01",
            "end_date": None,
            "duration_months": 10,
            "is_current": True,
            "industry": "Software",
            "company_size": "501-1000",
            "description": (
                "Leading enterprise sales for SaaS platform. Managing key accounts "
                "and driving quarterly revenue targets. Presenting to C-suite "
                "stakeholders. Coaching junior sales team members."
            ),
        },
        {
            "company": "Initech",
            "title": "Staff Sales Manager",
            "start_date": "2024-11-01",
            "end_date": "2025-08-31",
            "duration_months": 10,
            "is_current": False,
            "industry": "Software",
            "company_size": "201-500",
            "description": (
                "Owned mid-market sales pipeline for analytics product. "
                "Consistently hit 110% of quota. Collaborated with product "
                "team on roadmap based on customer feedback."
            ),
        },
        {
            "company": "Acme Corp",
            "title": "Senior Sales Manager",
            "start_date": "2024-01-01",
            "end_date": "2024-10-31",
            "duration_months": 10,
            "is_current": False,
            "industry": "Software",
            "company_size": "51-200",
            "description": (
                "Generated outbound pipeline through cold outreach and networking. "
                "Ran product demos and negotiated contracts. Managed CRM hygiene "
                "in Salesforce. Highest new-logo closer on the team."
            ),
        },
    ],
    "education": [
        {
            "institution": "Delhi University",
            "degree": "B.Com",
            "field_of_study": "Commerce",
            "start_year": 2017,
            "end_year": 2020,
            "grade": "72%",
            "tier": "tier_3",
        }
    ],
    "skills": [
        {"name": "Salesforce",       "proficiency": "advanced",     "endorsements": 15, "duration_months": 24},
        {"name": "Account Management","proficiency": "advanced",    "endorsements": 18, "duration_months": 30},
        {"name": "CRM",              "proficiency": "intermediate", "endorsements": 10, "duration_months": 24},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": _signals(
        # NOT open to work, last active 150 days ago, low response / interview rates
        last_active="2026-01-23",
        open_to_work=False,
        response_rate=0.20,
        interview_rate=0.30,
        completeness=60.0,
        notice=60,
        willing_relocate=False,
        github=-1,
        verified_email=False,
        verified_phone=False,
        linkedin=False,
    ),
}


# ---------------------------------------------------------------------------
# 6. CONSULTING_ONLY  —  4 jobs all at large IT services firms
# ---------------------------------------------------------------------------
CONSULTING_ONLY: dict = {
    "candidate_id": "CAND_9000006",
    "profile": {
        "anonymized_name": "Vikram Singh",
        "headline": "Senior Data Scientist | TCS | Machine Learning | Analytics",
        "summary": (
            "8 years across major IT services firms delivering machine learning "
            "and analytics projects for enterprise clients. Solid in Python, "
            "SQL, and classical ML. Looking to transition to a product company."
        ),
        "location": "Noida, Uttar Pradesh",
        "country": "India",
        "years_of_experience": 8.0,
        "current_title": "Senior Data Scientist",
        "current_company": "TCS",
        "current_company_size": "10001+",
        "current_industry": "IT Services",
    },
    "career_history": [
        {
            "company": "TCS",
            "title": "Senior Data Scientist",
            "start_date": "2023-01-01",
            "end_date": None,
            "duration_months": 42,
            "is_current": True,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Delivered ML solutions for banking client — churn prediction "
                "and credit scoring models using Python and scikit-learn. "
                "Built and deployed batch inference pipelines on Azure ML. "
                "Collaborated with client data teams on feature engineering."
            ),
        },
        {
            "company": "Infosys",
            "title": "Data Scientist",
            "start_date": "2020-06-01",
            "end_date": "2022-12-31",
            "duration_months": 31,
            "is_current": False,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Built NLP classifiers for insurance document processing. "
                "Used BERT-based models for entity extraction. Developed "
                "recommendation logic for retail client product catalogue."
            ),
        },
        {
            "company": "Wipro",
            "title": "Data Analyst",
            "start_date": "2018-06-01",
            "end_date": "2020-05-31",
            "duration_months": 24,
            "is_current": False,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Statistical analysis and reporting for telecom client. "
                "Built dashboards in Tableau. SQL queries for data extraction. "
                "Basic machine learning models for churn analysis in Python."
            ),
        },
        {
            "company": "Accenture",
            "title": "Analyst",
            "start_date": "2016-07-01",
            "end_date": "2018-05-31",
            "duration_months": 23,
            "is_current": False,
            "industry": "IT Services",
            "company_size": "10001+",
            "description": (
                "Business analysis and requirements gathering for ERP "
                "implementations. Data migration and testing. Stakeholder "
                "workshops and documentation."
            ),
        },
    ],
    "education": [
        {
            "institution": "Delhi Technological University",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2012,
            "end_year": 2016,
            "grade": "7.8 CGPA",
            "tier": "tier_2",
        }
    ],
    "skills": [
        {"name": "Python",       "proficiency": "advanced",     "endorsements": 22, "duration_months": 48},
        {"name": "scikit-learn", "proficiency": "advanced",     "endorsements": 14, "duration_months": 36},
        {"name": "NLP",          "proficiency": "intermediate", "endorsements": 10, "duration_months": 24},
        {"name": "SQL",          "proficiency": "advanced",     "endorsements": 18, "duration_months": 72},
        {"name": "Tableau",      "proficiency": "intermediate", "endorsements": 8,  "duration_months": 24},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": _signals(
        last_active="2026-06-15",
        open_to_work=True,
        response_rate=0.65,
        interview_rate=0.70,
        completeness=82.0,
        notice=60,
        willing_relocate=True,
        github=20.0,
    ),
}


# ---------------------------------------------------------------------------
# Exported list for easy iteration
# ---------------------------------------------------------------------------
ALL_FIXTURES: list[dict] = [
    IDEAL,
    KEYWORD_STUFFER,
    HONEYPOT,
    TIER5_FIT,
    TITLE_CHASER,
    CONSULTING_ONLY,
]
