"""
Disqualifier Filters Module.

This module applies hard exclusion criteria based on the Job Description rules:
- Pure research background (academic labs or research-only roles without production deployment).
- LangChain-only AI experience (under 12 months, without solid pre-LLM production experience).
- No production code written in the last 18 months.
- Exclusive employment history at IT consulting firms (e.g., TCS, Infosys, Wipro, Accenture) without product company experience.
- Specializations solely in computer vision, speech, or robotics (without NLP or Information Retrieval exposure).
- Logic checks to filter out flagged honeypots.
"""

# Placeholders for future functions
def check_disqualifiers(candidate_features):
    """
    Checks candidate features against all hard disqualification criteria.
    Returns a dictionary of boolean flags indicating if they are disqualified.
    """
    pass

def should_exclude(candidate_features):
    """
    Determines if a candidate should be completely excluded from ranking.
    """
    pass
