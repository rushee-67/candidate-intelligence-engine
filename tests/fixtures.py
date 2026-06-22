"""
Test Fixtures Module.

Provides pre-defined mock candidate profile payloads to test individual scoring rules, 
filters, and pipeline behavior, including:
- Ideal candidates: 6-8 years experience in applied ML/AI at a product company, with Pune/Noida location and high active engagement.
- Disqualified candidates: Pure research background, LangChain-only experience, no code written recently, and IT consulting-only career histories.
- Honeypot candidates: Mismatches in years of experience, expert proficiency flags with zero duration.
- Edge cases: Tie-break candidates with identical profiles or scores.
"""

# Placeholders for future test fixtures
def get_mock_ideal_candidate():
    """
    Returns an ideal candidate dictionary.
    """
    pass

def get_mock_disqualified_candidate():
    """
    Returns a candidate dictionary that should be filtered out.
    """
    pass

def get_mock_honeypot_candidate():
    """
    Returns an anomalous profile representing a honeypot.
    """
    pass
