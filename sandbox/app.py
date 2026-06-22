"""
Streamlit Sandbox Application.

This file implements the interactive web application used to:
- Load candidate files (JSONL) and the job description.
- Trigger the ranking pipeline and visualize the top candidates.
- Display score breakdowns (title, skill trust, experience, location, etc.) for debugging.
- Highlight flagged honeypot alerts or disqualification reasons.
- Render candidate details alongside their generated reasoning.
- Export ranking results to a CSV matching the hackathon submission format.
"""

import streamlit as st

def main():
    """
    Main Streamlit application layout and state logic.
    """
    st.title("Redrob Candidate Ranker Sandbox")

if __name__ == '__main__':
    main()
