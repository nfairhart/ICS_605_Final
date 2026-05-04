"""
history.py — Page 3: Analysis History

Browse all analyses from the current session and export to CSV.
"""

import io
import csv
import streamlit as st

from ui_components import render_match_score


def show():
    st.title("Analysis History")

    analyses = st.session_state.get("analyses", [])

    if not analyses:
        st.info("No analyses yet. Go to **Analyze Resume** to get started.")
        return

    st.caption(f"{len(analyses)} analysis/analyses this session")

    # Summary table
    rows = [
        {
            "Time": a["timestamp"],
            "Job Title": a["job_title"],
            "Score": a["match_score"],
            "Fit": a["experience_level_fit"],
        }
        for a in analyses
    ]
    st.dataframe(rows, use_container_width=True)

    # CSV export
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["Time", "Job Title", "Score", "Fit"])
    writer.writeheader()
    writer.writerows(rows)
    st.download_button(
        "Export CSV",
        data=buf.getvalue(),
        file_name="job_analyses.csv",
        mime="text/csv",
    )

    st.divider()

    # Expandable detail for each analysis
    st.subheader("Full Results")
    for i, a in enumerate(reversed(analyses), 1):
        with st.expander(f"{a['timestamp']} — {a['job_title']} (score: {a['match_score']})"):
            render_match_score(a["result"], job_title=a["job_title"])


show()
