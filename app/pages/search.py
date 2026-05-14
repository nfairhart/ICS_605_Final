"""
search.py — Page 2: Find Jobs

Semantic search over the ChromaDB job_postings collection.
Results can be sent directly to the Analyze page.
"""

import streamlit as st

from chroma_client import search_jobs
from ui_components import render_job_card


def show():
    st.title("Find Jobs")
    st.caption(
        "Describe the kind of role you're looking for and we'll find the best "
        "semantic matches from our job database."
    )

    with st.form("search_form"):
        query = st.text_input(
            "Describe the role you want",
            placeholder="e.g. machine learning engineer San Francisco fintech",
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            n_results = st.slider("Results", min_value=5, max_value=20, value=10)
        submitted = st.form_submit_button("Search", type="primary")

    if submitted and query.strip():
        with st.spinner("Searching..."):
            try:
                jobs = search_jobs(query.strip(), n=n_results)
                st.session_state["_search_results"] = jobs
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.pop("_search_results", None)
                return

        if not jobs:
            st.info("No results found. Try a different query.")
            st.session_state.pop("_search_results", None)
            return

    elif submitted:
        st.warning("Please enter a search query.")

    jobs = st.session_state.get("_search_results", [])
    if jobs:
        st.success(f"Found {len(jobs)} matching jobs")

        def send_to_analyze(job: dict) -> None:
            st.session_state["prefill_job_text"] = job["text"]
            st.session_state["prefill_job_title"] = job.get("title", "")
            st.switch_page("pages/analyze.py")

        for job in jobs:
            render_job_card(job, on_analyze=send_to_analyze)


show()
