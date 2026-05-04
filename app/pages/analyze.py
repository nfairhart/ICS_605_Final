"""
analyze.py — Page 1: Resume Analyzer

Upload a PDF or paste text, paste a job description, and get a full
MatchScore analysis. Results persist across reruns via session state.
"""

import streamlit as st
from datetime import datetime

from model_client import score_resume_job
from pdf_parser import extract_text
from ui_components import render_match_score


def show():
    st.title("Resume Analyzer")
    st.caption("Paste your resume and a job description to get a detailed 8-criteria match report.")

    # Pre-fill from the job search page
    prefill_job = st.session_state.pop("prefill_job_text", None)
    prefill_title = st.session_state.pop("prefill_job_title", "")
    if prefill_job:
        st.session_state["_jd_area"] = prefill_job
    if prefill_title:
        st.session_state["_job_title_input"] = prefill_title

    # ── Resume input ──────────────────────────────────────────────────────────
    st.subheader("Your Resume")
    tab_pdf, tab_paste = st.tabs(["Upload PDF", "Paste Text"])

    with tab_pdf:
        uploaded = st.file_uploader("Resume PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            file_id = f"{uploaded.name}::{uploaded.size}"
            if st.session_state.get("_pdf_id") != file_id:
                with st.spinner("Extracting text from PDF…"):
                    try:
                        extracted = extract_text(uploaded.read())
                        st.session_state["_resume_pdf"] = extracted
                        st.session_state["_pdf_id"] = file_id
                    except Exception as e:
                        st.error(f"PDF extraction failed: {e}")
            if "_resume_pdf" in st.session_state:
                rt = st.session_state["_resume_pdf"]
                st.success(f"Extracted {len(rt):,} characters from **{uploaded.name}**")
                with st.expander("Preview extracted text"):
                    st.text(rt[:2000] + ("…" if len(rt) > 2000 else ""))

    with tab_paste:
        st.text_area(
            "Paste your full resume here",
            height=260,
            placeholder="Copy and paste your complete resume text…",
            key="_resume_paste",
        )

    # Resolve which resume text to use: paste wins if non-empty, else fall back to PDF
    paste_val = st.session_state.get("_resume_paste", "").strip()
    pdf_val = st.session_state.get("_resume_pdf", "")
    resume_text = paste_val if paste_val else pdf_val

    if resume_text:
        src = "paste" if paste_val else "PDF"
        chars = len(resume_text)
        st.caption(f"Resume source: {src} — {chars:,} characters")

    # ── Job Description input ─────────────────────────────────────────────────
    st.subheader("Job Description")
    if prefill_job and prefill_title:
        st.caption(f"Pre-filled from search: **{prefill_title}**")

    st.text_area(
        "Paste the full job posting",
        height=260,
        placeholder="Copy and paste the complete job description here…",
        key="_jd_area",
    )
    jd_text = st.session_state.get("_jd_area", "").strip()

    st.text_input(
        "Job title (optional — used as the report header)",
        placeholder="e.g. Senior Data Scientist at Acme Corp",
        key="_job_title_input",
    )
    job_title = st.session_state.get("_job_title_input", "").strip()

    # ── Action buttons ────────────────────────────────────────────────────────
    st.divider()
    col_go, col_clear, _ = st.columns([1, 1, 5])
    with col_go:
        go = st.button("Analyze Match", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Report", use_container_width=True):
            st.session_state.pop("_result", None)
            st.session_state.pop("_result_title", None)
            st.rerun()

    if go:
        if not resume_text:
            st.error("Please upload a PDF or paste your resume text.")
        elif not jd_text:
            st.error("Please paste a job description.")
        else:
            with st.spinner("Sending to LM Studio for analysis…"):
                try:
                    result = score_resume_job(resume_text, jd_text)
                    label = job_title or jd_text[:60].replace("\n", " ")
                    st.session_state["_result"] = result
                    st.session_state["_result_title"] = label

                    if "analyses" not in st.session_state:
                        st.session_state["analyses"] = []
                    st.session_state["analyses"].append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "job_title": label,
                        "match_score": result.match_score,
                        "experience_level_fit": result.experience_level_fit,
                        "result": result,
                        "resume_snippet": resume_text[:200],
                        "job_snippet": jd_text[:200],
                    })
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.info("Make sure LM Studio is running with a model loaded at http://localhost:1234")

    # ── Results — persist across reruns via session state ─────────────────────
    result = st.session_state.get("_result")
    if result:
        st.divider()
        render_match_score(result, job_title=st.session_state.get("_result_title", ""))


show()
