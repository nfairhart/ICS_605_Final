"""
ui_components.py

Reusable Streamlit rendering helpers for MatchScore results and job cards.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from create_dataset.matching import MatchScore


_FIT_COLOR = {
    "well-matched": "green",
    "under-qualified": "orange",
    "over-qualified": "blue",
}
_FIT_ICON = {
    "well-matched": "✓",
    "under-qualified": "↑",
    "over-qualified": "↓",
}
_FIT_LABEL = {
    "well-matched": "Well Matched",
    "under-qualified": "Under-Qualified",
    "over-qualified": "Over-Qualified",
}


def render_match_score(score: MatchScore, job_title: str = "") -> None:
    """Render a full MatchScore report showing all 8 criteria."""

    header = f"Match Report — {job_title}" if job_title else "Match Report"
    st.subheader(header)

    # ── Criteria 1 & 2: Score + Experience Fit ───────────────────────────────
    col_score, col_fit = st.columns([1, 1])

    with col_score:
        with st.container(border=True):
            st.markdown("**① Overall Match Score**")
            pct = score.match_score
            color = "green" if pct >= 70 else ("orange" if pct >= 45 else "red")
            st.markdown(
                f"<div style='font-size:3rem;font-weight:700;color:{color};line-height:1'>"
                f"{pct}<span style='font-size:1.2rem;color:gray'> / 100</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(pct / 100)

    with col_fit:
        with st.container(border=True):
            st.markdown("**② Experience Level Fit**")
            fit = score.experience_level_fit
            fit_color = _FIT_COLOR.get(fit, "gray")
            fit_icon = _FIT_ICON.get(fit, "•")
            fit_label = _FIT_LABEL.get(fit, fit.replace("-", " ").title())
            st.markdown(
                f"<div style='font-size:1.6rem;font-weight:600;margin-top:0.5rem'>"
                f"<span style='color:{fit_color}'>{fit_icon} {fit_label}</span></div>",
                unsafe_allow_html=True,
            )

    # ── Criterion 3: Rationale ────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**③ Overall Rationale**")
        st.write(score.rationale)

    # ── Criteria 4 & 5: Strengths + Gaps ─────────────────────────────────────
    col_str, col_gap = st.columns(2)

    with col_str:
        with st.container(border=True):
            st.markdown("**④ Matching Strengths**")
            for item in score.matching_strengths:
                st.markdown(f":green[✓]&nbsp; {item}")

    with col_gap:
        with st.container(border=True):
            st.markdown("**⑤ Skill Gaps**")
            for item in score.skill_gaps:
                st.markdown(f":orange[△]&nbsp; {item}")

    # ── Criterion 6: ATS Keywords Missing ────────────────────────────────────
    with st.container(border=True):
        st.markdown("**⑥ ATS Keywords Missing**")
        st.caption("Add these to your resume to pass automated screening filters.")
        if score.ats_keywords_missing:
            st.markdown("  ".join(f"`{kw}`" for kw in score.ats_keywords_missing))
        else:
            st.markdown("_None identified — resume already covers key terms._")

    # ── Criteria 7 & 8: Improvements + Activities ────────────────────────────
    col_imp, col_act = st.columns(2)

    with col_imp:
        with st.container(border=True):
            st.markdown("**⑦ Resume Improvements**")
            st.caption("Targeted edits to strengthen this application.")
            for i, tip in enumerate(score.resume_improvements, 1):
                st.markdown(f"**{i}.** {tip}")

    with col_act:
        with st.container(border=True):
            st.markdown("**⑧ Recommended Activities**")
            st.caption("Actions to close experience gaps before applying.")
            if score.recommended_activities:
                for i, act in enumerate(score.recommended_activities, 1):
                    st.markdown(f"**{i}.** {act}")
            else:
                st.markdown("_No additional activities needed — gaps are minor._")


def render_job_card(job: dict, on_analyze=None) -> None:
    """Render a job search result card."""
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            title = job.get("title") or "Untitled Role"
            company = job.get("company", "")
            location = job.get("location", "")
            exp = job.get("experience_level", "")
            st.markdown(f"**{title}**")
            parts = [p for p in [company, location, exp] if p]
            st.caption("  ·  ".join(parts))
        with col2:
            dist = job.get("distance", None)
            if dist is not None:
                relevance = max(0, round((1 - dist) * 100))
                st.metric("Relevance", f"{relevance}%")

        with st.expander("View Description"):
            text = job.get("text", "")
            st.write(text[:1500] + ("…" if len(text) > 1500 else ""))

        if on_analyze is not None:
            if st.button("Analyze with My Resume", key=f"analyze_{job['id']}"):
                on_analyze(job)
