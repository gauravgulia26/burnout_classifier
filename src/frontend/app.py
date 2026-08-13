"""Streamlit application for individual burnout-risk assessments."""

import time
from typing import Any

import streamlit as st

from src.frontend.core.config import FrontendSettings
from src.frontend.services.api_client import BackendAPIClient, BackendAPIError

st.set_page_config(
    page_title="Burnout Compass | Student Risk Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> BackendAPIClient:
    """Keep a reusable HTTP session for the duration of a Streamlit process."""
    return BackendAPIClient(base_url)


@st.cache_data(ttl=30, show_spinner=False)
def get_health(base_url: str) -> dict[str, Any]:
    """Avoid polling the backend repeatedly during Streamlit reruns."""
    return get_api_client(base_url).health()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #0b1120; color: #e7edf8; }
          .block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 3rem; }
          [data-testid="stHeader"] { background: transparent; }
          [data-testid="stSidebar"] { background: #111a2d; border-right: 1px solid #243451; }
          [data-testid="stSidebar"] * { color: #dbe7f8; }
          .hero { background: radial-gradient(circle at 90% 5%, #2c8f9a 0, transparent 33%), linear-gradient(135deg, #162747 0%, #182d55 58%, #143e52 100%);
                  border: 1px solid #31547d; border-radius: 26px; padding: 2.5rem 2.7rem; color: #f7fbff; margin-bottom: 1.7rem; box-shadow: 0 18px 45px #00000035; }
          .hero h1 { font-size: 2.65rem; margin: 0 0 .45rem 0; letter-spacing: -.045em; }
          .hero p { color: #c5d7ef; font-size: 1.06rem; line-height: 1.6; max-width: 48rem; margin: 0; }
          .hero .eyebrow { color: #6fe0d3; text-transform: uppercase; letter-spacing: .15em; font-size: .71rem; font-weight: 800; margin-bottom: .65rem; }
          .section-label { color: #79d7d0; text-transform: uppercase; letter-spacing: .1em;
                           font-size: .73rem; font-weight: 700; margin-bottom: .4rem; }
          .section-help { color: #8da2bf; font-size: .86rem; margin: -.15rem 0 1rem; }
          .result-card { border-radius: 18px; padding: 1.4rem 1.6rem; margin-top: 1rem;
                         background: #111c31; border: 1px solid #2b405f; box-shadow: 0 12px 28px #00000028; }
          .risk-low { border-left: 6px solid #2da77c; }
          .risk-medium { border-left: 6px solid #e6a23c; }
          .risk-high { border-left: 6px solid #e15d66; }
          .risk-title { font-size: 1.55rem; font-weight: 750; margin: .15rem 0; }
          div[data-testid="stForm"] { background: #111a2d; padding: 1.45rem 1.55rem .6rem; border: 1px solid #263b5c; border-radius: 20px; box-shadow: 0 12px 28px #00000025; }
          label, [data-testid="stWidgetLabel"] p { color: #d7e3f4 !important; }
          [data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background: #17243a; border-color: #385275; color: #f3f7fc; }
          .stButton > button, .stFormSubmitButton > button { border-radius: 11px; font-weight: 700; background: linear-gradient(135deg, #2faaa6, #3177c5); border: 0; padding: .62rem 1rem; }
          .stButton > button:hover, .stFormSubmitButton > button:hover { background: linear-gradient(135deg, #48c8be, #438ddd); }
          .stAlert { background: #15233a; border-color: #31537d; }
          .stCaption { color: #8da2bf; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def field_choices() -> dict[str, list[str]]:
    return {
        "major": ["STEM", "Business", "Humanities", "Medical", "Arts"],
        "year": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
        "use_case": [
            "Ideation",
            "Copywriting/Drafting",
            "Summarizing_Reading",
            "Debugging/Troubleshooting",
            "Direct_Answer_Generation",
        ],
        "skill": ["Beginner", "Intermediate", "Advanced"],
        "policy": ["Strict_Ban", "Allowed_With_Citation", "Actively_Encouraged"],
    }


def assessment_form() -> tuple[bool, dict[str, Any]]:
    choices = field_choices()
    with st.form("burnout_assessment", clear_on_submit=False):
        st.markdown("<p class='section-label'>Student profile</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-help'>Tell us about the student’s academic context and AI usage.</p>", unsafe_allow_html=True)
        profile_left, profile_right = st.columns(2)
        with profile_left:
            major = st.selectbox("Academic discipline / major", choices["major"])
            year = st.selectbox("Current year of study", choices["year"], index=2)
            pre_gpa = st.number_input("GPA before the semester (0–4)", 0.0, 4.0, 3.2, 0.01)
            post_gpa = st.number_input("GPA after the semester (0–4)", 0.0, 4.0, 3.2, 0.01)
        with profile_right:
            use_case = st.selectbox("Most common generative-AI use", choices["use_case"])
            skill = st.selectbox("Prompt-writing skill level", choices["skill"], index=1)
            policy = st.selectbox("University AI policy", choices["policy"], index=1)
            paid = st.toggle("Student uses a paid AI subscription")

        st.markdown("<p class='section-label' style='margin-top:1.2rem'>Study habits & wellbeing</p>", unsafe_allow_html=True)
        st.markdown("<p class='section-help'>Use a typical study week and the student’s current self-assessment.</p>", unsafe_allow_html=True)
        habits_left, habits_center, habits_right = st.columns(3)
        with habits_left:
            ai_hours = st.number_input("Generative-AI hours per week", 0.0, 168.0, 6.0, 0.25)
            study_hours = st.number_input("Non-AI study hours per week", 0.0, 168.0, 12.0, 0.25)
        with habits_center:
            tool_diversity = st.slider("Number of AI tools used regularly", 1, 10, 3)
            ai_dependency = st.slider("Perceived dependence on AI (1–10)", 1, 10, 4)
        with habits_right:
            anxiety = st.slider("Exam anxiety level (1–10)", 1, 10, 5)
            retention = st.number_input("Skill retention score (0–100)", 0.0, 100.0, 75.0, 0.5)

        submitted = st.form_submit_button("Assess burnout risk", use_container_width=True)

    return submitted, {
        "Major_Category": major,
        "Year_of_Study": year,
        "Pre_Semester_GPA": pre_gpa,
        "Weekly_GenAI_Hours": ai_hours,
        "Primary_Use_Case": use_case,
        "Prompt_Engineering_Skill": skill,
        "Tool_Diversity": tool_diversity,
        "Paid_Subscription": paid,
        "Traditional_Study_Hours": study_hours,
        "Perceived_AI_Dependency": ai_dependency,
        "Institutional_Policy": policy,
        "Anxiety_Level_During_Exams": anxiety,
        "Post_Semester_GPA": post_gpa,
        "Skill_Retention_Score": retention,
    }


def show_result(risk: str) -> None:
    explanations = {
        "Low": "The model detected a relatively lower burnout-risk pattern.",
        "Medium": "The model detected signals worth monitoring over the coming weeks.",
        "High": "The model detected a higher burnout-risk pattern. Consider seeking support early.",
    }
    risk_class = f"risk-{risk.lower()}"
    st.markdown(
        f"""<div class="result-card {risk_class}">
          <div class="section-label">Assessment result</div>
          <div class="risk-title">{risk} burnout risk</div>
          <div>{explanations[risk]}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("This is a model estimate, not a clinical diagnosis.")


def main() -> None:
    inject_styles()
    settings = FrontendSettings.from_environment()

    api_url = settings.backend_api_url.rstrip("/")
    try:
        health = get_health(api_url)
    except BackendAPIError:
        st.markdown(
            """<section class="hero"><h1>Burnout Compass</h1>
            <p>Waiting for the prediction service to load its registered model. This page will be ready automatically.</p>
            </section>""",
            unsafe_allow_html=True,
        )
        st.info("Waiting for API to load…")
        time.sleep(2)
        st.rerun()
        return

    st.markdown(
        """<section class="hero"><div class="eyebrow">Student wellbeing · ML-powered early signal</div>
        <h1>Burnout Compass</h1>
        <p>A private, focused check-in that turns academic performance, study habits, AI usage, and wellbeing signals into an easy-to-understand burnout-risk estimate.</p>
        </section>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("Backend API URL", value=api_url).rstrip("/")
        if st.button("Refresh connection status", use_container_width=True):
            get_health.clear()

    try:
        health = get_health(api_url)
        st.success(f"Prediction service connected · {health['model_uri']}")
    except BackendAPIError as exc:
        st.error(f"Prediction service became unavailable: {exc}")
        return

    submitted, record = assessment_form()
    if submitted:
        with st.spinner("Assessing risk pattern..."):
            try:
                response = get_api_client(api_url).predict(record)
                show_result(response["predictions"][0]["burnout_risk"])
            except BackendAPIError as exc:
                st.error(f"Could not complete the assessment: {exc}")


if __name__ == "__main__":
    main()
