"""Streamlit application for individual burnout-risk assessments."""

import time
from typing import Any

import streamlit as st

from src.frontend.core.config import FrontendSettings
from src.frontend.services.api_client import BackendAPIClient, BackendAPIError

st.set_page_config(
    page_title="Burnout Compass",
    page_icon="◌",
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
          .stApp { background: #f6f8fc; color: #172033; }
          .block-container { max-width: 1160px; padding-top: 2.5rem; padding-bottom: 3rem; }
          .hero { background: linear-gradient(125deg, #14254a 0%, #285ea8 55%, #54b8b3 110%);
                  border-radius: 24px; padding: 2.4rem 2.5rem; color: #fff; margin-bottom: 1.6rem; }
          .hero h1 { font-size: 2.4rem; margin: 0 0 .35rem 0; letter-spacing: -.04em; }
          .hero p { font-size: 1.05rem; max-width: 42rem; margin: 0; opacity: .88; }
          .section-label { color: #5073a8; text-transform: uppercase; letter-spacing: .1em;
                           font-size: .73rem; font-weight: 700; margin-bottom: .4rem; }
          .result-card { border-radius: 18px; padding: 1.4rem 1.6rem; margin-top: 1rem;
                         background: #fff; border: 1px solid #e2e8f4; box-shadow: 0 8px 22px #243a6810; }
          .risk-low { border-left: 6px solid #2da77c; }
          .risk-medium { border-left: 6px solid #e6a23c; }
          .risk-high { border-left: 6px solid #e15d66; }
          .risk-title { font-size: 1.55rem; font-weight: 750; margin: .15rem 0; }
          div[data-testid="stForm"] { background: #fff; padding: 1.3rem 1.5rem .5rem;
            border: 1px solid #e2e8f4; border-radius: 18px; box-shadow: 0 8px 22px #243a680c; }
          .stButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 650;
            background: #245ea8; border-color: #245ea8; padding: .55rem 1rem; }
          .stButton > button:hover, .stFormSubmitButton > button:hover { background: #173f76; border-color: #173f76; }
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
        profile_left, profile_right = st.columns(2)
        with profile_left:
            major = st.selectbox("Academic discipline", choices["major"])
            year = st.selectbox("Year of study", choices["year"], index=2)
            pre_gpa = st.number_input("Pre-semester GPA", 0.0, 4.0, 3.2, 0.01)
            post_gpa = st.number_input("Post-semester GPA", 0.0, 4.0, 3.2, 0.01)
        with profile_right:
            use_case = st.selectbox("Primary AI use", choices["use_case"])
            skill = st.selectbox("Prompt engineering skill", choices["skill"], index=1)
            policy = st.selectbox("Institutional AI policy", choices["policy"], index=1)
            paid = st.toggle("Uses a paid AI subscription")

        st.markdown("<p class='section-label' style='margin-top:1.2rem'>Study habits & wellbeing</p>", unsafe_allow_html=True)
        habits_left, habits_center, habits_right = st.columns(3)
        with habits_left:
            ai_hours = st.number_input("Weekly GenAI hours", 0.0, 168.0, 6.0, 0.25)
            study_hours = st.number_input("Traditional study hours", 0.0, 168.0, 12.0, 0.25)
        with habits_center:
            tool_diversity = st.slider("AI tool diversity", 1, 10, 3)
            ai_dependency = st.slider("Perceived AI dependency", 1, 10, 4)
        with habits_right:
            anxiety = st.slider("Exam anxiety level", 1, 10, 5)
            retention = st.number_input("Skill retention score", 0.0, 100.0, 75.0, 0.5)

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
        """<section class="hero"><h1>Burnout Compass</h1>
        <p>A focused check-in for understanding student burnout-risk signals, powered by the registered prediction model.</p>
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
