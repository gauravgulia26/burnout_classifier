"""Streamlit Community Cloud entry point.

Streamlit Cloud hosts the UI process only. The FastAPI service and its MLflow
model remain a separately deployed service, configured through Streamlit
Secrets (or environment variables for local testing).
"""

import os
from typing import Any


def _read_secret(secrets: Any, key: str) -> str | None:
    """Read a secret without leaking its value or requiring it to exist."""
    try:
        value = secrets.get(key)
    except (AttributeError, KeyError):
        return None
    return str(value).strip() if value else None


def _normalize_backend_url(url: str) -> str:
    """Convert a host or endpoint URL into the versioned API base URL."""
    normalized_url = url.rstrip("/")
    for suffix in ("/api/v1/health", "/api/v1/predict", "/health", "/predict"):
        if normalized_url.endswith(suffix):
            normalized_url = normalized_url[: -len(suffix)]
            break
    if not normalized_url.endswith("/api/v1"):
        normalized_url = f"{normalized_url}/api/v1"
    return normalized_url


def _configure_backend_url() -> str | None:
    """Prefer Streamlit Cloud secrets, then support normal environment variables."""
    import streamlit as st

    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    backend_url = _read_secret(secrets, "BACKEND_API_URL") or os.getenv("BACKEND_API_URL")
    if backend_url:
        backend_url = _normalize_backend_url(backend_url)
        os.environ["BACKEND_API_URL"] = backend_url
    return backend_url


def main() -> None:
    """Launch the existing frontend with Cloud-specific configuration handling."""
    import streamlit as st

    backend_url = _configure_backend_url()
    if not backend_url:
        st.set_page_config(page_title="Burnout Compass", page_icon="◌")
        st.error("The frontend is not configured yet.")
        st.info(
            "Add BACKEND_API_URL to Streamlit Cloud → App settings → Secrets. "
            "It should end with /api/v1."
        )
        st.stop()

    from src.frontend.app import main as frontend_main

    frontend_main()


if __name__ == "__main__":
    main()
