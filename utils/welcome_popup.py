# utils/welcome_popup.py
# ============================================================
# Welcome popup shown once per session on first load.
# Uses st.session_state to avoid showing it again
# during the same browser session.
# ============================================================

import streamlit as st


@st.dialog("👋 Welcome to HERMES")
def _show_welcome_dialog():
    st.markdown(
        """
        **HERMES** — *Historical Ecology Retrieval & Multi-LLM Evaluation System*

        HERMES lets you query and evaluate marine knowledge graphs generated
        from a 19th-century marine biology corpus using different Large Language
        Models. Each graph was constructed with a different LLM, allowing direct
        comparison of how model choice affects knowledge structure and grounding
        fidelity.

        ---

        🎬 **First time here?**
        We recommend watching the demo video to get familiar with the system.
        Press the **"Watch demo"** button in the sidebar to get started.
        """
    )
    st.checkbox(
        "Don't show this again for this session",
        key="welcome_dismissed",
    )


def show_welcome_popup():
    """
    Call this once at the top of app.py.
    Shows the welcome dialog only on the first load of each session.
    """
    if "welcome_shown" not in st.session_state:
        st.session_state["welcome_shown"] = False

    if not st.session_state["welcome_shown"]:
        _show_welcome_dialog()
        st.session_state["welcome_shown"] = True