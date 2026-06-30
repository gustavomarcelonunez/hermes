import streamlit as st

@st.dialog("📋 Changelog")
def show_changelog():
    """Reads CHANGELOG.md and displays it as a modal popup."""
    try:
        with open("CHANGELOG.md", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = "_No changelog available._"
 
    st.markdown(content)