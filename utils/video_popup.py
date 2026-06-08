import streamlit as st

@st.dialog('HERMES Demo', width="stretch")
def show_video():
    video_file = open("media/soon.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes, autoplay=True)