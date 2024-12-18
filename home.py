import streamlit as st

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.title("🏠Home")

st.markdown(
    """
    # Hello!
    
    Welcome to my RAG & Agent study!
    
    Here are the apps I made:
    
    - [x] [AINews_RAG](/AINews)
    - [x] [Youtube & News Agent](/Youtube_Agent)
    - [ ] [Whisper](/Whisper)
    """
)