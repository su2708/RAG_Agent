import streamlit as st
from datetime import datetime

# Youtube 검색 결과를 보여주는 함수 
def display_videos(results):
    for result in results:
        # 구획 나누기 
        col1, col2 = st.columns(2)
        
        # 날짜 계산 
        upload_date = datetime.strptime(
            result['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'
        )
        days_since_upload = (datetime.now() - upload_date).days
        
        st.markdown('---')
        
        # 왼쪽 구획 
        with col1:
            video_url = f"https://www.youtube.com/watch?v={result['video_id']}"
            st.video(video_url)
        
        # 오른쪽 구획 
        with col2:
            # 제목 
            st.markdown(f"##### {result['title']}")
            
            # 업로드 날짜와 채널 이름 
            st.markdown(
                f"""
                <div>
                    <p style="color: #cccccc;">
                        createdAt: {days_since_upload} days ago
                    </p>
                    <p style="color: #cccccc;">
                        Channel: {result['channelTitle']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 영상 설명 
            st.markdown(f"{result['description'][:100]}...")

# News 검색 결과를 보여주는 함수 
def display_news(results):
    columns = st.columns(len(results))
    for i, col in enumerate(columns):
        with col:
            st.markdown(f"##### {results[i]["metadata"]["title"]}")
            st.markdown(
                f"""
                <div>
                    <p style="color: #cccccc;">
                        날짜: {results[i]["metadata"]["date"]}
                    </p>
                    <p style="color: #cccccc;">
                        <a href="{results[i]["metadata"]["url"]}">
                            URL: {results[i]["metadata"]["url"]}
                        </a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")
            st.markdown(f"{results[i]["content"][:200]}...")