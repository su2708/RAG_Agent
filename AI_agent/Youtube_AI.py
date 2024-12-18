import os
import googleapiclient.discovery
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_openai import OpenAIEmbeddings
from langchain.agents import Tool
from .NewsRAG import AINewsRAG, StreamHandler
from .display import display_news, display_videos

# 환경 변수 로드 
load_dotenv()

# 환경 변수에서 경로 가져오기
vector_store_path = "../ai_news_vectorstore"

# 임베딩 모델 초기화 
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL")
)

# 환경 변수에서 OpenAI API 키를 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("OPENAI_API_KEY 환경 변수를 설정해주세요.")

# 환경 변수에서 YouTube API 키를 불러오기
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
if not youtube_api_key:
    print("YOUTUBE_API_KEY 환경 변수를 설정해주세요.")

# RAG 시스템 초기화
rag = AINewsRAG(embedding_model)

try:
    # 기존 벡터 스토어 로드 시도
    rag.load_vector_store(vector_store_path)
    print("✅ 기존 벡터 스토어를 로드했습니다.")
    
except Exception as e:
    print(f"벡터 스토어 로드 실패: {str(e)}")

# tool 설정: 뉴스 검색 
@tool
def search_news(query: str, k: int = 3):
    """
    AI 뉴스를 검색합니다.
    """
    search_results = []
    
    while True:
        query = query.strip()

        if not query:
            continue
            
        if query.lower() in ['q', 'quit']:
            print("\n👋 검색을 종료합니다.")
            break

        try:
            print(f"\n'{query}' 검색을 시작합니다...")
            
            results = rag.search(query, k=k)
            
            print(f"\n✨ 검색 완료! {len(results)}개의 결과를 찾았습니다.\n")
            
            # 결과 출력
            for i, doc in enumerate(results):
                print(f"\n{'='*80}")
                print(f"검색 결과 {i+1}/{len(results)}")
                print(f"제목: {doc.metadata['title']}")
                print(f"날짜: {doc.metadata['date']}")
                print(f"URL: {doc.metadata['url']}")
                print(f"{'-'*40}")
                print(f"내용:\n{doc.page_content[:300]}...")
                
                search_results.append(
                    {
                        "metadata": doc.metadata,
                        "content": doc.page_content,
                    }
                )
            
            # 종료
            break
        
        except Exception as e:
            print(f"\n❌ 검색 중 오류가 발생했습니다: {str(e)}")
    
    return search_results

# tool 설정: 유튜브 검색 
@tool
def search_video(query, max_results=5):
        """
        YouTube API를 사용하여 검색.
        """
        youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=youtube_api_key
        )

        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            order="viewCount",
        )
        response = request.execute()

        results = [
            {
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "publishedAt": item["snippet"]["publishedAt"],
                "channelTitle": item["snippet"]["channelTitle"],
                "video_id": item["id"]["videoId"],
            }
            for item in response.get("items", [])
        ]
        return results

# tools 설정 
tools = [
    Tool(
        name="Search youtube tool",
        func=search_video,
        description="YouTube API를 사용하여 검색합니다."
    ),
    Tool(
        name="Search news tool",
        func=search_news,
        description="AI 뉴스를 검색합니다."
    )
]

# tool 이름 받기
tool_names = [tool.func.name for tool in tools]

# 검색 결과 자료형 설정 
class SearchResult(BaseModel):
    """
    사용자 질문: str
    액션: str
    검색 키워드: str
    tool 설정: str
    """
    user_query: str
    action: str
    search_keywords: str
    tool: str

# 검색 Agent 설정 
class AIAgent:
    def __init__(self, openai_api_key, youtube_api_key, llm_model="gpt-4o"):
        self.openai_api_key = openai_api_key
        self.youtube_api_key = youtube_api_key
        self.llm_model = llm_model
    
    def analyze_query(self, user_query):
        """
        LLM을 사용하여 유저 쿼리를 분석하고 그 결과를 반환.
        """
        llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0.1,
            api_key=self.openai_api_key,
        )
        
        self.output_parser = PydanticOutputParser(
            pydantic_object=SearchResult
        )
        
        self.prompt = PromptTemplate(
            input_variables=["user_query"],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            },
            template=
            """
            당신은 AI 관련 정보를 제공하는 도우미입니다.
            먼저 입력된 질의가 AI 관련 내용인지 확인하세요.

            AI 관련 주제 판단 기준:
            - AI 기술 (머신러닝, 딥러닝, 자연어처리 등)
            - AI 도구 및 서비스 (ChatGPT, DALL-E, Stable Diffusion 등)
            - AI 회사 및 연구소 소식
            - AI 정책 및 규제
            - AI 교육 및 학습
            - AI 윤리 및 영향

            AI 관련 질의가 아닌 경우:
            - action을 "not_supported"로 설정
            - search_keyword는 빈 문자열로 설정            

            AI 관련 질의인 경우 다음 작업을 수행하세요:
            1. 검색 도구 선정: 질의 의도 분석 기반 최적 도구 선택
            2. 키워드 추출: 최적화 검색어 생성

            사용 가능한 도구:
            1. search_video: AI 관련 영상 콘텐츠 검색 특화
            2. search_news: AI 관련 뉴스 및 기사 검색 특화

            도구 선택 기준:
            A) search_video 선정 조건:
            - 영상 콘텐츠 요구 (영상, 동영상)
            - 교육 자료 요청 (강의, 강좌, 수업)
            - 실습 가이드 (튜토리얼, 가이드, 설명)
            - 시각적 설명 (시연, 데모)

            B) search_news 선정 조건:
            - 뉴스 콘텐츠 (뉴스, 소식)
            - 기사 요청 (기사, 글)
            - 정보 탐색 (정보, 현황, 동향)
            - 연구 자료 (연구, 조사, 분석)

            키워드 추출 규칙:
            1. 핵심 주제어 분리
            - AI 관련 핵심 개념 추출
            - 매체 유형 지시어 제거 (정보, 뉴스, 영상, 기사 등)
            - 보조어 및 조사 제거

            2. 의미론적 최적화
            - 전문 용어 완전성 유지
            - 개념 간 관계성 보존
            - 맥락 적합성 확보

            분석 대상 질의: {user_query}

            {format_instructions}
            """,
        )

        # 실행 체인 생성 - 프롬프트 처리부터 결과 파싱까지의 전체 흐름
        self.chain = RunnableSequence(
            first= {"user_query": RunnablePassthrough()} | self.prompt,  # 먼저 프롬프트 처리
            middle=[llm],  # 그 다음 LLM으로 처리
            last=self.output_parser,  # 마지막으로 결과 파싱
        )
        
        response = self.chain.invoke(user_query)  # 질문 분석
        print(response)
        
        return response.model_dump()  # json 형식으로 변형형

    def display_results(self, tool, results):
        """
        검색 결과를 스트림릿으로 보여주기 
        """
        # Youtube 검색 결과 보여주기 
        if tool == 'search_video':
            display_videos(results)
        
        # News 검색 결과 보여주기 
        else:
            display_news(results)

# 이전 대화 기록을 출력해주는 함수
def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            # message type에 따라 다르게 출력
            
            # chat_message.content가 문자열인 경우 
            if isinstance(chat_message.content, str):
                st.chat_message(chat_message.role).write(chat_message.content)
            
            # chat_message.content가 뉴스나 Youtube 검색 결과인 경우 
            else:
                # chat_message.content에 metadata가 있다면 해당 대화는 뉴스를 출력한 것 
                if "metadata" in chat_message.content[0]:
                    display_news(chat_message.content)
                
                # Youtube 검색 결과 출력 
                else:
                    display_videos(chat_message.content)

# Streamlit Part 시작

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Youtube & News AI agent",
    page_icon="🤖",
)

st.title("🤖Youtube & News AI agent🤖")

st.markdown(
    """
    ### AI 관련 정보를 제공하는 도우미 서비스입니다.
    
    - AI와 관련된 주제의 질문을 해주세요.
    - 답변은 Youtube 영상 추천 혹은 관련 뉴스 검색으로 제공됩니다.

    - AI 관련 주제 판단 기준:
        - AI 기술 (머신러닝, 딥러닝, 자연어처리 등)
        - AI 도구 및 서비스 (ChatGPT, DALL-E, Stable Diffusion 등)
        - AI 회사 및 연구소 소식
        - AI 정책 및 규제
        - AI 교육 및 학습
        - AI 윤리 및 영향
    """
)

# Streamlit 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Agent 초기화
agent = AIAgent(openai_api_key, youtube_api_key)

try: 
    # 대화 기록 출력
    print_messages()
    
    # 사용자의 질문 받기 
    if user_input := st.chat_input("궁금한 것을 입력하세요."):
        st.chat_message("user").write(user_input)
        st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
        
        # 쿼리 분석
        print("="*30)
        print("LLM을 통해 입력 쿼리를 분석 중입니다...")
        result = agent.analyze_query(user_input)
        
        st.empty()
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
        
            # tool에 따른 동작 실행 
            if result['tool'] in tool_names:
                # YouTube 검색
                if result['tool'] == 'search_video':
                    print("="*30)
                    print("YouTube에서 검색 중입니다...")
                    search_results = search_video(result['search_keywords'])
                    print("검색을 완료했습니다.")

                    # 검색 결과 표시
                    if search_results:                       
                        st.write("Youtube 검색 결과입니다.")
                        agent.display_results(
                            tool='search_video',
                            results=search_results
                        )
                        
                        st.session_state["messages"].append(
                            ChatMessage(role="assistant", content=search_results)
                        )
                    else:
                        response = "검색 결과가 없습니다."
                        st.write(f"{response}")
                        st.session_state["messages"].append(
                            ChatMessage(role="assistant", content=response)
                        )
            
                # 뉴스 검색
                else:
                    print("="*30)
                    print("뉴스에서 검색 중입니다...")
                    search_results = search_news(result['search_keywords'])

                    # 검색 결과 표시
                    if search_results:                        
                        st.write("News 검색 결과입니다.")
                        agent.display_results(
                            tool='search_news',
                            results=search_results
                        )
                        
                        st.session_state["messages"].append(
                            ChatMessage(role="assistant", content=search_results)
                        )
                    else:
                        response = "검색 결과가 없습니다."
                        st.write(f"{response}")
                        st.session_state["messages"].append(
                            ChatMessage(role="assistant", content=response)
                        )
            
            else:
                response = "AI와 관련된 질문만 받을 수 있습니다."
                st.write(f"{response}")
                st.session_state["messages"].append(
                    ChatMessage(role="assistant", content=response)
                )

except KeyboardInterrupt:
    print("Shutting down process...")
    st.write("Shutting down process...")

except Exception as e:
    print(f"Error occurred: {e}")
    st.write(f"Error occurred: {e}")