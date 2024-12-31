import os
import streamlit as st
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import OpenAIEmbeddings


# Streamlit 페이지 설정
st.set_page_config(
    page_title="AI Book Search",
    page_icon="🤖",
)

st.title("🤖AI Book Search🤖")

st.markdown(
    """
    ### 사용자의 질문을 통해 관련 도서를 추천해주는 AI입니다. 
    
    - 시험이나 자격증을 말해주시고, 본인의 현재 실력도 말해주세요(ex. 상, 중, 하)
    - 답변은 검색된 도서 목록으로 제공됩니다.

    - 도서 관련 주제 판단 기준:
        1. 질문에 도서 제목, 저자, 출판사, 출판 연도, 장르 등의 키워드가 포함되어 있는지 확인하세요.
        2. 질문의 의도가 도서 정보를 요구하거나 책 추천을 요청하는지 분석하세요.
        3. 도서 선택, 세부 정보, 리뷰, 활용과 관련된 일반적인 주제인지 판단하세요.
        4. 질의 유형이 정보 검색형인지, 도서 추천 의도가 있는지 확인하세요.
        5. 도서와 관련 없는 질문(예: \"책상 추천\")은 제외하세요.
    """
)


# 환경 변수 로드 
load_dotenv()

# 임베딩 모델 초기화 
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL")
)

# 환경 변수에서 OpenAI API 키를 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("OPENAI_API_KEY 환경 변수를 설정해주세요.")

# 네이버 clinet 값 불러오기
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
if not NAVER_CLIENT_ID:
    print("NAVER_CLIENT_ID 환경 변수를 설정해주세요.")

NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
if not NAVER_CLIENT_SECRET:
    print("NAVER_CLIENT_SECRET 환경 변수를 설정해주세요.")

NAVER_BOOKS_URL = "https://openapi.naver.com/v1/search/book.json?"


# 도서 검색 
def search_books(query: str, k: int = 10):
    """
    네이버 도서 검색 API를 사용하여 도서를 검색합니다.
    
    Args:
        query (str): 검색어
        k (int): 반환할 결과 수 (기본값 3)
    
    Returns:
        list: 검색 결과 (책 정보 리스트)
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
            
            # HTTP 요청 헤더 설정
            headers = {
                "X-Naver-Client-Id": NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
            }
            
            # 요청 파라미터 설정
            params = {
                "query": query,
                "display": k
            }
            
            # API 요청 보내기
            response = requests.get(NAVER_BOOKS_URL, headers=headers, params=params)
            response.raise_for_status()  # 요청 에러 확인
            
            data = response.json()
            items = data.get("items", [])
            
            search_results = search_results + items
            print(f"\n✨ 검색 완료! {len(search_results)}개의 결과를 찾았습니다.\n")
            
            # 종료
            break
        
        except Exception as e:
            print(f"\n❌ 검색 중 오류가 발생했습니다: {str(e)}")
    
    return search_results


# 검색 결과 자료형 설정 
class SearchResult(BaseModel):
    """
    사용자 질문: str
    액션: str
    검색 키워드: str
    """
    user_query: str
    action: str
    search_keywords: str


# 검색 Agent 설정 
class AIAgent:
    def __init__(self, openai_api_key, llm_model="gpt-4o"):
        self.openai_api_key = openai_api_key
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
            당신은 도서 관련 정보를 제공하는 도우미입니다.
            먼저 입력된 질의가 도서 관련 내용인지 확인하세요.

            도서 관련 주제 판단 기준:
            1. 질문에 도서 제목, 저자, 출판사, 출판 연도, 장르 등의 키워드가 포함되어 있는지 확인하세요.
            2. 질문의 의도가 도서 정보를 요구하거나 책 추천을 요청하는지 분석하세요.
            3. 도서 선택, 세부 정보, 리뷰, 활용과 관련된 일반적인 주제인지 판단하세요.
            4. 질의 유형이 정보 검색형인지, 도서 추천 의도가 있는지 확인하세요.
            5. 도서와 관련 없는 질문(예: \"책상 추천\")은 제외하세요.

            도서 관련 질의가 아닌 경우:
            - action을 "not_supported"로 설정
            - search_keyword는 빈 문자열로 설정            

            도서 관련 질의인 경우 다음 작업을 수행하세요:
            - action을 "search_books"로 설정 
            - 키워드 추출: 최적화 검색어 생성

            키워드 추출 규칙:
            1. 핵심 주제어 분리
            - 도서 관련 핵심 개념 추출
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

# 결과를 보여주는 함수 
def display_results(results):
    """
    검색 결과를 스트림릿으로 보여주기 
    """
    st.write(results)
        


# 이전 대화 기록을 출력해주는 함수
def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)


#text streaming
class StreamHandler(BaseCallbackHandler):
    def __init__ (self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Streamlit Part 시작

# Streamlit 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Agent 초기화
agent = AIAgent(openai_api_key)

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
            st.empty()
            stream_handler = StreamHandler(st.empty())
        
            # YouTube 검색
            if result['action'] == 'search_books':
                print("="*30)
                print("도서 검색 중입니다...")
                search_results = search_books(result['search_keywords'])
                print("검색을 완료했습니다.")

                # 검색 결과 표시
                if search_results:                       
                    st.write("도서 검색 결과입니다.")
                    display_results(search_results)
                    
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
                response = "도서와 관련된 질문만 받을 수 있습니다."
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