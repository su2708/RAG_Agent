import os
import googleapiclient.discovery
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain.agents import Tool
from Keyword_Hybrid_RAG import AINewsRAG

load_dotenv()

# 환경 변수에서 경로 가져오기
vector_store_path = os.getenv("VECTOR_STORE_NAME", "ai_news_vectorstore")
news_dir = os.getenv("NEWS_FILE_PATH", "./ai_news")
processed_doc_path = os.getenv("PROCESSED_DOCS_PATH", "processed_docs/processed_docs.pkl")

# 임베딩 모델 초기화 
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
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
    rag.load_vector_store(vector_store_path, processed_doc_path)
    print("✅ 기존 벡터 스토어를 로드했습니다.")
    
except Exception as e:
    print(f"벡터 스토어 로드 실패: {str(e)}")

# tool 설정: 뉴스 검색 
@tool
def search_news(query: str, k: int = 5):
    """
    하이브리드 검색 방식으로 AI 뉴스를 검색합니다.
    """
    search_mode = "hybrid" # 검색 방식 변경은 'mode [semantic/keyword/hybrid]'를 입력하세요.
    while True:
        query = query.strip()

        if not query:
            continue
            
        if query.lower() in ['q', 'quit']:
            print("\n👋 검색을 종료합니다.")
            break
            
        if query.lower().startswith('mode '):
            mode = query.split()[1].lower()
            if mode in ['semantic', 'keyword', 'hybrid']:
                search_mode = mode
                print(f"\n✅ 검색 모드를 '{mode}'로 변경했습니다.")
            else:
                print("\n❌ 잘못된 검색 모드입니다. semantic/keyword/hybrid 중 선택하세요.")
            continue

        try:
            print(f"\n'{query}' 검색을 시작합니다... (모드: {search_mode})")
            
            if search_mode == "hybrid":
                results = rag.hybrid_search(query, k=k, semantic_weight=0.5)
            elif search_mode == "semantic":
                results = rag.vector_store.similarity_search_with_score(query, k=k)
            else:  # keyword
                results = rag.keyword_search(query, k=k)
            
            print(f"\n✨ 검색 완료! {len(results)}개의 결과를 찾았습니다.\n")
            
            # 결과 출력
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n{'='*80}")
                print(f"검색 결과 {i}/{len(results)}")
                print(f"제목: {doc.metadata['title']}")
                print(f"날짜: {doc.metadata['date']}")
                if search_mode == "hybrid":
                    print(f"통합 점수: {score:.4f}")
                elif search_mode == "semantic":
                    print(f"유사도 점수: {1 - (score/2):.4f}")
                else:
                    print(f"BM25 점수: {score:.4f}")
                print(f"URL: {doc.metadata['url']}")
                print(f"{'-'*40}")
                print(f"내용:\n{doc.page_content[:300]}...")
            
            # 종료
            break
        
        except Exception as e:
            print(f"\n❌ 검색 중 오류가 발생했습니다: {str(e)}")

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
            maxResults=max_results
        )
        response = request.execute()

        results = [
            {
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "video_id": item["id"]["videoId"]
            }
            for item in response.get("items", [])
        ]
        return results

tools = [
    Tool(
        name="Search youtube tool",
        func=search_video,
        description="YouTube API를 사용하여 검색합니다."
    ),
    Tool(
        name="Search news tool",
        func=search_news,
        description="하이브리드 검색 방식으로 AI 뉴스를 검색합니다."
    )
]

# tool 이름 받기
tool_names = [tool.func.name for tool in tools]

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

    def format_results_for_display(self, results):
        """
        검색 결과를 스트림릿에서 보여줄 수 있도록 포맷팅.
        """
        for result in results:
            print(f"### {result['title']}")
            if result['video_id']:
                video_url = f"https://www.youtube.com/watch?v={result['video_id']}"
                print(video_url)

def main():
    try:
        # 유저 입력 받기
        print("AI Search Agent")
        user_query = input("검색할 내용을 입력하세요 (예: AI 뉴스 관련 영상을 알려줘):")

        if user_query:
            # Agent 초기화
            agent = AIAgent(openai_api_key, youtube_api_key)
            
            # 쿼리 분석
            print("="*30)
            print("LLM을 통해 입력 쿼리를 분석 중입니다...")
            result = agent.analyze_query(user_query)
            print(f"검색 결과: {result}")
            
            # tool에 따른 동작 실행 
            if result['tool'] in tool_names:
                # YouTube 검색
                if result['tool'] == 'search_video':
                    print("="*30)
                    print("YouTube에서 검색 중입니다...")
                    search_results = search_video(result['search_keywords'])

                    # 검색 결과 표시
                    if search_results:
                        print("검색 결과:")
                        agent.format_results_for_display(search_results)
                    else:
                        print("검색 결과가 없습니다.")
            
                # 뉴스 검색
                else:
                    print("="*30)
                    print("뉴스에서 검색 중입니다...")
                    search_results = search_news(result['search_keywords'])

                    # 검색 결과 표시
                    if search_results:
                        print("검색 결과:")
                        agent.format_results_for_display(search_results)
                    else:
                        print("검색 결과가 없습니다.")
            
            else:
                print("AI와 관련된 질문만 받을 수 있습니다.")
            
    except KeyboardInterrupt:
        print("Shutting down process...")
    
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()