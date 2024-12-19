import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from AINewsRAG.NewsRAG import AINewsRAG, StreamHandler
from AI_agent.Youtube_AI import print_messages
from dotenv import load_dotenv
from tqdm import tqdm
import streamlit as st

# Streamlit 페이지 설정
st.set_page_config(
    page_title="AI News",
    page_icon="📰",
)

st.title("📰AI News📰")

st.markdown(
    """
    ### AI 관련 뉴스를 검색하는 서비스입니다.
    
    - AI와 관련된 주제의 질문을 해주세요.
    - 답변은 관련 뉴스 검색으로 제공됩니다.

    - AI 관련 주제 판단 기준:
        - AI 기술 (머신러닝, 딥러닝, 자연어처리 등)
        - AI 도구 및 서비스 (ChatGPT, DALL-E, Stable Diffusion 등)
        - AI 회사 및 연구소 소식
        - AI 정책 및 규제
        - AI 교육 및 학습
        - AI 윤리 및 영향
    """
)

# 환경 변수 로드
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

vector_store_path = "ai_news_vectorstore/"

# 임베딩 모델
embed_model = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

# RAG 시스템 초기화
rag = AINewsRAG(embed_model)

try:
    # 기존 벡터 스토어 로드 시도
    rag.load_vector_store(vector_store_path)
    print("✅ 기존 벡터 스토어를 로드했습니다.")
    
except Exception as e:
    print(f"벡터 스토어 로드 실패: {str(e)}")


# Streamlit part 시작

# Streamlit 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    summarize = st.radio(
        "뉴스를 LLM으로 요약해서 임베딩 후 저장할까요?",
        (False, True)
    )

if summarize:
    # 경로 설정 
    json_files_path = "ai_news/"
    vector_store_path = "summarized_ai_news_vectorstore/"
    
    # 요약 폴더 확인 
    if not os.path.exists(vector_store_path):
        print(f"{vector_store_path} 폴더를 새로 생성합니다.")
        os.makedirs(vector_store_path)
    
        # JSON 파일들 로드
        documents = rag.load_json_files(json_files_path)
        
        # 문서 처리
        processed_docs = rag.process_documents(documents[:100])
        
        # 요약 프롬프트 생성 
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", "질문으로 들어온 문서를 요약해주세요."),
            ("human", "{doc}"),
        ])
        
        # 요약 LLM 생성
        summarize_llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0.1,
        )
        
        # 요약 체인 설정 
        summarize_chain = {"doc": RunnablePassthrough()}|summarize_prompt|summarize_llm
        
        # 문서 요약하기 
        summarized_processed_docs = []
        for processed_doc in tqdm(processed_docs, desc="문서 요약 중"):
            metadata = processed_doc.metadata
            doc = processed_doc.page_content
            
            # 문서 요약 처리하는 부분 
            res = summarize_chain.invoke(doc).content
            
            summarized_processed_docs.append(Document(
                metadata=metadata,
                page_content=res
            ))
        
        # 요약된 문서로 벡터스토어 생성 
        rag.create_vector_store(summarized_processed_docs)
        
        # 요약 벡터스토어 저장 
        rag.save_vector_store(vector_store_path)
        
        try:
            # 저장된 요약 벡터 스토어 로드 시도
            rag.load_vector_store(vector_store_path)
            print("✅ 신규 요약 벡터 스토어를 로드했습니다.")
            
        except Exception as e:
            print(f"신규 요약 벡터 스토어 로드 실패: {str(e)}")
            
    else:
        try:
            # 요약 벡터 스토어 로드 시도
            rag.load_vector_store(vector_store_path)
            print("✅ 기존 요약 벡터 스토어를 로드했습니다.")
            
        except Exception as e:
            print(f"기존 요약 벡터 스토어 로드 실패: {str(e)}")

# 대화 내역 출력
print_messages()

# 사용자의 질문 받기 
if user_input := st.chat_input("궁금한 것을 입력하세요."):
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    
    # 리트리버에서 문서 검색
    relevant_docs = rag.search(user_input, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    max_context_length = 3000  # 검색된 문서의 최대 길이 제한
    context = context[:max_context_length]
    
    # 대화 기록 출력
    print_messages()

    # AI 응답 생성
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    context를 참고하여 답변을 생성하세요.
                    모르면 답변을 지어내지 말고 모른다고 말하세요.
                    
                    아래는 리트리버에서 가져온 데이터입니다:\n{context}
                    """
                ),
                ("human", "{question}"),
            ]
        )
        
        # 모델 생성
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0.1,
            streaming=True,
            callbacks=[stream_handler],
            max_tokens=500
        )

        chain = {
            "context": lambda x:context,
            "question": RunnablePassthrough(),
        } | prompt | llm

        # 사용자 입력 처리 및 AI 응답 생성
        response = chain.invoke(
            {"question": user_input},
        )
        
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response.content)
        )