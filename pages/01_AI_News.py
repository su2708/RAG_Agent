from langchain_community.document_loaders import JSONLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.schema import Document
import streamlit as st
import os

st.set_page_config(
    page_title="AI News",
    page_icon="📰",
)

st.title("AI News")

st.markdown(
    """
    환영합니다!
    
    이 챗봇을 사용하여 파일에 대해 AI에게 질문하세요!
    
    사이드바에서 AI 뉴스를 업로드하세요.
    """
)

# llm의 streaming 응답을 표시하기 위한 callback handler
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.message = ""  # 빈 message 문자열 생성
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()  # 빈 message box 생성
    
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")  # 응답이 끝나면 세션에 AI 응답으로 저장
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token  # streaming 응답을 위해 토큰을 계속 덧붙임
        self.message_box.markdown(self.message) 

llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])

# 문서 요약 시 요약 내용이 화면에 출력되는 것을 막기 위한 llm
llm_silent = ChatOpenAI(temperature=0.1, streaming=True)

# LLM을 활용한 문서 요약 함수
def summarize_documents(docs, llm):
    summarized_docs = []
    for doc in docs:
        # 요약 요청을 LLM으로 보내기
        summary = llm.predict(f"이 내용 요약해줘: {doc.page_content}")
        
        # Document 객체에 담아 summarized_docs에 추가 
        summarized_docs.append(Document(page_content=summary))
    return summarized_docs

# 같은 file에 대해 embed_file()을 실행했었다면 cache에서 결과를 바로 반환하는 decorator
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, summarize=False):
    file_content = file.read()  # 파일의 내용을 읽어오기
    file_path = f"./.cache/files/{file.name}"  # 저장될 파일의 경로
    with open(file_path, "wb") as f:
        f.write(file_content)  # 선택한 파일의 내용을 .cache/files 디렉토리로 옮김

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=200,
        chunk_overlap=20,
    )

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[:5] | .[].content",  # 5개의 기사만 가져오기
        text_content=True,
    )

    data = loader.load()
    
    # 요약하기 버튼을 선택한 경우
    if summarize:
        # 요약한 문서의 임베딩 결과를 저장할 디렉토리 만들기
        cache_dir_path = f"./.cache/summarized_embeddings/{file.name}"
        os.makedirs(cache_dir_path, exist_ok=True) # 디렉토리가 없으면 만들기
        cache_dir = LocalFileStore(cache_dir_path)
        
        docs = splitter.split_documents(data)  # splitter에 맞게 문서 분할
        
        # 문서 요약
        docs = summarize_documents(docs, llm_silent)
        
        embeddings = OpenAIEmbeddings()

        # 중복 요청 시 캐시된 결과를 반환
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        
    # 요약하기 버튼을 선택하지 않은 경우
    else:
        # 문서의 임베딩 결과를 저장할 디렉토리 만들기
        cache_dir_path = f"./.cache/embeddings/{file.name}"
        os.makedirs(cache_dir_path, exist_ok=True) # 디렉토리가 없으면 만들기
        cache_dir = LocalFileStore(cache_dir_path)
        
        docs = splitter.split_documents(data)  # splitter에 맞게 문서 분할

        embeddings = OpenAIEmbeddings()

        # 중복 요청 시 캐시된 결과를 반환
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )

    # FAISS 라이브러리로 캐시에서 임베딩 벡터 검색
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    # docs를 불러오는 역할
    retriever = vectorstore.as_retriever()

    return retriever

# 주고 받은 메시지를 세션 상태에 저장하는 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

# role에 맞게 메시지를 보내는 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        save_message(message, role)

# 채팅 기록을 채팅 화면에 보여주는 함수
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

# docs를 이중 줄바꿈으로 구분된 하나의 문자열로 반환하는 함수
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# 문서에 있는 내용으로만 답변하라는 프롬프트 생성 
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DON'T make anything up.
        
        Context: {context}
        """,
    ),
    ("human", "{question}"),
])

# 사이드바에서 문서 요약 여부 선택 & json 파일 업로드
with st.sidebar:
    summarize = st.radio(
        "Summarize content for embedding?",
        ["No", "Yes"]
    )
    file = st.file_uploader("Upload a .json file", type=["json"])

# summarize의 초기 상태를 No로 설정
if "previous_radio_value" not in st.session_state:
    st.session_state.previous_radio_value = "No"

# 파일이 업로드 되었을 때 
if file:
    # 1. summarize == 'Yes'
    #     - 업로드한 문서를 LLM을 통해 요약한 후 임베딩 
    #     - summarize 값이 이전 상태와 다른 경우, 기존 대화 내역을 삭제 
    #     - 문서를 요약해서 임베딩한 파일을 retriever로 불러와 사용
    
    # 2. summarize == 'No'
    #     - 업로드한 문서를 그대로 임베딩 
    #     - summarize 값이 이전 상태와 다른 경우, 기존 대화 내역을 삭제 
    #     - 문서를 그대로 임베딩한 파일을 retriever로 불러와 사용
        
    # 3. 임베딩이 완료되어야 메시지 창 활성화 

    if summarize == "Yes":
        if summarize != st.session_state.previous_radio_value:
            if "messages" in st.session_state:
                st.session_state["messages"] = []
        st.session_state.previous_radio_value = summarize
        retriever = embed_file(file, summarize=True)
    else:
        if summarize != st.session_state.previous_radio_value:
            if "messages" in st.session_state:
                st.session_state["messages"] = []
        st.session_state.previous_radio_value = summarize
        retriever = embed_file(file, summarize=False)

    # 임베딩 완료 신호 
    send_message("준비됐습니다! 질문해주세요!","ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")

        # chain_type = stuff
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm

        with st.chat_message("ai"):
            chain.invoke(message)

# 파일이 업로드 되지 않았을 때 
else:
    st.session_state["messages"] = []  # 대화 내역 없음 