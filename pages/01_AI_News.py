from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="AI News",
    page_icon="📰",
)

st.title("AI News")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the sidebar.
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

# 같은 file에 대해 embed_file()을 실행했었다면 cache에서 결과를 바로 반환하는 decorator
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)  # 선택한 파일의 내용을 .cache/files 디렉토리로 옮김

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
    )

    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[:10] | .[].content",  # 
        text_content=True,
    )

    data = loader.load()
    
    #docs = '\n\n'.join(document.page_content for document in data)
    docs = splitter.split_documents(data)

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

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

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

# 사이드바에서 json 파일 업로드
with st.sidebar:
    file = st.file_uploader("Upload a .json", type=["json"])

if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!","ai", save=False)
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

else:
    st.session_state["messages"] = []