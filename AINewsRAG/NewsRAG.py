import json
import glob
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from tqdm import tqdm
import logging
import sys
import streamlit as st

# 의미 기반 검색 시스템 
class AINewsRAG:
    def __init__(self, embedding_model):
        self.embeddings = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        
        # 로깅 설정
        self.logger = logging.getLogger('AINewsRAG')
        # 기존 핸들러 제거
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        # 로그 중복 방지
        self.logger.propagate = False
        
    def load_json_files(self, directory_path: str) -> List[Dict]:
        """여러 JSON 파일들을 로드합니다."""
        all_documents = []
        json_files = glob.glob(f"{directory_path}/ai_times_news_*.json")
        
        self.logger.info(f"총 {len(json_files)}개의 JSON 파일을 로드합니다...")
        
        for file_path in tqdm(json_files, desc="JSON 파일 로드 중"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents = json.load(file)
                    if documents :
                        documents = [doc for doc in documents if len(doc['content']) > 10]
                    #기사 내용이 없으면 생략
                    if len(documents) >= 10 : 
                        all_documents.extend(documents)
            except Exception as e:
                self.logger.error(f"파일 로드 중 오류 발생: {file_path} - {str(e)}")
        
        self.logger.info(f"총 {len(all_documents)}개의 뉴스 기사를 로드했습니다.")
        return all_documents
    
    def process_documents(self, documents: List[Dict]) -> List[Document]:
        """문서를 처리하고 청크로 분할합니다."""
        processed_docs = []
        self.logger.info("문서 처리 및 청크 분할을 시작합니다...")
        
        for doc in tqdm(documents, desc="문서 처리 중"):
            try:
                full_text = f"제목: {doc['title']}\n내용: {doc['content']}"
                metadata = {
                    'title': doc['title'],
                    'url': doc['url'],
                    'date': doc['date']
                }
                
                chunks = self.text_splitter.split_text(full_text)
                
                for chunk in chunks:
                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))
            except Exception as e:
                self.logger.error(f"문서 처리 중 오류 발생: {doc.get('title', 'Unknown')} - {str(e)}")
        
        self.logger.info(f"총 {len(processed_docs)}개의 청크가 생성되었습니다.")
        return processed_docs
    
    def create_vector_store(self, documents: List[Document]):
        """FAISS 벡터 스토어를 생성합니다."""
        self.logger.info("벡터 스토어 생성을 시작합니다...")
        total_docs = len(documents)
        
        try:
            # 청크를 더 작은 배치로 나누어 처리
            batch_size = 100
            for i in tqdm(range(0, total_docs, batch_size), desc="벡터 생성 중"):
                batch = documents[i:i+batch_size]
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                    self.vector_store.merge_from(batch_vectorstore)
            
            self.logger.info("벡터 스토어 생성이 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
            raise
    
    def save_vector_store(self, path: str):
        """벡터 스토어를 로컬에 저장합니다."""
        try:
            self.logger.info(f"벡터 스토어를 {path}에 저장합니다...")
            self.vector_store.save_local(path)
            self.logger.info("벡터 스토어 저장이 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"벡터 스토어 저장 중 오류 발생: {str(e)}")
            raise
    
    def load_vector_store(self, path: str):
        """저장된 벡터 스토어를 로드합니다."""
        try:
            self.logger.info(f"벡터 스토어를 {path}에서 로드합니다...")
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization =True)
            self.logger.info("벡터 스토어 로드가 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"벡터 스토어 로드 중 오류 발생: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """쿼리와 관련된 문서를 검색합니다."""
        if self.vector_store is None:
            raise ValueError("Vector store has not been initialized")
        
        self.logger.info(f"'{query}' 검색을 시작합니다...")
        results = self.vector_store.similarity_search(query, k=k)
        self.logger.info(f"{len(results)}개의 관련 문서를 찾았습니다.")
        return results

#text streaming
class StreamHandler(BaseCallbackHandler):
    def __init__ (self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# 이전 대화 기록을 출력해주는 함수
def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)
