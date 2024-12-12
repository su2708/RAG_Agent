#!/usr/bin/env python
# coding: utf-8

# # 하이브리드 검색 시스템
# 
# 이 코드는 AI 관련 뉴스 기사를 효과적으로 검색하기 위한 하이브리드 검색 시스템을 구현한 것입니다.
# 벡터 기반 의미론적 검색과 키워드 기반 검색을 결합하여 더 정확한 검색 결과를 제공합니다.
# 
# ## 주요 기능
# 
# 1. **문서 처리**
#    - JSON 형식의 뉴스 데이터 로드
#    - 문서를 청크 단위로 분할
#    - 벡터 DB 및 키워드 검색용 인덱스 생성
# 
# 2. **하이브리드 검색**
#    - 벡터 기반 의미론적 검색 (FAISS)
#    - 키워드 기반 검색 (BM25)
#    - 두 검색 방식의 결과를 가중치를 적용하여 통합
# 
# 3. **데이터 관리**
#    - 벡터 스토어 저장/로드
#    - 처리된 문서 데이터 저장/로드
#    - 진행 상황 로깅
# 
# 
# ## 검색 가중치 설정 가이드
# 
# - **의미론적 검색 중심 (semantic_weight=0.7)**
#   - 문맥과 의미를 더 중요하게 고려
#   - 유사한 주제의 문서도 검색 가능
#   - 예: "AI 기술의 미래 전망" → AI 발전 방향, 기술 트렌드 등 관련 문서 포함
# 
# - **키워드 검색 중심 (semantic_weight=0.3)**
#   - 정확한 키워드 매칭을 중시
#   - 특정 용어나 개념이 포함된 문서 우선
#   - 예: "삼성전자 AI 칩" → 정확히 해당 키워드가 포함된 문서 우선
# 
# - **균형잡힌 검색 (semantic_weight=0.5)**
#   - 두 방식의 장점을 균형있게 활용
#   - 일반적인 검색에 적합
#   - 예: "자율주행 안전" → 키워드 매칭과 의미적 연관성 모두 고려

# In[ ]:


import os
import json
import glob
import pickle
import logging
import sys
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class AINewsRAG:
    """
    AI 뉴스 검색을 위한 RAG(Retrieval-Augmented Generation) 시스템
    
    이 클래스는 뉴스 기사를 벡터 DB로 변환하고, 의미론적 검색과 키워드 기반 검색을
    결합한 하이브리드 검색 기능을 제공합니다.

    Attributes:
        embeddings: OpenAI 임베딩 모델
        text_splitter: 문서 분할을 위한 스플리터
        vector_store: FAISS 벡터 저장소
        bm25: 키워드 기반 검색을 위한 BM25 모델
        processed_docs: 처리된 문서들의 리스트
        doc_mapping: 문서 ID와 문서 객체 간의 매핑
        logger: 로깅을 위한 로거 객체
    """

    def __init__(self, embedding_model):
        """
        AINewsRAG 클래스 초기화

        Args:
            embedding_model: OpenAI 임베딩 모델 인스턴스
        """
        self.embeddings = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.bm25 = None
        self.processed_docs = None
        self.doc_mapping = None
        
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
        """
        여러 JSON 파일에서 뉴스 기사를 로드합니다.

        Args:
            directory_path (str): JSON 파일들이 있는 디렉토리 경로

        Returns:
            List[Dict]: 로드된 뉴스 기사 리스트

        Raises:
            Exception: 파일 로드 중 오류 발생 시
        """
        all_documents = []
        json_files = glob.glob(f"{directory_path}/ai_times_news_*.json")
        
        self.logger.info(f"총 {len(json_files)}개의 JSON 파일을 로드합니다...")
        
        for file_path in tqdm(json_files, desc="JSON 파일 로드 중"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents = json.load(file)
                    if documents:
                        documents = [doc for doc in documents if len(doc['content']) > 10]
                    if len(documents) >= 10:
                        all_documents.extend(documents)
            except Exception as e:
                self.logger.error(f"파일 로드 중 오류 발생: {file_path} - {str(e)}")
        
        self.logger.info(f"총 {len(all_documents)}개의 뉴스 기사를 로드했습니다.")
        return all_documents
    
    def process_documents(self, documents: List[Dict]) -> List[Document]:
        """문서를 처리하고 청크로 분할합니다."""
        processed_docs = []
        self.logger.info("문서 처리 및 청크 분할을 시작합니다...")
        
        for idx, doc in enumerate(tqdm(documents, desc="문서 처리 중")):
            try:
                #첫 정크에 문서의 제목 포함
                full_text = f"{doc['title']}\n {doc['content']}"
                metadata = {
                    'doc_id': idx, 
                    'title': doc['title'],
                    'url': doc['url'],
                    'date': doc['date']
                }
                
                chunks = self.text_splitter.split_text(full_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            'chunk_id': f"doc_{idx}_chunk_{chunk_idx}"  # 청크별 고유 ID
                        }
                    ))
            except Exception as e:
                self.logger.error(f"문서 처리 중 오류 발생: {doc.get('title', 'Unknown')} - {str(e)}")
        
        self.processed_docs = processed_docs
        self.initialize_bm25(processed_docs)
        
        return processed_docs

    def initialize_bm25(self, documents: List[Document]):
        """
        BM25 검색 엔진을 초기화합니다.

        Args:
            documents (List[Document]): 처리된 문서 리스트
        """
        self.logger.info("BM25 검색 엔진을 초기화합니다...")
        
        tokenized_corpus = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_mapping = {
            i: doc for i, doc in enumerate(documents)
        }
        
        self.logger.info("BM25 검색 엔진 초기화가 완료되었습니다.")
    
    def create_vector_store(self, documents: List[Document]):
        """
        FAISS 벡터 스토어를 생성합니다.

        Args:
            documents (List[Document]): 벡터화할 문서 리스트

        Raises:
            Exception: 벡터 스토어 생성 중 오류 발생 시
        """
        self.logger.info("벡터 스토어 생성을 시작합니다...")
        total_docs = len(documents)
        
        try:
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

    def keyword_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        키워드 기반 BM25 검색을 수행합니다.

        Args:
            query (str): 검색 쿼리
            k (int): 반환할 결과 수

        Returns:
            List[Tuple[Document, float]]: (문서, 점수) 튜플의 리스트

        Raises:
            ValueError: BM25가 초기화되지 않은 경우
        """
        if self.bm25 is None:
            raise ValueError("BM25가 초기화되지 않았습니다.")
        
        self.logger.info(f"'{query}' 키워드 검색을 시작합니다...")
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_k_idx = np.argsort(bm25_scores)[-k:][::-1]
        results = [
            (self.doc_mapping[idx], bm25_scores[idx])
            for idx in top_k_idx
        ]
        
        self.logger.info(f"{len(results)}개의 키워드 검색 결과를 찾았습니다.")
        return results

    def hybrid_search(
            self, 
            query: str, 
            k: int = 5, 
            semantic_weight: float = 0.5
        ) -> List[Tuple[Document, float]]:
            """
            의미론적 검색과 키워드 검색을 결합한 하이브리드 검색을 수행합니다.
            """
            self.logger.info(f"'{query}' 하이브리드 검색을 시작합니다...")
            
            # 의미론적 검색 수행
            self.logger.info(f"'{query}' 의미론적 검색을 시작합니다...")
            semantic_results = self.vector_store.similarity_search_with_score(query, k=k)
            self.logger.info(f"{len(semantic_results)}개의 의미론적 검색 결과를 찾았습니다.")

            # 키워드 기반 검색 수행
            keyword_results = self.keyword_search(query, k=k)
            
            # 문서 ID를 키로 사용
            combined_scores = {}
            
            # 의미론적 검색 결과 처리
            max_semantic_score = max(score for _, score in semantic_results)
            for doc, score in semantic_results:
                doc_id = doc.metadata['chunk_id']
                
                #5개의 문서의 점수가
                normalized_score = 1 - (score / max_semantic_score) 
                combined_scores[doc_id] = {
                    'doc': doc,
                    'score': semantic_weight * normalized_score
                }
            
            # 키워드 검색 결과 처리
            max_keyword_score = max(score for _, score in keyword_results)
            for doc, score in keyword_results:
                doc_id = doc.metadata['chunk_id']
                normalized_score = score / max_keyword_score
                if doc_id in combined_scores:
                    combined_scores[doc_id]['score'] += (1 - semantic_weight) * normalized_score
                else:
                    combined_scores[doc_id] = {
                        'doc': doc,
                        'score': (1 - semantic_weight) * normalized_score
                    }
            
            # 결과 정렬
            sorted_results = sorted(
                [(info['doc'], info['score']) for info in combined_scores.values()],
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            self.logger.info(f"{len(sorted_results)}개의 하이브리드 검색 결과를 찾았습니다.")
            return sorted_results

    def save_vector_store(self, vector_store_path: str, processed_docs_path:str=None):
        """
        벡터 스토어와 BM25 데이터를 저장합니다.
        """
        try:
            self.logger.info(f"데이터를 {vector_store_path}에 저장합니다...")
            
            # 벡터 스토어 저장
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)
            
            # processed_docs 저장
            if self.processed_docs:
                os.makedirs(os.path.dirname(processed_docs_path), exist_ok=True)
                with open(processed_docs_path, 'wb') as f:
                    pickle.dump(self.processed_docs, f)
            
            self.logger.info("저장이 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"저장 중 오류 발생: {str(e)}")
            raise

    def load_vector_store(self, vector_store_path: str, processed_docs_path):
        """
        벡터 스토어와 BM25 데이터를 로드합니다.
        """
        try:
            self.logger.info(f"데이터를 {vector_store_path}에서 로드합니다...")
            
            # 벡터 스토어 로드
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # processed_docs 로드
            if os.path.exists(processed_docs_path):
                with open(processed_docs_path, 'rb') as f:
                    self.processed_docs = pickle.load(f)
                self.initialize_bm25(self.processed_docs)
            
            self.logger.info("로드가 완료되었습니다.")
        except Exception as e:
            self.logger.error(f"로드 중 오류 발생: {str(e)}")
            raise


# ### AI뉴스 데이터 Vector DB 구축

# In[ ]:


# 환경 변수 로드
from dotenv import load_dotenv
import os
load_dotenv()

# 임베딩 모델 초기화 
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

# 환경 변수에서 경로 가져오기
vector_store_path = os.getenv("VECTOR_STORE_NAME", "ai_news_vectorstore")
news_dir = os.getenv("NEWS_FILE_PATH", "./ai_news")
processed_doc_path = os.getenv("PROCESSED_DOCS_PATH", "processed_docs/processed_docs.pkl")

# RAG 시스템 초기화
rag = AINewsRAG(embedding_model)

print("새로운 벡터 스토어를 생성합니다...")

# JSON 파일에서 뉴스 데이터 로드
documents = rag.load_json_files(news_dir)

# 문서 처리 및 벡터 스토어 생성
processed_docs = rag.process_documents(documents)
rag.create_vector_store(processed_docs)

# 벡터 스토어 저장
rag.save_vector_store(vector_store_path, processed_doc_path)
print("✅ 새로운 벡터 스토어 생성 및 저장 완료")


# ### 하이브리드 서치 테스트

# In[ ]:


# 환경 변수 로드
from dotenv import load_dotenv
import os
load_dotenv()

# 환경 변수에서 경로 가져오기
vector_store_path = os.getenv("VECTOR_STORE_NAME", "ai_news_vectorstore")
news_dir = os.getenv("NEWS_FILE_PATH", "./ai_news")
processed_doc_path = os.getenv("PROCESSED_DOCS_PATH", "processed_docs/processed_docs.pkl")

# 임베딩 모델 초기화 
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

# RAG 시스템 초기화
rag = AINewsRAG(embedding_model)

try:
    # 기존 벡터 스토어 로드 시도
    rag.load_vector_store(vector_store_path, processed_doc_path)
    print("✅ 기존 벡터 스토어를 로드했습니다.")
    
except Exception as e:
    print(f"벡터 스토어 로드 실패: {str(e)}")

# 대화형 검색 시작
print("\n🔍 AI 뉴스 검색 시스템을 시작합니다.")
print("- 종료하려면 'q' 또는 'quit'를 입력하세요.")

search_mode = "hybrid" # 검색 방식 변경은 'mode [semantic/keyword/hybrid]'를 입력하세요.
while True:
    query = input("\n🔍 검색할 내용을 입력하세요: ").strip()

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
            results = rag.hybrid_search(query, k=5, semantic_weight=0.5)
        elif search_mode == "semantic":
            results = rag.vector_store.similarity_search_with_score(query, k=5)
        else:  # keyword
            results = rag.keyword_search(query, k=5)
        
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
            
    except Exception as e:
        print(f"\n❌ 검색 중 오류가 발생했습니다: {str(e)}")

