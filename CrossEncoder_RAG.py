#!/usr/bin/env python
# coding: utf-8

# # AI 뉴스 RAG 시스템 구축 및 테스트
# 
# 이 노트북에서는 CrossEncoder를 활용한 재순위화(Reranking)가 포함된 고급 RAG 검색 시스템을 구현하고 테스트합니다.
# 
# ## 1. RAG 시스템 구현
# 
# `AINewsRAG` 클래스는 다음과 같은 고급 검색 기능을 제공합니다:
# 
# 1. **하이브리드 검색**: 
#    - 벡터 기반 의미 검색 (FAISS)
#    - 키워드 기반 검색 (BM25)
#    - 두 검색 결과의 가중치 기반 통합
# 
# 2. **CrossEncoder 재순위화**: 
#    - 초기 검색 결과의 정확한 관련성 평가
#    - 쿼리-문서 쌍의 문맥 기반 평가
#    - 초기 검색 점수와 CE 점수의 가중치 기반 통합
# 
# 3. **유연한 가중치 조정**: 
#    - 하이브리드 검색 가중치 (semantic_weight)
#    - CrossEncoder 재순위화 가중치 (ce_weight)
# 
# ## 2. 시스템 설정 및 파라미터
# 
# ### 2.1 기본 설정
# ```python
# rag = AINewsRAG(
#     embedding_model=OpenAIEmbeddings(),
#     cross_encoder_name="BM-K/KoSimCSE-roberta-multitask"
# )
# ```
# 
# ### 2.2 주요 파라미터
# 1. **하이브리드 검색 가중치** (semantic_weight)
#    - 의미 중심: 0.7~0.8 (문맥 이해 중요)
#    - 키워드 중심: 0.2~0.3 (정확한 매칭 중요)
#    - 균형: 0.5 (두 방식 균형)
# 
# 2. **CrossEncoder 가중치** (ce_weight)
#    - 강한 재순위화: 0.8 (CE 점수 중심)
#    - 중간 재순위화: 0.7 (권장값)
#    - 약한 재순위화: 0.6 (초기 순위 유지)
# 
# 3. **재순위화 설정**
#    - initial_fetch_k: 20 (k의 4배 권장)
#    - use_reranking: True/False
# 
# ## 3. 검색 프로세스
# 
# ### 3.1 기본 하이브리드 검색
# ```python
# results = rag.hybrid_search(
#     query="AI 기술",
#     k=5,
#     semantic_weight=0.7
# )
# ```
# 
# ### 3.2 CrossEncoder 재순위화 검색
# ```python
# results = rag.advanced_search(
#     query="AI 기술",
#     k=5,                    # 최종 결과 수
#     semantic_weight=0.7,    # 하이브리드 검색 가중치
#     ce_weight=0.7,         # CrossEncoder 가중치
#     use_reranking=True,    # 재순위화 사용
#     initial_fetch_k=20     # 재순위화 후보 수
# )
# ```
# 
# ## 4. 성능 최적화 가이드
# 
# ### 4.1 검색 품질 최적화
# - 하이브리드 검색 가중치 조정
# - CrossEncoder 가중치 조정
# - initial_fetch_k 값 설정 (k의 4~5배 권장)
# 
# ### 4.2 속도 최적화
# - CrossEncoder 배치 처리
# - 적절한 initial_fetch_k 설정
# - 캐싱 활용
# 
# ## 5. 환경 설정
# ```env
# # OpenAI 설정
# OPENAI_API_KEY=your-api-key
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# 
# # 경로 설정
# VECTOR_STORE_NAME=ai_news_vectorstore
# PROCESSED_DOCS_PATH=processed_docs/processed_docs.pkl
# 
# # CrossEncoder 모델
# CROSS_ENCODER_MODEL=BM-K/KoSimCSE-roberta-multitask
# ```
# 
# ## 6. 결과 해석
# 
# ### 6.1 검색 결과 포맷
# - 문서 제목과 날짜
# - 관련성 점수 (정규화된 값)
# - URL 및 내용 미리보기
# - 점수 해석:
#   - 하이브리드 점수: 0~1 (높을수록 관련성 높음)
#   - CrossEncoder 점수: 0~1 (재순위화 후 최종 점수)
# 
# ### 6.2 성능 분석
# ```python
# results_df = compare_search_results(
#     query="AI 교육",
#     rag=rag,
#     k=5
# )
# ```
# - 키워드 검색 vs 의미 검색 vs 하이브리드 vs CrossEncoder 결과 비교
# - 순위 변화 분석
# - 점수 분포 확인
# 
# 이 시스템은 문맥 기반의 정확한 검색과 키워드 기반의 정확한 매칭을 결합하여, 더 관련성 높은 검색 결과를 제공합니다.

# **Cross Encoder 핵심 원리**
# - 쿼리와 문서를 하나의 시퀀스로 입력: [CLS] Query [SEP] Document [SEP]
# - 관련성 있는 쌍(1)과 없는 쌍(0)으로 학습
# - 전체 문맥을 고려한 상호작용 학습
# 
# **RAG에서의 실제 예시**
# ```
# 쿼리: "파이썬 머신러닝 입문 방법"
# 
# 1. 초기 검색 (Bi-encoder)
#    - "자바 프로그래밍 기초" (0.8)
#    - "파이썬 머신러닝 가이드" (0.75)
#    - "R로 시작하는 통계분석" (0.7)
#    
# 2. Cross Encoder 재순위화
#    - "파이썬 머신러닝 가이드" (0.95) ⬆️
#    - "R로 시작하는 통계분석" (0.4)
#    - "자바 프로그래밍 기초" (0.2) ⬇️
# ```
# 
# **개선 효과**
# - Bi-encoder: 단순 키워드 매칭으로 "자바..."가 높은 점수
# - Cross Encoder: 문맥 이해로 실제 관련 문서 "파이썬 머신러닝..."이 상위로 이동
# - 결과: LLM에 더 관련성 높은 문서 제공 → 더 정확한 답변 생성

# In[6]:


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
from sentence_transformers import CrossEncoder

class AINewsRAG:
    def __init__(self, embedding_model, 
                cross_encoder_name: str = "BM-K/KoSimCSE-roberta-multitask"):
        """
        AINewsRAG 클래스 초기화
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
        
        # CrossEncoder 초기화
        try:
            self.cross_encoder = CrossEncoder(cross_encoder_name)
        except Exception as e:
            self.logger.warning(f"CrossEncoder 초기화 실패: {str(e)}")
            self.cross_encoder = None

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
        
    def rerank_with_cross_encoder(
        self, 
        query: str, 
        initial_results: List[Tuple[Document, float]], 
        top_k: int = 5,
        alpha: float = 0.7  # CrossEncoder 점수 가중치
    ) -> List[Tuple[Document, float]]:
        """
        CrossEncoder와 초기 검색 점수를 결합하여 재순위화를 수행합니다.
        
        Args:
            alpha: CrossEncoder 점수의 가중치 (0~1)
            1-alpha: 초기 검색 점수의 가중치
        """
        pairs = [[query, doc.page_content] for doc, _ in initial_results]
        
        try:
            # CrossEncoder 점수 계산
            cross_scores = self.cross_encoder.predict(pairs)
            
            # 점수 정규화
            max_cross_score = max(cross_scores)
            norm_cross_scores = [score/max_cross_score for score in cross_scores]
            
            max_initial_score = max(score for _, score in initial_results)
            norm_initial_scores = [score/max_initial_score for _, score in initial_results]
            
            # 가중치를 적용한 점수 결합
            reranked = [
                (doc, alpha * cross_score + (1-alpha) * init_score)
                for (doc, _), cross_score, init_score 
                in zip(initial_results, norm_cross_scores, norm_initial_scores)
            ]
            
            # 결과 정렬
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_k]
            
        except Exception as e:
            self.logger.error(f"재순위화 중 오류 발생: {str(e)}")
            return initial_results[:top_k]
    
    

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
            
            #키워드 기반 검색 수행
            keyword_results = self.keyword_search(query, k=k)
            
            # 문서 ID를 키로 사용
            combined_scores = {}
            
            # 의미론적 검색 결과 처리
            max_semantic_score = max(score for _, score in semantic_results)
            for doc, score in semantic_results:
                doc_id = doc.metadata['chunk_id']
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
        
    def advanced_search(
        self, 
        query: str, 
        k: int = 5, 
        semantic_weight: float = 0.5,     #
        ce_weight: float = 0.7,         
        use_reranking: bool = True,
        initial_fetch_k: int = 20
    ) -> List[Tuple[Document, float]]:
        """
        하이브리드 검색에 재순위화를 추가한 고급 검색을 수행합니다.

        Args:
            query: 검색 쿼리
            k: 최종 반환할 결과 수
            semantic_weight: 하이브리드 검색에서 의미론적 검색의 가중치 (0~1)
                        - 0: 키워드 검색만 사용
                        - 1: 의미 검색만 사용
            ce_weight: CrossEncoder 재순위화 점수의 가중치 (0.5~0.9)
                    - 높을수록 CrossEncoder 점수를 더 중요하게 고려
                    - 기본값 0.7 권장
            use_reranking: CrossEncoder 재순위화 사용 여부
            initial_fetch_k: 재순위화를 위한 초기 검색 결과 수

        Returns:
            검색 결과 리스트 [(Document, score)]

        Raises:
            ValueError: 가중치 값이 범위를 벗어난 경우
        """
        # 가중치 값 검증
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight는 0과 1 사이의 값이어야 합니다.")
        
        if not 0.5 <= ce_weight <= 0.9:
            self.logger.warning(
                f"권장 범위(0.5~0.9)를 벗어난 ce_weight: {ce_weight}"
            )
        
        self.logger.info(
            f"고급 검색 시작 - 쿼리: '{query}', "
            f"의미검색 가중치: {semantic_weight}, "
            f"CE 가중치: {ce_weight}"
        )

        # 1단계: 하이브리드 검색으로 후보 문서 추출
        initial_k = initial_fetch_k if use_reranking else k
        initial_results = self.hybrid_search(
            query=query,
            k=initial_k,
            semantic_weight=semantic_weight
        )

        # 2단계: CrossEncoder로 재순위화 (선택적)
        if use_reranking and self.cross_encoder:
            final_results = self.rerank_with_cross_encoder(
                query=query,
                initial_results=initial_results,
                top_k=k,
                alpha=ce_weight  # CrossEncoder 가중치 전달
            )
        else:
            final_results = initial_results[:k]

        self.logger.info(
            f"검색 완료: {len(final_results)}개 결과, "
            f"최고 점수: {final_results[0][1]:.4f}"
        )
        
        return final_results

    def format_search_results(
        self, 
        results: List[Tuple[Document, float]], 
        show_score: bool = True
    ) -> str:
        """검색 결과를 보기 좋게 포맷팅합니다."""
        output = []
        for i, (doc, score) in enumerate(results, 1):
            output.append(f"\n{'='*80}")
            output.append(f"검색 결과 {i}/{len(results)}")
            output.append(f"제목: {doc.metadata['title']}")
            output.append(f"날짜: {doc.metadata['date']}")
            if show_score:
                output.append(f"관련도 점수: {score:.4f}")
            output.append(f"URL: {doc.metadata['url']}")
            output.append(f"{'-'*40}")
            output.append(f"내용:\n{doc.page_content[:300]}...")
        
        return "\n".join(output)


# ### RAG 고급 검색 시스템 테스트 및 검색 방법에 따른 비교

# In[7]:


# 환경 변수 로드
from dotenv import load_dotenv
import os
load_dotenv()

# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

# RAG 시스템 초기화
rag = AINewsRAG(embedding_model, os.getenv("CROSS_ENCODER_MODEL"))

# 벡터 스토어 로드
vector_store_path = os.getenv("VECTOR_STORE_NAME")
processed_docs_path = os.getenv("PROCESSED_DOCS_PATH")
rag.load_vector_store(vector_store_path, processed_docs_path)

# 검색 수행
query = "최신 AI 기술 동향"

# CrossEncoder를 활용한 고급 검색
results = rag.advanced_search(
    query=query,
    k=5,                     # 최종 결과 수
    semantic_weight=0.7,     # 하이브리드 검색 가중치
    ce_weight=0.7,          # CrossEncoder 가중치
    use_reranking=True,     # 재순위화 사용
    initial_fetch_k=20      # 재순위화 후보 수
)

# 검색 결과 출력
formatted_results = rag.format_search_results(results)
print(formatted_results)


# #### 검색 결과 비교

# In[ ]:


import pandas as pd
from tabulate import tabulate

def compare_search_results(query: str, rag: AINewsRAG, k: int = 5):
    """
    다양한 검색 방식의 결과를 비교하여 표로 정리합니다.
    """
    # 1. 키워드 검색 중심
    keyword_focused = rag.advanced_search(
        query=query,
        k=k,
        semantic_weight=0.0,
        use_reranking=False
    )

    # 2. 의미론적 검색 중심
    semantic_focused = rag.advanced_search(
        query=query,
        k=k,
        semantic_weight=1.0,
        use_reranking=False
    )


    # 3. 하이브리드 검색 
    hybrid_results = rag.hybrid_search(query, k=k, semantic_weight=0.5)


    # 4. CrossEncoder 재순위화 검색
    advanced_results = rag.advanced_search(
        query=query,
        k=k,
        semantic_weight=0.5,
        ce_weight=0.7,
        use_reranking=True,
        initial_fetch_k=20
    )


    # 결과를 DataFrame으로 변환
    results = []

    for idx in range(k):
        row = {"순위": idx + 1}
        
        # 키워드 중심 결과
        if idx < len(keyword_focused):
            doc, score = keyword_focused[idx]
            row.update({
                "키워드_제목": doc.metadata['title'],
                "키워드_점수": f"{score:.4f}"
            })
            
        
        # 의미론적 중심 결과
        if idx < len(semantic_focused):
            doc, score = semantic_focused[idx]
            row.update({
                "의미론적_제목": doc.metadata['title'],
                "의미론적_점수": f"{score:.4f}"
            })
            
            
        # 하이브리드 검색 결과
        if idx < len(hybrid_results):
            doc, score = hybrid_results[idx]
            row.update({
                "하이브리드_제목": doc.metadata['title'],
                "하이브리드_점수": f"{score:.4f}"
            })
            
        # CrossEncoder 결과
        if idx < len(advanced_results):
            doc, score = advanced_results[idx]
            row.update({
                "CrossEncoder_제목": doc.metadata['title'],
                "CrossEncoder_점수": f"{score:.4f}"
            })
            
        
        results.append(row)

    # DataFrame 생성
    df = pd.DataFrame(results)

    # 표 출력
    print(f"\n검색어: {query}")
    print("\n검색 결과 비교:")
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

    # 결과 분석
    print("\n결과 분석:")

    # 중복 문서 분석
    all_titles = []
    for results in [hybrid_results, advanced_results, semantic_focused, keyword_focused]:
        all_titles.extend([doc.metadata['title'] for doc, _ in results])

    unique_titles = set(all_titles)
    print(f"\n총 unique 문서 수: {len(unique_titles)}")

    return df


# 환경 변수 로드
from dotenv import load_dotenv
import os
load_dotenv()

# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

# RAG 시스템 초기화
rag = AINewsRAG(embedding_model, os.getenv("CROSS_ENCODER_MODEL"))

# 벡터 스토어 로드
vector_store_path = os.getenv("VECTOR_STORE_NAME")
processed_docs_path = os.getenv("PROCESSED_DOCS_PATH")
rag.load_vector_store(vector_store_path, processed_docs_path)

# 검색 수행
query = "AI 교육"
results_df = compare_search_results(query, rag, k=5)

