# AI 뉴스 검색 시스템 만들기

안녕하세요! 이 프로젝트는 AI 뉴스를 똑똑하게 검색하는 시스템을 만드는 과정을 담고 있습니다. 기본적인 검색부터 시작해서 점점 더 고도화된 검색 방법을 배워볼 수 있도록 구성되어 있습니다.

## 🌟 이런 걸 배울 수 있어요

1. **기본 검색부터 고급 검색까지** 단계별로 발전하는 검색 시스템 구현
2. **벡터 DB**를 사용한 의미 기반 검색 방법
3. **키워드 검색**과 **의미 기반 검색**을 결합하는 방법
4. 검색 결과의 품질을 높이는 **재순위화** 기법

## 📚 단계별 학습 가이드

### Step 1: 의미 기반 검색 구현 (`1_Semantic_RAG.ipynb`)
첫 번째 단계에서는 기본적인 의미 기반 검색을 구현합니다.
- 문서를 적절한 크기로 나누는 방법
- 문서를 벡터로 변환하는 방법
- FAISS를 이용한 벡터 검색


### Step 2: 하이브리드 검색 구현 (`2_Keyword_Hybrid_RAG.ipynb`)
두 번째 단계에서는 키워드 검색을 추가하여 검색의 정확도를 높입니다.
- BM25 알고리즘을 이용한 키워드 검색
- 의미 검색과 키워드 검색의 결합


### Step 3: 고급 검색 기법 (`3_CrossEncoder_RAG.ipynb`)
마지막 단계에서는 검색 결과의 품질을 더욱 높이는 방법을 배웁니다.
- CrossEncoder를 이용한 재순위화
- 검색 결과 비교 및 분석


## 🛠 시작하기 전에 준비할 것들

### 1. 필요한 패키지 설치하기
```bash
# 가상환경 만들기
python -m venv venv

# 가상환경 활성화
# Windows의 경우:
venv\Scripts\activate
# Mac/Linux의 경우:
source venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 주요 패키지들
```bash
# RAG 및 임베딩 관련
langchain==0.3.9
langchain-community==0.3.8
langchain-core==0.3.21
openai==1.55.2
sentence-transformers==3.3.1
faiss-cpu==1.9.0
rank-bm25==0.2.2

# 머신러닝/딥러닝
torch==2.5.1
transformers==4.46.3

# 데이터 처리
numpy==1.26.4
pandas==2.2.3
tqdm==4.67.1

# 유틸리티
python-dotenv==1.0.1
jupyter==1.1.1
tabulate==0.9.0
```

### 3. 환경 설정하기
프로젝트 폴더에 `.env` 파일을 만들고 아래 내용을 입력합니다:
```env
# OpenAI API 키 (꼭 필요!)
OPENAI_API_KEY="여기에_본인의_API_키를_넣으세요"

# 임베딩 모델 설정
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"

# 저장 경로 설정
VECTOR_STORE_NAME="ai_news_vectorstore"
PROCESSED_DOCS_PATH="processed_docs/processed_docs.pkl"
NEWS_FILE_PATH="./ai_news"

# CrossEncoder 모델
CROSS_ENCODER_MODEL="BM-K/KoSimCSE-roberta-multitask"
```

## 🎯 실제로 사용해보기

### 기본적인 검색 시스템 시작하기
```python
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# 환경 변수 불러오기
load_dotenv()

# 임베딩 모델 준비
embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL")
)

# RAG 시스템 시작
rag = AINewsRAG(embedding_model)
```

### 다양한 검색 방법 시도해보기
```python
### 다양한 검색 방법 시도해보기

# 1. 의미 기반 검색
semantic_results = rag.vector_store.similarity_search(
    "인공지능 교육", 
    k=5
)

# 2. 키워드 기반 검색
keyword_results = rag.keyword_search(
    "인공지능 교육", 
    k=5
)

# 3. 하이브리드 검색
hybrid_results = rag.hybrid_search(
    query="인공지능 교육",
    k=5,
    semantic_weight=0.7  # 의미:키워드 = 7:3
)

# 4. CrossEncoder로 고급 검색
advanced_results = rag.advanced_search(
    query="인공지능 교육",
    k=5,                     # 최종적으로 보여줄 결과 수
    semantic_weight=0.7,     # 하이브리드 검색에서 의미 검색 비중
    ce_weight=0.7,          # CrossEncoder 점수 반영 비중
    use_reranking=True,     # 재순위화 사용 여부
    initial_fetch_k=20      # 재순위화할 후보 문서 수
)

# 검색 결과 예쁘게 출력하기
formatted_results = rag.format_search_results(advanced_results)
print(formatted_results)
```

## 💡 검색 가중치 조절하기

### 의미 검색 가중치 (semantic_weight)
- **0.7~0.8**: 비슷한 의미의 문서도 잘 찾고 싶을 때
- **0.2~0.3**: 정확한 단어가 포함된 문서를 찾고 싶을 때
- **0.5**: 둘 다 골고루 고려하고 싶을 때

예시:
- "AI 기술의 미래" → 0.7 (의미 중심)
- "삼성전자 AI 칩" → 0.3 (키워드 중심)

### CrossEncoder 가중치 설정
- **0.8**: 재순위화를 강하게
- **0.7**: 중간 정도로 (권장)
- **0.6**: 약하게

## 📁 프로젝트 구조
```
project/
│
├── 1_Semantic_RAG.ipynb     # 1단계: 기본 검색
├── 2_Keyword_Hybrid_RAG.ipynb  # 2단계: 하이브리드 검색
├── 3_CrossEncoder_RAG.ipynb    # 3단계: 고급 검색
│
├── processed_docs/          # 처리된 문서 저장소
└── ai_news_vectorstore/     # 벡터 DB 저장소
```

## ⚠️ 주의사항
- OpenAI API 키가 필요해요
- `.env` 파일은 절대 GitHub에 올리지 마세요!

## 🤔 자주 하는 질문

1. **벡터 DB가 뭔가요?**
   - 텍스트를 숫자로 변환해서 저장하는 특별한 데이터베이스예요
   - 비슷한 의미를 가진 문서를 빠르게 찾을 수 있어요

2. **하이브리드 검색이 왜 좋나요?**
   - 의미 기반 검색과 키워드 검색의 장점을 모두 활용할 수 있어요
   - 질문의 키워들르 포함하면서 의미까지 고려해 더 정확하고 다양한 검색 결과를 얻을 수 있어요

3. **CrossEncoder는 왜 사용하나요?**
   - 검색 결과의 순서를 더 똑똑하게 조정해주는 도구예요
   - 문맥을 이해해서 더 관련성 높은 결과를 앞으로 배치해줘요


## 7. 참고 자료

### 기술 문서

1. **LangChain Documentation**
- [LangChain](https://python.langchain.com/docs/introduction/)

2. **FAISS (Facebook AI Similarity Search)**
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/)
- 효율적인 벡터 검색 구현 참고

3. **Sentence Transformers**
- [Cross-Encoders Documentation](https://www.sbert.net/docs/package_reference/cross_encoder.html)
- [KoSimCSE Model Card](https://huggingface.co/BM-K/KoSimCSE-roberta-multitask)
- 한국어 특화 Cross-Encoder 구현 참고

4. **Hybrid Search a method to Optimize RAG implementation**
- [하이브리드 서치](https://medium.com/@csakash03/hybrid-search-is-a-method-to-optimize-rag-implementation-98d9d0911341)