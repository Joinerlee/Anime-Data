# 🎌 애니메이션 AI 추천 시스템

임베딩 기반 애니메이션 추천 시스템입니다. 유저의 시청 이력을 기반으로 취향을 분석하고, 콘텐츠 기반 필터링과 협업 필터링을 결합한 하이브리드 추천 알고리즘을 제공합니다.

## 🚀 주요 기능

- **🎯 개인화된 추천**: 유저 시청 이력 기반 맞춤형 추천
- **🧠 하이브리드 알고리즘**: 콘텐츠 기반 + 협업 필터링 결합
- **📊 취향 분석**: 장르, 태그, 연도 등 상세한 취향 프로필 생성
- **🔥 트렌딩 추천**: 최신 인기 애니메이션 추천
- **⚡ 고성능 API**: FastAPI 기반 RESTful API 제공
- **📚 자동 문서화**: Swagger UI 및 ReDoc 지원

## 🛠️ 기술 스택

- **Backend**: Python 3.8+
- **ML/AI**: scikit-learn, numpy, pandas
- **API Framework**: FastAPI
- **임베딩**: 카카오 Kanana 2.1B 모델 (kakaocorp/kanana-nano-2.1b-embedding)
- **Fallback 임베딩**: TF-IDF + SVD
- **딥러닝**: PyTorch, Transformers (Hugging Face)
- **유사도 계산**: Cosine Similarity

## 📦 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
CSV 파일 `anilife_data_20250915_214030.csv`가 프로젝트 루트에 있어야 합니다.

### 3. 데모 실행
```bash
python demo.py
```

### 4. API 서버 실행
```bash
python api_server.py
```

서버가 시작되면 다음 URL에서 확인할 수 있습니다:
- API 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 API 사용법

### 1. 사용자 프로필 생성
```bash
curl -X POST "http://localhost:8000/api/user/profile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "watched_anime": [129, 102, 116],
    "ratings": [5.0, 4.5, 4.8]
  }'
```

### 2. 하이브리드 추천 받기
```bash
curl -X POST "http://localhost:8000/api/recommend/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "n_recommendations": 5
  }'
```

### 3. 애니메이션 검색
```bash
curl "http://localhost:8000/api/anime/search?q=스파이&limit=5"
```

### 4. 트렌딩 애니메이션
```bash
curl "http://localhost:8000/api/trending?year_start=2020&year_end=2025&limit=10"
```

## 📊 추천 알고리즘

### 1. 콘텐츠 기반 필터링 (Kanana 임베딩)
- **카카오 Kanana 2.1B 모델**: 한국어에 최적화된 고성능 임베딩
- **의미론적 임베딩**: 제목, 장르, 태그, 줄거리를 768차원 Dense Vector로 변환
- **문맥 이해**: Transformer 기반으로 텍스트의 의미와 맥락 파악
- **Fallback TF-IDF**: Kanana 불가시 TF-IDF + SVD 사용
- **코사인 유사도**: 고차원 벡터 공간에서의 정밀한 유사도 계산

### 2. 협업 필터링
- **아이템 기반**: 유사한 애니메이션을 본 사용자들의 패턴 분석
- **유사도 매트릭스**: 애니메이션 간 관계 분석

### 3. 하이브리드 추천
- **가중치 결합**: 콘텐츠 기반(60%) + 협업 필터링(40%)
- **점수 정규화**: 두 방법의 점수를 정규화하여 결합
- **최종 랭킹**: 통합 점수로 최종 추천 목록 생성

## 📈 성능 특징

- **데이터 규모**: 4,417개 애니메이션 처리
- **임베딩 차원**: Kanana 768차원 / TF-IDF 5,000차원 → SVD 768차원
- **한국어 최적화**: Kanana 모델로 한국어 애니메이션 제목/설명 정확 처리
- **GPU 가속**: CUDA 지원으로 빠른 임베딩 생성 (선택사항)
- **실시간 추천**: 사용자 요청시 즉시 추천 생성
- **배치 처리**: 메모리 효율적인 배치 임베딩
- **확장성**: 새로운 애니메이션 및 사용자 데이터 쉽게 추가

## 🗂️ 프로젝트 구조

```
Anime-Data/
├── anime_recommender.py    # 핵심 추천 엔진
├── api_server.py          # FastAPI 서버
├── demo.py               # 데모 스크립트
├── requirements.txt      # 의존성 목록
├── README.md            # 프로젝트 문서
└── anilife_data_20250915_214030.csv  # 애니메이션 데이터
```

## 📋 데이터 구조

CSV 파일에는 다음과 같은 컬럼들이 포함되어 있습니다:

- `id`: 애니메이션 고유 ID
- `title_korean`, `title_japanese`, `title_english`: 다국어 제목
- `genres`: 장르 정보 (파이프(|) 구분)
- `tags`: 태그 정보 (파이프(|) 구분)
- `synopsis`: 줄거리
- `year`: 제작 연도
- `director`: 감독
- `studio`: 제작사

## 🔍 추천 시스템 특징

### 개인화 요소
- **장르 선호도**: 사용자가 시청한 애니메이션의 장르 분석
- **태그 선호도**: 세부적인 태그 기반 취향 파악
- **평점 가중치**: 사용자 평점을 고려한 선호도 계산
- **시간대 선호**: 선호하는 제작 연도대 분석

### 추천 다양성
- **다중 알고리즘**: 콘텐츠 기반 + 협업 필터링
- **트렌드 반영**: 최신 애니메이션 트렌드 고려
- **필터 옵션**: 연도, 장르별 필터링 지원

## 🚧 향후 개선 계획

- [ ] **딥러닝 모델**: Transformer 기반 임베딩 적용
- [ ] **사용자 행동**: 클릭, 시청 시간 등 암시적 피드백 활용
- [ ] **실시간 학습**: 새로운 사용자 데이터로 모델 업데이트
- [ ] **A/B 테스트**: 추천 알고리즘 성능 비교 및 최적화
- [ ] **소셜 기능**: 친구 추천, 커뮤니티 기반 추천

## 📞 문의 및 기여

이 프로젝트에 대한 문의사항이나 개선 제안이 있으시면 언제든지 연락해 주세요!

---
**Made with ❤️ for Anime Lovers**