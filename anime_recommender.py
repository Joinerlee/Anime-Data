import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import json
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Kanana 임베딩 모델 (Hugging Face)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    KANANA_AVAILABLE = True
except ImportError:
    KANANA_AVAILABLE = False
    print("WARNING: transformers or torch not installed. Using TF-IDF instead.")
    print("To use Kanana model: pip install transformers torch")

class KananaEmbeddingModel:
    def __init__(self, model_name="kakaocorp/kanana-nano-2.1b-embedding", use_kanana=True):
        self.model_name = model_name
        self.use_kanana = use_kanana and KANANA_AVAILABLE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if KANANA_AVAILABLE else None
        
        if self.use_kanana:
            try:
                print(f"Loading Kanana embedding model... ({model_name})")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                self.model.to(self.device)
                self.model.eval()
                print(f"Kanana model loaded successfully (Device: {self.device})")
            except Exception as e:
                print(f"Kanana model loading failed: {e}")
                print("Switching to TF-IDF...")
                self.use_kanana = False
        
        # Fallback to TF-IDF if Kanana is not available
        if not self.use_kanana:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            self.svd = TruncatedSVD(n_components=768, random_state=42)  # Kanana와 비슷한 차원
    
    def encode_texts(self, texts, batch_size=32):
        """텍스트 리스트를 임베딩으로 변환"""
        if self.use_kanana:
            return self._encode_with_kanana(texts, batch_size)
        else:
            return self._encode_with_tfidf(texts)
    
    def _encode_with_kanana(self, texts, batch_size=32):
        """Kanana 모델로 임베딩 생성"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 토큰화
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # 임베딩 생성
                outputs = self.model(**inputs)
                
                # Mean pooling (CLS 토큰 대신 평균 풀링 사용)
                attention_mask = inputs['attention_mask']
                last_hidden_states = outputs.last_hidden_state
                
                # 마스크를 고려한 평균 풀링
                masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
                sum_embeddings = masked_embeddings.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1, keepdim=True)
                mean_embeddings = sum_embeddings / sum_mask
                
                embeddings.extend(mean_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def _encode_with_tfidf(self, texts):
        """TF-IDF + SVD로 임베딩 생성 (fallback)"""
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        embeddings = self.svd.fit_transform(tfidf_matrix.toarray())
        
        # L2 정규화
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings, norm='l2')
        
        return embeddings
    
    def compute_similarity(self, embeddings1, embeddings2=None):
        """임베딩 간 코사인 유사도 계산"""
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        return cosine_similarity(embeddings1, embeddings2)

class AnimeRecommendationSystem:
    def __init__(self, use_kanana=True):
        self.anime_data = None
        self.content_features = None
        self.collaborative_features = None
        self.user_profiles = {}
        self.content_similarity_matrix = None
        self.svd_model = None
        
        # Kanana 임베딩 모델 초기화
        self.embedding_model = KananaEmbeddingModel(use_kanana=use_kanana)
        self.use_kanana = self.embedding_model.use_kanana

        # TF-IDF는 fallback용으로만 사용
        if not self.use_kanana:
            self.tfidf_vectorizer = None

        # 새로운 추천 알고리즘을 위한 변수들
        self.user_item_matrix = None
        self.svd_collaborative = None
        self.nmf_model = None
        self.knn_model = None
        self.popularity_scores = None
        
    def load_data(self, csv_path):
        """CSV 데이터 로드 및 전처리"""
        print("데이터 로딩 중...")
        self.anime_data = pd.read_csv(csv_path, encoding='utf-8')
        print(f"총 {len(self.anime_data)}개의 애니메이션 데이터 로드됨")
        
        # 데이터 클리닝
        self._clean_data()
        print("데이터 전처리 완료")
        
    def _clean_data(self):
        """데이터 전처리 및 클리닝"""
        # 결측값 처리
        self.anime_data['genres'] = self.anime_data['genres'].fillna('')
        self.anime_data['tags'] = self.anime_data['tags'].fillna('')
        self.anime_data['synopsis'] = self.anime_data['synopsis'].fillna('')
        self.anime_data['year'] = pd.to_numeric(self.anime_data['year'], errors='coerce')
        self.anime_data['duration'] = pd.to_numeric(self.anime_data['duration'], errors='coerce')
        
        # 텍스트 정규화
        self.anime_data['combined_features'] = (
            self.anime_data['title_korean'].fillna('') + ' ' +
            self.anime_data['title_japanese'].fillna('') + ' ' +
            self.anime_data['title_english'].fillna('') + ' ' +
            self.anime_data['genres'] + ' ' +
            self.anime_data['tags'] + ' ' +
            self.anime_data['synopsis']
        )
        
    def build_content_features(self):
        """콘텐츠 기반 특성 벡터 생성 (Kanana 임베딩 사용)"""
        print("콘텐츠 기반 특성 벡터 생성 중...")
        
        if self.use_kanana:
            print("Generating vectors with Kanana embedding...")

            # 텍스트 데이터 준비
            texts = self.anime_data['combined_features'].tolist()

            # Kanana 모델로 임베딩 생성
            self.content_features = self.embedding_model.encode_texts(texts, batch_size=16)

            print(f"Kanana embedding completed - size: {self.content_features.shape}")

        else:
            print("Using TF-IDF vectorization...")

            # TF-IDF 벡터화 (fallback)
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )

            tfidf_features = self.tfidf_vectorizer.fit_transform(
                self.anime_data['combined_features']
            )

            # SVD로 차원 축소
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=768, random_state=42)
            self.content_features = svd.fit_transform(tfidf_features.toarray())

            # L2 정규화
            from sklearn.preprocessing import normalize
            self.content_features = normalize(self.content_features, norm='l2')

            print(f"TF-IDF + SVD completed - size: {self.content_features.shape}")

        # 코사인 유사도 매트릭스 계산
        print("Computing similarity matrix...")
        self.content_similarity_matrix = self.embedding_model.compute_similarity(self.content_features)
        print(f"Similarity matrix completed: {self.content_similarity_matrix.shape}")
        
    def create_user_profile(self, user_id, watched_anime_ids, ratings=None):
        """유저 시청 이력을 기반으로 프로필 생성"""
        if ratings is None:
            ratings = [3.0] * len(watched_anime_ids)  # 기본 평점을 중간값으로 변경
        
        user_profile = {
            'user_id': user_id,
            'watched_anime': watched_anime_ids,
            'ratings': ratings,
            'preferences': self._analyze_user_preferences(watched_anime_ids, ratings)
        }
        
        self.user_profiles[user_id] = user_profile
        return user_profile
    
    def _analyze_user_preferences(self, watched_anime_ids, ratings):
        """유저 취향 분석"""
        watched_data = self.anime_data[self.anime_data['id'].isin(watched_anime_ids)]
        
        # 장르 선호도
        genre_preferences = {}
        tag_preferences = {}
        
        for idx, anime in watched_data.iterrows():
            rating = ratings[watched_anime_ids.index(anime['id'])] if anime['id'] in watched_anime_ids else 3.0
            
            # 장르 분석
            genres = str(anime['genres']).split('|') if pd.notna(anime['genres']) else []
            for genre in genres:
                if genre.strip():
                    genre_preferences[genre.strip()] = genre_preferences.get(genre.strip(), 0) + rating
            
            # 태그 분석
            tags = str(anime['tags']).split('|') if pd.notna(anime['tags']) else []
            for tag in tags:
                if tag.strip():
                    tag_preferences[tag.strip()] = tag_preferences.get(tag.strip(), 0) + rating
        
        return {
            'genre_preferences': dict(sorted(genre_preferences.items(), key=lambda x: x[1], reverse=True)),
            'tag_preferences': dict(sorted(tag_preferences.items(), key=lambda x: x[1], reverse=True)),
            'avg_rating': np.mean(ratings),
            'preferred_years': watched_data['year'].dropna().tolist(),
            'preferred_formats': watched_data['format'].dropna().tolist()
        }
    
    def content_based_recommend(self, user_id, n_recommendations=10):
        """콘텐츠 기반 추천 (Kanana 임베딩 사용)"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        watched_indices = [self.anime_data[self.anime_data['id'] == anime_id].index[0] 
                          for anime_id in user_profile['watched_anime']
                          if not self.anime_data[self.anime_data['id'] == anime_id].empty]
        
        if not watched_indices:
            return []
        
        # 시청한 애니메이션들의 평균 임베딩 벡터 계산
        if self.use_kanana:
            # Kanana 임베딩은 이미 numpy array
            user_vector = np.mean(self.content_features[watched_indices], axis=0)
            
            # 모든 애니메이션과의 유사도 계산
            similarities = self.embedding_model.compute_similarity([user_vector], self.content_features)[0]
        else:
            # TF-IDF + SVD 벡터
            user_vector = np.mean(self.content_features[watched_indices], axis=0)
            similarities = cosine_similarity([user_vector], self.content_features)[0]
        
        # 이미 시청한 애니메이션 제외
        for idx in watched_indices:
            similarities[idx] = -1
        
        # 상위 N개 추천
        top_indices = similarities.argsort()[-n_recommendations:][::-1]
        
        recommendations = []
        for idx in top_indices:
            anime = self.anime_data.iloc[idx]

            # 상세 점수 정보 계산
            detailed_scores = self._get_detailed_scores(anime['id'], user_id)

            recommendations.append({
                'id': anime['id'],
                'title': anime['title_korean'] or anime['title_japanese'] or anime['title_english'],
                'similarity_score': float(similarities[idx]),  # numpy float을 Python float로 변환
                'genres': anime['genres'],
                'year': anime['year'],
                'synopsis': anime['synopsis'][:200] + "..." if len(str(anime['synopsis'])) > 200 else anime['synopsis'],
                # 추가된 상세 정보
                'genre_similarity': detailed_scores['genre_similarity'],
                'preference_score': detailed_scores['preference_score'],
                'anime_genres': detailed_scores['anime_genres'],
                'user_top_genres': detailed_scores['user_top_genres'],
                'matched_genres': detailed_scores['matched_genres'],
                'recommendation_reason': self._generate_recommendation_reason(detailed_scores, 'content')
            })
        
        return recommendations
    
    def item_based_collaborative_recommend(self, user_id, n_recommendations=10):
        """개선된 아이템 기반 협업 필터링 추천"""
        if user_id not in self.user_profiles:
            return []

        user_profile = self.user_profiles[user_id]
        watched_anime_ids = set(user_profile['watched_anime'])
        user_ratings = dict(zip(user_profile['watched_anime'], user_profile['ratings']))

        if not watched_anime_ids:
            return []

        # 선호 장르/태그 기반 가중치 계산
        genre_prefs = user_profile['preferences']['genre_preferences']
        tag_prefs = user_profile['preferences']['tag_preferences']

        # 사용자가 좋아할 만한 애니메이션 찾기 (평점 3.5 이상 시청작 기준)
        liked_anime = [anime_id for anime_id, rating in user_ratings.items() if rating >= 3.5]
        liked_indices = []

        for anime_id in liked_anime:
            anime_rows = self.anime_data[self.anime_data['id'] == anime_id]
            if not anime_rows.empty:
                liked_indices.append(anime_rows.index[0])

        if not liked_indices:
            # 좋아하는 애니메이션이 없으면 콘텐츠 기반으로 폴백
            return self._content_based_fallback(user_id, n_recommendations)

        # 선호도 기반 점수 계산
        candidate_scores = {}

        for idx, anime in self.anime_data.iterrows():
            if anime['id'] in watched_anime_ids:
                continue

            score = 0.0

            # 1. 장르 선호도 점수
            anime_genres = str(anime['genres']).split('|') if pd.notna(anime['genres']) else []
            for genre in anime_genres:
                genre = genre.strip()
                if genre in genre_prefs:
                    score += genre_prefs[genre] * 0.3

            # 2. 태그 선호도 점수
            anime_tags = str(anime['tags']).split('|') if pd.notna(anime['tags']) else []
            for tag in anime_tags:
                tag = tag.strip()
                if tag in tag_prefs:
                    score += tag_prefs[tag] * 0.2

            # 3. 콘텐츠 유사도 (좋아한 애니메이션과의 평균 유사도)
            if idx < len(self.content_similarity_matrix):
                similarities = [self.content_similarity_matrix[idx][liked_idx]
                              for liked_idx in liked_indices
                              if liked_idx < len(self.content_similarity_matrix)]
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    score += avg_similarity * 0.5

            candidate_scores[idx] = score

        # 상위 N개 추천
        top_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

        recommendations = []
        for idx, score in top_items:
            if score > 0:  # 최소 점수 필터링
                anime = self.anime_data.iloc[idx]

                # 상세 점수 정보 계산
                detailed_scores = self._get_detailed_scores(anime['id'], user_id)

                recommendations.append({
                    'id': anime['id'],
                    'title': anime['title_korean'] or anime['title_japanese'] or anime['title_english'],
                    'similarity_score': score,
                    'genres': anime['genres'],
                    'year': anime['year'],
                    'synopsis': anime['synopsis'][:200] + "..." if len(str(anime['synopsis'])) > 200 else anime['synopsis'],
                    # 추가된 상세 정보
                    'genre_similarity': detailed_scores['genre_similarity'],
                    'preference_score': detailed_scores['preference_score'],
                    'anime_genres': detailed_scores['anime_genres'],
                    'user_top_genres': detailed_scores['user_top_genres'],
                    'matched_genres': detailed_scores['matched_genres'],
                    'recommendation_reason': self._generate_recommendation_reason(detailed_scores, 'collaborative')
                })

        return recommendations

    def _content_based_fallback(self, user_id, n_recommendations=10):
        """협업 필터링 실패시 콘텐츠 기반 폴백"""
        # 기존 콘텐츠 기반 추천을 간단히 호출
        return self.content_based_recommend(user_id, n_recommendations)

    def _calculate_genre_similarity(self, anime_id, user_id):
        """애니메이션과 사용자 선호 장르 간의 유사성 계산"""
        if user_id not in self.user_profiles:
            return 0.0

        user_profile = self.user_profiles[user_id]
        user_genre_prefs = user_profile['preferences']['genre_preferences']

        if not user_genre_prefs:
            return 0.0

        # 애니메이션의 장르 가져오기
        anime_row = self.anime_data[self.anime_data['id'] == anime_id]
        if anime_row.empty:
            return 0.0

        anime_genres = str(anime_row.iloc[0]['genres']).split('|') if pd.notna(anime_row.iloc[0]['genres']) else []
        anime_genres = [g.strip() for g in anime_genres if g.strip()]

        if not anime_genres:
            return 0.0

        # 장르별 선호도 점수 계산
        total_preference = 0.0
        max_possible_score = 0.0

        for genre in anime_genres:
            if genre in user_genre_prefs:
                total_preference += user_genre_prefs[genre]
            max_possible_score += max(user_genre_prefs.values()) if user_genre_prefs else 5.0

        # 정규화 (0~1 범위)
        similarity = total_preference / max_possible_score if max_possible_score > 0 else 0.0
        return min(similarity, 1.0)

    def _calculate_preference_score(self, anime_id, user_id):
        """애니메이션에 대한 사용자 종합 선호도 점수 계산"""
        if user_id not in self.user_profiles:
            return 0.0

        user_profile = self.user_profiles[user_id]
        genre_prefs = user_profile['preferences']['genre_preferences']
        tag_prefs = user_profile['preferences']['tag_preferences']

        # 애니메이션 정보 가져오기
        anime_row = self.anime_data[self.anime_data['id'] == anime_id]
        if anime_row.empty:
            return 0.0

        anime = anime_row.iloc[0]
        score = 0.0

        # 1. 장르 선호도 (가중치 40%)
        anime_genres = str(anime['genres']).split('|') if pd.notna(anime['genres']) else []
        genre_score = 0.0
        for genre in anime_genres:
            genre = genre.strip()
            if genre in genre_prefs:
                genre_score += genre_prefs[genre]

        if anime_genres:
            genre_score = genre_score / len(anime_genres)  # 평균
        score += genre_score * 0.4

        # 2. 태그 선호도 (가중치 30%)
        anime_tags = str(anime['tags']).split('|') if pd.notna(anime['tags']) else []
        tag_score = 0.0
        for tag in anime_tags:
            tag = tag.strip()
            if tag in tag_prefs:
                tag_score += tag_prefs[tag]

        if anime_tags:
            tag_score = tag_score / len(anime_tags)  # 평균
        score += tag_score * 0.3

        # 3. 년도 선호도 (가중치 20%)
        preferred_years = user_profile['preferences']['preferred_years']
        if preferred_years and pd.notna(anime['year']):
            year_diff = min([abs(anime['year'] - year) for year in preferred_years])
            year_score = max(0, 5.0 - year_diff * 0.5)  # 년도 차이가 클수록 점수 감소
            score += year_score * 0.2

        # 4. 기본 점수 (가중치 10%)
        score += 2.5 * 0.1  # 기본 점수

        return min(score, 5.0)  # 최대 5점으로 제한

    def _get_detailed_scores(self, anime_id, user_id):
        """애니메이션에 대한 상세 점수 정보 반환"""
        genre_similarity = self._calculate_genre_similarity(anime_id, user_id)
        preference_score = self._calculate_preference_score(anime_id, user_id)

        # 애니메이션 장르 정보
        anime_row = self.anime_data[self.anime_data['id'] == anime_id]
        if not anime_row.empty:
            anime_genres = str(anime_row.iloc[0]['genres']).split('|') if pd.notna(anime_row.iloc[0]['genres']) else []
            anime_genres = [g.strip() for g in anime_genres if g.strip()]
        else:
            anime_genres = []

        # 사용자 상위 선호 장르
        if user_id in self.user_profiles:
            user_genre_prefs = self.user_profiles[user_id]['preferences']['genre_preferences']
            top_user_genres = list(user_genre_prefs.keys())[:5] if user_genre_prefs else []
        else:
            top_user_genres = []

        return {
            'genre_similarity': round(genre_similarity, 4),
            'preference_score': round(preference_score, 4),
            'anime_genres': anime_genres,
            'user_top_genres': top_user_genres,
            'matched_genres': [g for g in anime_genres if g in top_user_genres]
        }

    def _generate_recommendation_reason(self, detailed_scores, method):
        """추천 이유 텍스트 생성"""
        reasons = []

        # 장르 매칭 기반 이유
        matched_genres = detailed_scores['matched_genres']
        if matched_genres:
            if len(matched_genres) == 1:
                reasons.append(f"'{matched_genres[0]}' 장르를 선호하시는 취향과 일치")
            else:
                genre_text = "', '".join(matched_genres[:2])  # 최대 2개만 표시
                reasons.append(f"'{genre_text}' 장르를 선호하시는 취향과 일치")

        # 선호도 점수 기반 이유
        pref_score = detailed_scores['preference_score']
        if pref_score >= 4.0:
            reasons.append("높은 선호도 예상 (★★★★★)")
        elif pref_score >= 3.5:
            reasons.append("좋은 선호도 예상 (★★★★☆)")
        elif pref_score >= 3.0:
            reasons.append("보통 선호도 예상 (★★★☆☆)")

        # 추천 방법 기반 이유
        if method == 'content':
            reasons.append("시청 작품과 내용 유사성 높음")
        elif method == 'collaborative':
            reasons.append("선호 패턴 분석 기반 추천")
        elif method == 'hybrid':
            reasons.append("종합 분석을 통한 최적 매칭")

        # 장르 유사성 기반 이유
        genre_sim = detailed_scores['genre_similarity']
        if genre_sim >= 0.8:
            reasons.append("장르 취향 매우 일치")
        elif genre_sim >= 0.6:
            reasons.append("장르 취향 일치")

        # 기본 이유 (아무것도 없을 때)
        if not reasons:
            reasons.append("시청 이력 기반 추천")

        return " | ".join(reasons[:3])  # 최대 3개 이유만 표시
    
    def hybrid_recommend(self, user_id, n_recommendations=10, content_weight=0.6, collaborative_weight=0.4):
        """개선된 하이브리드 추천 (콘텐츠 기반 + 사용자 선호도 기반)"""
        if user_id not in self.user_profiles:
            return []

        # 더 많은 후보를 생성하여 다양성 확보
        content_recs = self.content_based_recommend(user_id, n_recommendations * 3)
        collab_recs = self.item_based_collaborative_recommend(user_id, n_recommendations * 3)

        # 점수 정규화를 위한 최대값 계산
        content_max = max([rec['similarity_score'] for rec in content_recs]) if content_recs else 1.0
        collab_max = max([rec['similarity_score'] for rec in collab_recs]) if collab_recs else 1.0

        # 추천 점수 정규화 및 결합
        combined_scores = {}

        # 콘텐츠 기반 추천 처리
        for rec in content_recs:
            anime_id = rec['id']
            normalized_content_score = (rec['similarity_score'] / content_max) * content_weight

            combined_scores[anime_id] = {
                'anime': rec,
                'content_score': normalized_content_score,
                'collab_score': 0,
                'method': 'content'
            }

        # 협업 필터링 추천 처리
        for rec in collab_recs:
            anime_id = rec['id']
            normalized_collab_score = (rec['similarity_score'] / collab_max) * collaborative_weight

            if anime_id in combined_scores:
                # 이미 콘텐츠 기반에서 추천된 경우
                combined_scores[anime_id]['collab_score'] = normalized_collab_score
                combined_scores[anime_id]['method'] = 'hybrid'
            else:
                # 협업 필터링에서만 추천된 경우
                combined_scores[anime_id] = {
                    'anime': rec,
                    'content_score': 0,
                    'collab_score': normalized_collab_score,
                    'method': 'collaborative'
                }

        # 최종 점수 계산 및 다양성 보너스
        for anime_id in combined_scores:
            base_score = (combined_scores[anime_id]['content_score'] +
                         combined_scores[anime_id]['collab_score'])

            # 하이브리드 추천에 보너스 (두 방법 모두에서 추천된 경우)
            if combined_scores[anime_id]['method'] == 'hybrid':
                base_score *= 1.1  # 10% 보너스

            combined_scores[anime_id]['final_score'] = base_score

        # 다양성을 위한 장르 균형 조정
        final_scores = self._apply_diversity_filter(combined_scores, user_id)

        # 상위 N개 반환
        sorted_recommendations = sorted(
            final_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:n_recommendations]

        final_recommendations = []
        for anime_id, scores in sorted_recommendations:
            rec = scores['anime'].copy()
            rec['final_score'] = scores['final_score']
            rec['content_score'] = scores['content_score']
            rec['collab_score'] = scores['collab_score']
            rec['recommendation_method'] = scores['method']

            # 하이브리드 추천에서는 상세 점수가 이미 있을 수 있지만,
            # 없는 경우를 위해 다시 계산
            if 'genre_similarity' not in rec:
                detailed_scores = self._get_detailed_scores(anime_id, user_id)
                rec.update({
                    'genre_similarity': detailed_scores['genre_similarity'],
                    'preference_score': detailed_scores['preference_score'],
                    'anime_genres': detailed_scores['anime_genres'],
                    'user_top_genres': detailed_scores['user_top_genres'],
                    'matched_genres': detailed_scores['matched_genres'],
                    'recommendation_reason': self._generate_recommendation_reason(detailed_scores, scores['method'])
                })

            final_recommendations.append(rec)

        return final_recommendations

    def _apply_diversity_filter(self, combined_scores, user_id):
        """추천 다양성을 위한 장르 균형 조정"""
        if user_id not in self.user_profiles:
            return combined_scores

        user_profile = self.user_profiles[user_id]
        watched_genres = set()

        # 사용자가 시청한 장르 추출
        for anime_id in user_profile['watched_anime']:
            anime_rows = self.anime_data[self.anime_data['id'] == anime_id]
            if not anime_rows.empty:
                anime_genres = str(anime_rows.iloc[0]['genres']).split('|')
                watched_genres.update([g.strip() for g in anime_genres if g.strip()])

        # 추천 애니메이션의 장르 분포 계산
        genre_count = {}
        for anime_id, score_data in combined_scores.items():
            anime_row = self.anime_data[self.anime_data['id'] == anime_id]
            if not anime_row.empty:
                anime_genres = str(anime_row.iloc[0]['genres']).split('|')
                for genre in anime_genres:
                    genre = genre.strip()
                    if genre:
                        genre_count[genre] = genre_count.get(genre, 0) + 1

        # 장르별 다양성 보너스/페널티 적용
        for anime_id, score_data in combined_scores.items():
            anime_row = self.anime_data[self.anime_data['id'] == anime_id]
            if not anime_row.empty:
                anime_genres = str(anime_row.iloc[0]['genres']).split('|')
                diversity_bonus = 0

                for genre in anime_genres:
                    genre = genre.strip()
                    if genre:
                        # 사용자가 시청하지 않은 새로운 장르에 보너스
                        if genre not in watched_genres:
                            diversity_bonus += 0.05

                        # 너무 많이 추천된 장르에 페널티
                        if genre_count.get(genre, 0) > 3:
                            diversity_bonus -= 0.02

                # 다양성 보너스 적용
                score_data['final_score'] = score_data.get('final_score', 0) + diversity_bonus

        return combined_scores
    
    def get_trending_anime(self, year_range=(2020, 2025), n_recommendations=10):
        """트렌딩 애니메이션 추천"""
        recent_anime = self.anime_data[
            (self.anime_data['year'] >= year_range[0]) & 
            (self.anime_data['year'] <= year_range[1])
        ].copy()
        
        # 장르 다양성과 년도를 고려한 점수 계산
        recent_anime['trend_score'] = (
            recent_anime['year'] / recent_anime['year'].max() * 0.5 +
            recent_anime['genres'].str.count('\\|') * 0.1 +
            recent_anime['tags'].str.count('\\|') * 0.1 +
            0.3  # 기본 점수
        )
        
        trending = recent_anime.nlargest(n_recommendations, 'trend_score')
        
        recommendations = []
        for idx, anime in trending.iterrows():
            recommendations.append({
                'id': anime['id'],
                'title': anime['title_korean'] or anime['title_japanese'] or anime['title_english'],
                'trend_score': anime['trend_score'],
                'genres': anime['genres'],
                'year': anime['year'],
                'synopsis': anime['synopsis'][:200] + "..." if len(str(anime['synopsis'])) > 200 else anime['synopsis']
            })
        
        return recommendations
    
    def save_model(self, filepath):
        """모델 저장 (Kanna 모델 제외)"""
        model_data = {
            'user_profiles': self.user_profiles,
            'content_features': self.content_features,
            'content_similarity_matrix': self.content_similarity_matrix,
            'use_kanana': self.use_kanana,
            'anime_data_columns': self.anime_data.columns.tolist() if self.anime_data is not None else None
        }
        
        # TF-IDF 모델이 있으면 저장
        if not self.use_kanana and hasattr(self.embedding_model, 'tfidf_vectorizer'):
            model_data['tfidf_vectorizer'] = self.embedding_model.tfidf_vectorizer
            model_data['svd_model'] = self.embedding_model.svd
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"모델이 {filepath}에 저장되었습니다.")
        print("⚠️ Kanana 모델은 별도로 다운로드됩니다.")
    
    def load_model(self, filepath):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_profiles = model_data['user_profiles']
        self.content_features = model_data['content_features']
        self.content_similarity_matrix = model_data['content_similarity_matrix']
        
        # Kanana 모델 사용 여부 복원
        saved_use_kanana = model_data.get('use_kanana', False)
        if saved_use_kanana and not self.use_kanana:
            print("⚠️ 저장된 모델은 Kanana를 사용했지만 현재 Kanana가 비활성화되어 있습니다.")
        
        # TF-IDF 모델 복원 (필요한 경우)
        if not self.use_kanana:
            if 'tfidf_vectorizer' in model_data:
                self.embedding_model.tfidf_vectorizer = model_data['tfidf_vectorizer']
            if 'svd_model' in model_data:
                self.embedding_model.svd = model_data['svd_model']
        
        print(f"모델이 {filepath}에서 로드되었습니다.")
        if saved_use_kanana:
            print("✅ Kanana 모델 설정이 복원되었습니다.")
    
    def evaluate_recommendations(self, test_users, test_data, k=10):
        """추천 성능 평가"""
        precision_scores = []
        recall_scores = []
        
        for user_id, actual_anime in test_data.items():
            if user_id in test_users:
                recommendations = self.hybrid_recommend(user_id, k)
                recommended_ids = [rec['id'] for rec in recommendations]
                
                # Precision과 Recall 계산
                relevant_items = set(actual_anime)
                recommended_items = set(recommended_ids)
                
                if len(recommended_items) > 0:
                    precision = len(relevant_items & recommended_items) / len(recommended_items)
                    precision_scores.append(precision)
                
                if len(relevant_items) > 0:
                    recall = len(relevant_items & recommended_items) / len(relevant_items)
                    recall_scores.append(recall)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score
        }