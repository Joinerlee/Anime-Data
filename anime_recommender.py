import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
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
    print("⚠️ transformers 또는 torch가 설치되지 않았습니다. TF-IDF를 대신 사용합니다.")
    print("Kanana 모델을 사용하려면: pip install transformers torch")

class KananaEmbeddingModel:
    def __init__(self, model_name="kakaocorp/kanana-nano-2.1b-embedding", use_kanana=True):
        self.model_name = model_name
        self.use_kanana = use_kanana and KANANA_AVAILABLE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if KANANA_AVAILABLE else None
        
        if self.use_kanana:
            try:
                print(f"🚀 Kanana 임베딩 모델 로딩 중... ({model_name})")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Kanana 모델 로드 완료 (Device: {self.device})")
            except Exception as e:
                print(f"❌ Kanana 모델 로드 실패: {e}")
                print("🔄 TF-IDF로 대체합니다...")
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
            print("🧠 Kanana 임베딩으로 벡터 생성 중...")
            
            # 텍스트 데이터 준비
            texts = self.anime_data['combined_features'].tolist()
            
            # Kanana 모델로 임베딩 생성
            self.content_features = self.embedding_model.encode_texts(texts, batch_size=16)
            
            print(f"✅ Kanana 임베딩 완료 - 크기: {self.content_features.shape}")
            
        else:
            print("📝 TF-IDF 벡터화 사용 중...")
            
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
            
            print(f"✅ TF-IDF + SVD 완료 - 크기: {self.content_features.shape}")
        
        # 코사인 유사도 매트릭스 계산
        print("🔄 유사도 매트릭스 계산 중...")
        self.content_similarity_matrix = self.embedding_model.compute_similarity(self.content_features)
        print(f"✅ 유사도 매트릭스 생성 완료: {self.content_similarity_matrix.shape}")
        
    def create_user_profile(self, user_id, watched_anime_ids, ratings=None):
        """유저 시청 이력을 기반으로 프로필 생성"""
        if ratings is None:
            ratings = [5.0] * len(watched_anime_ids)
        
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
            recommendations.append({
                'id': anime['id'],
                'title': anime['title_korean'] or anime['title_japanese'] or anime['title_english'],
                'similarity_score': float(similarities[idx]),  # numpy float을 Python float로 변환
                'genres': anime['genres'],
                'year': anime['year'],
                'synopsis': anime['synopsis'][:200] + "..." if len(str(anime['synopsis'])) > 200 else anime['synopsis']
            })
        
        return recommendations
    
    def item_based_collaborative_recommend(self, user_id, n_recommendations=10):
        """아이템 기반 협업 필터링 추천"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        watched_indices = [self.anime_data[self.anime_data['id'] == anime_id].index[0] 
                          for anime_id in user_profile['watched_anime']
                          if not self.anime_data[self.anime_data['id'] == anime_id].empty]
        
        if not watched_indices:
            return []
        
        # 시청한 애니메이션들과 유사한 애니메이션 찾기
        item_similarities = {}
        for watched_idx in watched_indices:
            similar_items = self.content_similarity_matrix[watched_idx]
            for i, similarity in enumerate(similar_items):
                if i not in watched_indices and similarity > 0.1:
                    item_similarities[i] = item_similarities.get(i, 0) + similarity
        
        # 상위 N개 추천
        top_items = sorted(item_similarities.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        recommendations = []
        for idx, score in top_items:
            anime = self.anime_data.iloc[idx]
            recommendations.append({
                'id': anime['id'],
                'title': anime['title_korean'] or anime['title_japanese'] or anime['title_english'],
                'similarity_score': score,
                'genres': anime['genres'],
                'year': anime['year'],
                'synopsis': anime['synopsis'][:200] + "..." if len(str(anime['synopsis'])) > 200 else anime['synopsis']
            })
        
        return recommendations
    
    def hybrid_recommend(self, user_id, n_recommendations=10, content_weight=0.6, collaborative_weight=0.4):
        """하이브리드 추천 (콘텐츠 기반 + 협업 필터링)"""
        content_recs = self.content_based_recommend(user_id, n_recommendations*2)
        collab_recs = self.item_based_collaborative_recommend(user_id, n_recommendations*2)
        
        # 추천 점수 정규화 및 결합
        combined_scores = {}
        
        for rec in content_recs:
            anime_id = rec['id']
            combined_scores[anime_id] = {
                'anime': rec,
                'content_score': rec['similarity_score'] * content_weight,
                'collab_score': 0
            }
        
        for rec in collab_recs:
            anime_id = rec['id']
            if anime_id in combined_scores:
                combined_scores[anime_id]['collab_score'] = rec['similarity_score'] * collaborative_weight
            else:
                combined_scores[anime_id] = {
                    'anime': rec,
                    'content_score': 0,
                    'collab_score': rec['similarity_score'] * collaborative_weight
                }
        
        # 최종 점수 계산
        for anime_id in combined_scores:
            total_score = (combined_scores[anime_id]['content_score'] + 
                          combined_scores[anime_id]['collab_score'])
            combined_scores[anime_id]['final_score'] = total_score
        
        # 상위 N개 반환
        sorted_recommendations = sorted(
            combined_scores.items(), 
            key=lambda x: x[1]['final_score'], 
            reverse=True
        )[:n_recommendations]
        
        final_recommendations = []
        for anime_id, scores in sorted_recommendations:
            rec = scores['anime'].copy()
            rec['final_score'] = scores['final_score']
            rec['content_score'] = scores['content_score']
            rec['collab_score'] = scores['collab_score']
            final_recommendations.append(rec)
        
        return final_recommendations
    
    def get_trending_anime(self, year_range=(2020, 2025), n_recommendations=10):
        """트렌딩 애니메이션 추천"""
        recent_anime = self.anime_data[
            (self.anime_data['year'] >= year_range[0]) & 
            (self.anime_data['year'] <= year_range[1])
        ].copy()
        
        # 장르 다양성과 년도를 고려한 점수 계산
        recent_anime['trend_score'] = (
            recent_anime['year'] / recent_anime['year'].max() * 0.5 +
            recent_anime['genres'].str.count('\|') * 0.1 +
            recent_anime['tags'].str.count('\|') * 0.1 +
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