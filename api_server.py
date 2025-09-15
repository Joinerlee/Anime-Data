#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
애니메이션 추천 시스템 REST API 서버 (FastAPI)
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from anime_recommender import AnimeRecommendationSystem
import os
import uvicorn

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="애니메이션 추천 API",
    description="임베딩 기반 애니메이션 추천 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 추천 시스템 객체
recommender = None

# Pydantic 모델들
class UserProfileRequest(BaseModel):
    user_id: str
    watched_anime: List[int]
    ratings: Optional[List[float]] = None

class RecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: Optional[int] = 10

class HybridRecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: Optional[int] = 10
    content_weight: Optional[float] = 0.6
    collaborative_weight: Optional[float] = 0.4

class AnimeResponse(BaseModel):
    id: int
    title_korean: Optional[str] = None
    title_japanese: Optional[str] = None
    title_english: Optional[str] = None
    genres: Optional[str] = None
    year: Optional[float] = None
    synopsis: Optional[str] = None

class RecommendationResponse(BaseModel):
    id: int
    title: str
    similarity_score: Optional[float] = None
    final_score: Optional[float] = None
    content_score: Optional[float] = None
    collab_score: Optional[float] = None
    genres: Optional[str] = None
    year: Optional[float] = None
    synopsis: Optional[str] = None

def initialize_recommender():
    """추천 시스템 초기화"""
    global recommender
    try:
        recommender = AnimeRecommendationSystem()
        csv_path = "anilife_data_20250915_214030.csv"
        
        if os.path.exists(csv_path):
            recommender.load_data(csv_path)
            recommender.build_content_features()
            logger.info("추천 시스템 초기화 완료")
            return True
        else:
            logger.error(f"데이터 파일을 찾을 수 없습니다: {csv_path}")
            return False
    except Exception as e:
        logger.error(f"추천 시스템 초기화 실패: {str(e)}")
        return False

def get_recommender():
    """의존성 주입을 위한 추천 시스템 반환"""
    if recommender is None:
        raise HTTPException(status_code=500, detail="추천 시스템이 초기화되지 않았습니다.")
    return recommender

@app.on_event("startup")
async def startup_event():
    """서버 시작시 추천 시스템 초기화"""
    if not initialize_recommender():
        logger.error("추천 시스템 초기화 실패")

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "message": "애니메이션 추천 API 서버가 정상 작동 중입니다."
    }

@app.get("/api/anime/search", response_model=Dict[str, Any])
async def search_anime(
    q: str = Query(..., description="검색어"),
    limit: int = Query(10, description="결과 개수"),
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """애니메이션 검색"""
    try:
        if not q:
            raise HTTPException(status_code=400, detail="검색어가 필요합니다.")
        
        # 제목에서 검색
        results = rec.anime_data[
            rec.anime_data['title_korean'].str.contains(q, case=False, na=False) |
            rec.anime_data['title_japanese'].str.contains(q, case=False, na=False) |
            rec.anime_data['title_english'].str.contains(q, case=False, na=False)
        ].head(limit)
        
        anime_list = []
        for idx, anime in results.iterrows():
            anime_list.append({
                "id": anime['id'],
                "title_korean": anime['title_korean'],
                "title_japanese": anime['title_japanese'],
                "title_english": anime['title_english'],
                "genres": anime['genres'],
                "year": anime['year'],
                "synopsis": anime['synopsis']
            })
        
        return {
            "results": anime_list,
            "count": len(anime_list)
        }
        
    except Exception as e:
        logger.error(f"애니메이션 검색 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="검색 중 오류가 발생했습니다.")

@app.post("/api/user/profile")
async def create_user_profile(
    request: UserProfileRequest,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """사용자 프로필 생성"""
    try:
        if not request.watched_anime:
            raise HTTPException(status_code=400, detail="시청한 애니메이션 목록이 필요합니다.")
        
        # 평점이 제공되지 않은 경우 기본값 사용
        ratings = request.ratings
        if not ratings:
            ratings = [5.0] * len(request.watched_anime)
        elif len(ratings) != len(request.watched_anime):
            raise HTTPException(status_code=400, detail="시청한 애니메이션 수와 평점 수가 일치하지 않습니다.")
        
        # 사용자 프로필 생성
        profile = rec.create_user_profile(request.user_id, request.watched_anime, ratings)
        
        return {
            "message": "사용자 프로필이 생성되었습니다.",
            "profile": {
                "user_id": profile['user_id'],
                "watched_count": len(profile['watched_anime']),
                "avg_rating": profile['preferences']['avg_rating'],
                "top_genres": list(profile['preferences']['genre_preferences'].items())[:5],
                "top_tags": list(profile['preferences']['tag_preferences'].items())[:5]
            }
        }
        
    except Exception as e:
        logger.error(f"사용자 프로필 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="프로필 생성 중 오류가 발생했습니다.")

@app.post("/api/recommend/content", response_model=Dict[str, Any])
async def content_based_recommend(
    request: RecommendationRequest,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """콘텐츠 기반 추천"""
    try:
        if request.user_id not in rec.user_profiles:
            raise HTTPException(status_code=404, detail="사용자 프로필이 존재하지 않습니다.")
        
        recommendations = rec.content_based_recommend(request.user_id, request.n_recommendations)
        
        return {
            "user_id": request.user_id,
            "method": "content_based",
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"콘텐츠 기반 추천 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="추천 생성 중 오류가 발생했습니다.")

@app.post("/api/recommend/collaborative", response_model=Dict[str, Any])
async def collaborative_recommend(
    request: RecommendationRequest,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """협업 필터링 추천"""
    try:
        if request.user_id not in rec.user_profiles:
            raise HTTPException(status_code=404, detail="사용자 프로필이 존재하지 않습니다.")
        
        recommendations = rec.item_based_collaborative_recommend(request.user_id, request.n_recommendations)
        
        return {
            "user_id": request.user_id,
            "method": "collaborative_filtering",
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"협업 필터링 추천 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="추천 생성 중 오류가 발생했습니다.")

@app.post("/api/recommend/hybrid", response_model=Dict[str, Any])
async def hybrid_recommend(
    request: HybridRecommendationRequest,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """하이브리드 추천"""
    try:
        if request.user_id not in rec.user_profiles:
            raise HTTPException(status_code=404, detail="사용자 프로필이 존재하지 않습니다.")
        
        recommendations = rec.hybrid_recommend(
            request.user_id, 
            request.n_recommendations, 
            request.content_weight, 
            request.collaborative_weight
        )
        
        return {
            "user_id": request.user_id,
            "method": "hybrid",
            "weights": {
                "content": request.content_weight,
                "collaborative": request.collaborative_weight
            },
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"하이브리드 추천 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="추천 생성 중 오류가 발생했습니다.")

@app.get("/api/trending", response_model=Dict[str, Any])
async def get_trending(
    year_start: int = Query(2020, description="시작 년도"),
    year_end: int = Query(2025, description="끝 년도"),
    limit: int = Query(10, description="결과 개수"),
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """트렌딩 애니메이션"""
    try:
        trending = rec.get_trending_anime((year_start, year_end), limit)
        
        return {
            "method": "trending",
            "year_range": [year_start, year_end],
            "recommendations": trending
        }
        
    except Exception as e:
        logger.error(f"트렌딩 애니메이션 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="트렌딩 조회 중 오류가 발생했습니다.")

@app.get("/api/user/{user_id}/profile", response_model=Dict[str, Any])
async def get_user_profile(
    user_id: str,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """사용자 프로필 조회"""
    try:
        if user_id not in rec.user_profiles:
            raise HTTPException(status_code=404, detail="사용자 프로필이 존재하지 않습니다.")
        
        profile = rec.user_profiles[user_id]
        
        return {
            "user_id": profile['user_id'],
            "watched_count": len(profile['watched_anime']),
            "avg_rating": profile['preferences']['avg_rating'],
            "preferences": {
                "genres": profile['preferences']['genre_preferences'],
                "tags": profile['preferences']['tag_preferences'],
                "years": profile['preferences']['preferred_years'],
                "formats": profile['preferences']['preferred_formats']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"사용자 프로필 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="프로필 조회 중 오류가 발생했습니다.")

@app.get("/api/stats", response_model=Dict[str, Any])
async def get_stats(rec: AnimeRecommendationSystem = Depends(get_recommender)):
    """시스템 통계"""
    try:
        stats = {
            "total_anime": len(rec.anime_data),
            "total_users": len(rec.user_profiles),
            "unique_genres": len(set(
                genre.strip() 
                for genres in rec.anime_data['genres'].dropna() 
                for genre in str(genres).split('|') 
                if genre.strip()
            )),
            "year_range": [
                int(rec.anime_data['year'].min()),
                int(rec.anime_data['year'].max())
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"통계 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="통계 조회 중 오류가 발생했습니다.")

if __name__ == '__main__':
    print("🎌 애니메이션 추천 API 서버 시작 중... (FastAPI)")
    print("🚀 API 서버 실행 중... (http://localhost:8000)")
    print("\n📖 사용 가능한 엔드포인트:")
    print("  GET  /health - 서버 상태 확인")
    print("  GET  /api/anime/search?q=검색어 - 애니메이션 검색")
    print("  POST /api/user/profile - 사용자 프로필 생성")
    print("  POST /api/recommend/content - 콘텐츠 기반 추천")
    print("  POST /api/recommend/collaborative - 협업 필터링 추천")
    print("  POST /api/recommend/hybrid - 하이브리드 추천")
    print("  GET  /api/trending - 트렌딩 애니메이션")
    print("  GET  /api/user/{user_id}/profile - 사용자 프로필 조회")
    print("  GET  /api/stats - 시스템 통계")
    print("\n📚 API 문서: http://localhost:8000/docs")
    print("📋 ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")