#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
애니메이션 추천 시스템 REST API 서버 (FastAPI)
"""

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from anime_recommender import AnimeRecommendationSystem
import os
import uvicorn
import redis
import json
import requests
import asyncio
from datetime import datetime
import uuid

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

# Redis 연결 설정
redis_client = None

# Spring 서버 설정
SPRING_SERVER_URL = os.getenv("SPRING_SERVER_URL", "http://localhost:8080")

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
    # 새로운 상세 정보 필드들
    genre_similarity: Optional[float] = None
    preference_score: Optional[float] = None
    anime_genres: Optional[List[str]] = None
    user_top_genres: Optional[List[str]] = None
    matched_genres: Optional[List[str]] = None
    recommendation_reason: Optional[str] = None
    recommendation_method: Optional[str] = None

# 새로운 API 모델들
class UserLikesDislikesUpdate(BaseModel):
    user_id: str
    liked_anime_ids: List[int]
    disliked_anime_ids: List[int]

class BatchUpdateRequest(BaseModel):
    updated_user_profiles: List[UserLikesDislikesUpdate]

class BatchCompleteRequest(BaseModel):
    job_id: str
    status: str
    successful_user_ids: List[str]
    failed_user_ids: List[str]

class NewAnimeRequest(BaseModel):
    title_korean: Optional[str] = None
    title_japanese: Optional[str] = None
    title_english: Optional[str] = None
    year: Optional[int] = None
    genres: List[str] = []
    tags: List[str] = []
    synopsis: Optional[str] = None
    format: Optional[str] = None
    duration: Optional[int] = None

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

def initialize_redis():
    """Redis 연결 초기화"""
    global redis_client
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))

        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
            socket_timeout=5
        )

        # 연결 테스트
        redis_client.ping()
        logger.info(f"Redis 연결 완료: {redis_host}:{redis_port}/{redis_db}")
        return True
    except Exception as e:
        logger.warning(f"Redis 연결 실패: {str(e)} - Redis 기능 비활성화")
        redis_client = None
        return False

async def send_callback_to_spring(job_id: str, status: str, successful_user_ids: List[str], failed_user_ids: List[str]):
    """Spring 서버에 완료 콜백 전송"""
    try:
        callback_data = {
            "job_id": job_id,
            "status": status,
            "successful_user_ids": successful_user_ids,
            "failed_user_ids": failed_user_ids
        }

        response = requests.post(
            f"{SPRING_SERVER_URL}/api/internal/recommendations/batch-complete",
            json=callback_data,
            timeout=10
        )

        if response.status_code == 200:
            logger.info(f"Spring 콜백 성공: {job_id}")
        else:
            logger.error(f"Spring 콜백 실패: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Spring 콜백 전송 오류: {str(e)}")

def generate_anime_combined_features(anime_data: Dict) -> str:
    """애니메이션 데이터로부터 '종합 특징 텍스트' 생성"""
    features = []

    # 제목들
    if anime_data.get("title_korean"):
        features.append(anime_data["title_korean"])
    if anime_data.get("title_japanese"):
        features.append(anime_data["title_japanese"])
    if anime_data.get("title_english"):
        features.append(anime_data["title_english"])

    # 장르
    if anime_data.get("genres"):
        if isinstance(anime_data["genres"], list):
            features.extend(anime_data["genres"])
        else:
            features.append(anime_data["genres"])

    # 태그
    if anime_data.get("tags"):
        if isinstance(anime_data["tags"], list):
            features.extend(anime_data["tags"])
        else:
            features.append(anime_data["tags"])

    # 줄거리
    if anime_data.get("synopsis"):
        features.append(anime_data["synopsis"])

    return " ".join(features)

def get_recommender():
    """의존성 주입을 위한 추천 시스템 반환"""
    if recommender is None:
        raise HTTPException(status_code=500, detail="추천 시스템이 초기화되지 않았습니다.")
    return recommender

@app.on_event("startup")
async def startup_event():
    """서버 시작시 추천 시스템 및 Redis 초기화"""
    if not initialize_recommender():
        logger.error("추천 시스템 초기화 실패")

    initialize_redis()  # Redis 연결 실패해도 계속 진행

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

# ==================== 새로운 API 엔드포인트들 ====================

async def process_batch_recommendations(job_id: str, user_updates: List[UserLikesDislikesUpdate]):
    """백그라운드에서 배치 추천 처리"""
    successful_users = []
    failed_users = []
    start_time = datetime.now()

    logger.info(f"배치 추천 작업 시작: {job_id}, 사용자 수: {len(user_updates)}")

    try:
        for i, user_update in enumerate(user_updates, 1):
            try:
                user_id = user_update.user_id
                logger.debug(f"처리 중 ({i}/{len(user_updates)}): {user_id}")

                # 데이터 검증
                if not user_update.liked_anime_ids and not user_update.disliked_anime_ids:
                    logger.warning(f"사용자 {user_id}: 좋아요/싫어요 데이터가 없음")
                    failed_users.append(user_id)
                    continue

                # 좋아요/싫어요를 기반으로 새로운 평점 생성
                watched_anime = user_update.liked_anime_ids + user_update.disliked_anime_ids
                ratings = ([5.0] * len(user_update.liked_anime_ids) +
                          [1.0] * len(user_update.disliked_anime_ids))

                logger.debug(f"사용자 {user_id}: 좋아요 {len(user_update.liked_anime_ids)}개, 싫어요 {len(user_update.disliked_anime_ids)}개")

                # 사용자 프로필 업데이트
                profile = recommender.create_user_profile(user_id, watched_anime, ratings)

                # 하이브리드 추천 생성
                recommendations = recommender.hybrid_recommend(user_id, n_recommendations=12)

                if not recommendations:
                    logger.warning(f"사용자 {user_id}: 추천 결과가 없음")
                    failed_users.append(user_id)
                    continue

                # 추천 결과 검증
                rec_count = len(recommendations)
                avg_score = sum(rec.get('final_score', 0) for rec in recommendations) / rec_count

                logger.debug(f"사용자 {user_id}: {rec_count}개 추천 생성, 평균 점수: {avg_score:.4f}")

                # Redis에 추천 목록 저장
                if redis_client:
                    redis_key = f"recommendations:{user_id}"

                    # JSON 직렬화 가능하도록 float 변환
                    serializable_recs = []
                    for rec in recommendations:
                        serializable_rec = rec.copy()
                        for key in ['final_score', 'content_score', 'collab_score', 'similarity_score']:
                            if key in serializable_rec:
                                serializable_rec[key] = float(serializable_rec[key])
                        serializable_recs.append(serializable_rec)

                    redis_client.setex(
                        redis_key,
                        86400,  # 24시간 유효
                        json.dumps(serializable_recs, ensure_ascii=False)
                    )

                    logger.debug(f"사용자 {user_id}: Redis 저장 완료")
                else:
                    logger.warning(f"사용자 {user_id}: Redis 연결 없음, 메모리에만 저장")

                successful_users.append(user_id)
                logger.info(f"사용자 {user_id} 추천 생성 완료 ({i}/{len(user_updates)})")

            except Exception as e:
                logger.error(f"사용자 {user_update.user_id} 추천 생성 실패: {str(e)}")
                import traceback
                logger.debug(f"상세 오류 정보: {traceback.format_exc()}")
                failed_users.append(user_update.user_id)

        # 배치 작업 완료
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"배치 추천 작업 완료: {job_id}")
        logger.info(f"  - 성공: {len(successful_users)}명")
        logger.info(f"  - 실패: {len(failed_users)}명")
        logger.info(f"  - 소요시간: {duration:.2f}초")

        # Spring 서버에 완료 콜백 전송
        status = "completed" if successful_users else "failed"
        await send_callback_to_spring(job_id, status, successful_users, failed_users)

    except Exception as e:
        logger.error(f"배치 추천 처리 전체 오류: {str(e)}")
        import traceback
        logger.error(f"상세 오류 정보: {traceback.format_exc()}")
        await send_callback_to_spring(job_id, "failed", successful_users, failed_users)

@app.post("/api/recommendations/trigger-batch-update")
async def trigger_batch_update(
    request: BatchUpdateRequest,
    background_tasks: BackgroundTasks,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """다수의 사용자 프로필 데이터를 전달받아 배치 추천 작업 시작"""
    try:
        # 작업 ID 생성
        job_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M')}-{str(uuid.uuid4())[:8]}"

        # 백그라운드 작업 시작
        background_tasks.add_task(
            process_batch_recommendations,
            job_id,
            request.updated_user_profiles
        )

        logger.info(f"배치 추천 작업 시작: {job_id}, 사용자 수: {len(request.updated_user_profiles)}")

        return {
            "message": "배치 추천 작업이 시작되었습니다.",
            "job_id": job_id,
            "user_count": len(request.updated_user_profiles),
            "status": "started"
        }

    except Exception as e:
        logger.error(f"배치 추천 작업 시작 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="배치 작업 시작 중 오류가 발생했습니다.")

@app.post("/api/internal/recommendations/batch-complete")
async def batch_complete_callback(request: BatchCompleteRequest):
    """FastAPI가 Spring 서버로부터 받는 내부 콜백 (테스트용)"""
    try:
        logger.info(f"배치 완료 콜백 수신: {request.job_id}, 상태: {request.status}")
        logger.info(f"성공: {len(request.successful_user_ids)}명, 실패: {len(request.failed_user_ids)}명")

        return {
            "message": "콜백 수신 완료",
            "job_id": request.job_id,
            "received_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"콜백 처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="콜백 처리 중 오류가 발생했습니다.")

@app.post("/api/animations")
async def add_new_animation(
    request: NewAnimeRequest,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """새로운 애니메이션 추가 및 임베딩"""
    try:
        # 새 ID 생성 (기존 최대 ID + 1)
        new_id = int(rec.anime_data['id'].max()) + 1

        # 애니메이션 데이터 생성
        anime_data = {
            "id": new_id,
            "title_korean": request.title_korean or "",
            "title_japanese": request.title_japanese or "",
            "title_english": request.title_english or "",
            "year": request.year,
            "genres": "|".join(request.genres) if request.genres else "",
            "tags": "|".join(request.tags) if request.tags else "",
            "synopsis": request.synopsis or "",
            "format": request.format or "",
            "duration": request.duration
        }

        # 종합 특징 텍스트 생성
        combined_features = generate_anime_combined_features({
            "title_korean": anime_data["title_korean"],
            "title_japanese": anime_data["title_japanese"],
            "title_english": anime_data["title_english"],
            "genres": request.genres,
            "tags": request.tags,
            "synopsis": anime_data["synopsis"]
        })

        anime_data["combined_features"] = combined_features

        # 데이터프레임에 추가
        import pandas as pd
        new_row = pd.DataFrame([anime_data])
        rec.anime_data = pd.concat([rec.anime_data, new_row], ignore_index=True)

        # 새로운 애니메이션의 임베딩 생성
        new_embedding = rec.embedding_model.encode_texts([combined_features])

        # 기존 특성 매트릭스에 추가
        import numpy as np
        rec.content_features = np.vstack([rec.content_features, new_embedding])

        # 유사도 매트릭스 재계산 (전체)
        rec.content_similarity_matrix = rec.embedding_model.compute_similarity(rec.content_features)

        logger.info(f"새 애니메이션 추가 완료: ID {new_id}, 제목: {request.title_korean or request.title_japanese or request.title_english}")

        return {
            "message": "애니메이션이 성공적으로 추가되었습니다.",
            "anime_id": new_id,
            "title": request.title_korean or request.title_japanese or request.title_english,
            "embedding_model": "kanana" if rec.use_kanana else "tfidf",
            "total_anime_count": len(rec.anime_data)
        }

    except Exception as e:
        logger.error(f"애니메이션 추가 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="애니메이션 추가 중 오류가 발생했습니다.")

async def process_global_update():
    """모든 사용자의 추천 목록 전체 갱신"""
    try:
        updated_users = []
        failed_users = []

        for user_id in recommender.user_profiles.keys():
            try:
                # 하이브리드 추천 재생성
                recommendations = recommender.hybrid_recommend(user_id, n_recommendations=12)

                # Redis에 저장
                if redis_client:
                    redis_key = f"recommendations:{user_id}"
                    redis_client.setex(
                        redis_key,
                        86400,  # 24시간 유효
                        json.dumps(recommendations, ensure_ascii=False)
                    )

                updated_users.append(user_id)

            except Exception as e:
                logger.error(f"사용자 {user_id} 글로벌 갱신 실패: {str(e)}")
                failed_users.append(user_id)

        logger.info(f"전체 추천 갱신 완료: 성공 {len(updated_users)}명, 실패 {len(failed_users)}명")

        # Spring 서버에 완료 알림
        job_id = f"global-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        await send_callback_to_spring(job_id, "completed", updated_users, failed_users)

    except Exception as e:
        logger.error(f"전체 추천 갱신 오류: {str(e)}")

@app.post("/api/recommendations/trigger-global-update")
async def trigger_global_update(
    background_tasks: BackgroundTasks,
    rec: AnimeRecommendationSystem = Depends(get_recommender)
):
    """모든 사용자의 추천 목록 갱신 트리거"""
    try:
        # 백그라운드 작업으로 전체 갱신 시작
        background_tasks.add_task(process_global_update)

        user_count = len(rec.user_profiles)
        logger.info(f"전체 추천 갱신 작업 시작: 총 {user_count}명의 사용자")

        return {
            "message": "전체 추천 갱신 작업이 시작되었습니다.",
            "total_users": user_count,
            "status": "started",
            "estimated_duration": "수 분 소요 예상"
        }

    except Exception as e:
        logger.error(f"전체 갱신 작업 시작 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="전체 갱신 작업 시작 중 오류가 발생했습니다.")

@app.get("/api/recommendations/{user_id}")
async def get_user_recommendations(user_id: str):
    """Redis에서 사용자별 추천 목록 조회"""
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis 서비스를 사용할 수 없습니다.")

        redis_key = f"recommendations:{user_id}"
        recommendations_json = redis_client.get(redis_key)

        if not recommendations_json:
            raise HTTPException(status_code=404, detail="해당 사용자의 추천 목록을 찾을 수 없습니다.")

        recommendations = json.loads(recommendations_json)

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "cached_at": "Redis에서 조회됨"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"추천 목록 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="추천 목록 조회 중 오류가 발생했습니다.")

if __name__ == '__main__':
    print("Starting Anime Recommendation API Server... (FastAPI)")
    print("API Server running... (http://localhost:8000)")
    print("\nBasic endpoints:")
    print("  GET  /health - Server status")
    print("  GET  /api/anime/search?q=query - Search anime")
    print("  POST /api/user/profile - 사용자 프로필 생성")
    print("  POST /api/recommend/content - 콘텐츠 기반 추천")
    print("  POST /api/recommend/collaborative - 협업 필터링 추천")
    print("  POST /api/recommend/hybrid - 하이브리드 추천")
    print("  GET  /api/trending - 트렌딩 애니메이션")
    print("  GET  /api/user/{user_id}/profile - 사용자 프로필 조회")
    print("  GET  /api/stats - 시스템 통계")
    print("\n🆕 새로운 엔드포인트:")
    print("  POST /api/recommendations/trigger-batch-update - 배치 추천 작업 시작")
    print("  POST /api/internal/recommendations/batch-complete - 배치 완료 콜백")
    print("  POST /api/animations - 신규 애니메이션 추가")
    print("  POST /api/recommendations/trigger-global-update - 전체 추천 갱신")
    print("  GET  /api/recommendations/{user_id} - 사용자별 추천 목록 조회")
    print("\n📚 API 문서: http://localhost:8000/docs")
    print("📋 ReDoc: http://localhost:8000/redoc")
    print("\n⚙️ 환경 변수:")
    print("  REDIS_HOST, REDIS_PORT, REDIS_DB - Redis 설정")
    print("  SPRING_SERVER_URL - Spring 서버 콜백 URL")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")