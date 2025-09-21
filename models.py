"""
데이터베이스 모델 정의 (PostgreSQL + pgvector)
Spring 서버와 연동되는 추천 및 검색 전용 데이터 모델
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, Index, func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from database import Base
import uuid
from datetime import datetime
from typing import List, Optional

class Animation(Base):
    """
    애니메이션 정보 모델 (Spring 서버에서 동기화)
    """
    __tablename__ = "animations"

    # 기본 정보
    id = Column(Integer, primary_key=True, index=True)  # Spring DB와 동일한 ID
    title_korean = Column(String(500), nullable=False, index=True)
    title_japanese = Column(String(500), nullable=True)
    title_english = Column(String(500), nullable=True)

    # 메타데이터
    genres = Column(Text, nullable=True)  # 파이프(|) 구분
    tags = Column(Text, nullable=True)    # 파이프(|) 구분
    synopsis = Column(Text, nullable=True)
    year = Column(Integer, nullable=True, index=True)
    director = Column(String(200), nullable=True)
    studio = Column(String(200), nullable=True)

    # 추천 시스템용 필드
    popularity_score = Column(Float, default=0.0)  # 인기도 점수
    average_rating = Column(Float, default=0.0)    # 평균 평점
    total_ratings = Column(Integer, default=0)     # 총 평점 수

    # 임베딩 벡터 (pgvector)
    content_embedding = Column(Vector(768), nullable=True)  # Kanana 임베딩
    genre_embedding = Column(Vector(256), nullable=True)    # 장르 임베딩
    synopsis_embedding = Column(Vector(768), nullable=True)  # 줄거리 임베딩

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_sync_at = Column(DateTime, nullable=True)  # Spring 서버와 마지막 동기화 시간

    # 인덱스
    __table_args__ = (
        Index('idx_title_korean_trgm', 'title_korean', postgresql_using='gin'),
        Index('idx_genres_trgm', 'genres', postgresql_using='gin'),
        Index('idx_year_popularity', 'year', 'popularity_score'),
        Index('idx_content_embedding_cosine', 'content_embedding', postgresql_using='ivfflat', postgresql_ops={'content_embedding': 'vector_cosine_ops'}),
        Index('idx_synopsis_embedding_cosine', 'synopsis_embedding', postgresql_using='ivfflat', postgresql_ops={'synopsis_embedding': 'vector_cosine_ops'}),
    )

    def to_dict(self):
        """모델을 딕셔너리로 변환"""
        return {
            'id': self.id,
            'title_korean': self.title_korean,
            'title_japanese': self.title_japanese,
            'title_english': self.title_english,
            'genres': self.genres.split('|') if self.genres else [],
            'tags': self.tags.split('|') if self.tags else [],
            'synopsis': self.synopsis,
            'year': self.year,
            'director': self.director,
            'studio': self.studio,
            'popularity_score': self.popularity_score,
            'average_rating': self.average_rating,
            'total_ratings': self.total_ratings,
        }

class UserPreference(Base):
    """
    사용자 선호도 정보 (Spring 서버에서 받은 liked/disliked 데이터)
    """
    __tablename__ = "user_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False, index=True)  # Spring의 사용자 ID

    # 선호도 데이터
    liked_anime_ids = Column(JSON, default=list)      # 좋아하는 애니메이션 ID 목록
    disliked_anime_ids = Column(JSON, default=list)   # 싫어하는 애니메이션 ID 목록

    # 분석된 선호도 프로필
    preferred_genres = Column(JSON, default=dict)     # 장르별 선호도 점수
    preferred_tags = Column(JSON, default=dict)       # 태그별 선호도 점수
    preferred_years = Column(JSON, default=dict)      # 연도대별 선호도

    # 사용자 프로필 벡터
    profile_embedding = Column(Vector(768), nullable=True)  # 사용자 취향 임베딩

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_sync_at = Column(DateTime, default=datetime.utcnow)  # Spring에서 마지막 업데이트 시간

    # 인덱스
    __table_args__ = (
        Index('idx_user_id_updated', 'user_id', 'updated_at'),
        Index('idx_profile_embedding_cosine', 'profile_embedding', postgresql_using='ivfflat', postgresql_ops={'profile_embedding': 'vector_cosine_ops'}),
    )

class Recommendation(Base):
    """
    생성된 추천 결과 캐시 (Redis 보조용)
    """
    __tablename__ = "recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False, index=True)
    anime_id = Column(Integer, nullable=False, index=True)

    # 추천 점수 상세
    final_score = Column(Float, nullable=False)
    content_score = Column(Float, default=0.0)
    collaborative_score = Column(Float, default=0.0)
    popularity_bonus = Column(Float, default=0.0)

    # 추천 상세 정보
    recommendation_method = Column(String(50), default='hybrid')  # content/collaborative/hybrid
    recommendation_reason = Column(Text, nullable=True)
    genre_similarity = Column(Float, default=0.0)
    preference_score = Column(Float, default=0.0)

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # 캐시 만료 시간

    # 인덱스
    __table_args__ = (
        Index('idx_user_score', 'user_id', 'final_score'),
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_expires_at', 'expires_at'),
    )

class SearchLog(Base):
    """
    검색 로그 (분석용)
    """
    __tablename__ = "search_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=True, index=True)  # 비로그인 사용자는 None

    # 검색 정보
    query = Column(String(500), nullable=False)
    search_type = Column(String(50), default='text')  # text/semantic/hybrid
    results_count = Column(Integer, default=0)

    # 검색 결과
    top_results = Column(JSON, default=list)  # 상위 결과 ID 목록

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Integer, nullable=True)  # 응답 시간

    # 인덱스
    __table_args__ = (
        Index('idx_query_created', 'query', 'created_at'),
        Index('idx_user_created', 'user_id', 'created_at'),
    )

class SyncStatus(Base):
    """
    Spring 서버와의 동기화 상태 관리
    """
    __tablename__ = "sync_status"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sync_type = Column(String(50), nullable=False, unique=True)  # 'animations', 'user_preferences'

    # 동기화 정보
    last_sync_at = Column(DateTime, nullable=True)
    last_successful_sync = Column(DateTime, nullable=True)
    total_records = Column(Integer, default=0)
    sync_errors = Column(JSON, default=list)

    # 상태
    is_syncing = Column(String(20), default='idle')  # idle/running/error
    sync_version = Column(String(50), nullable=True)  # Spring 서버의 데이터 버전

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 유틸리티 함수들

def create_sample_data():
    """
    개발용 샘플 데이터 생성
    """
    from database import SyncSessionLocal

    session = SyncSessionLocal()
    try:
        # 샘플 애니메이션 데이터
        sample_anime = Animation(
            id=1,
            title_korean="원피스",
            title_japanese="ONE PIECE",
            title_english="One Piece",
            genres="액션|어드벤처|드라마",
            tags="해적|우정|모험|전투",
            synopsis="고무고무 열매를 먹고 고무인간이 된 루피가 해적왕을 꿈꾸며...",
            year=1999,
            director="오다 에이치로",
            studio="토에이 애니메이션",
            popularity_score=9.5,
            average_rating=4.8,
            total_ratings=50000
        )

        session.add(sample_anime)

        # 동기화 상태 초기화
        animation_sync = SyncStatus(
            sync_type='animations',
            total_records=1,
            is_syncing='idle'
        )

        user_sync = SyncStatus(
            sync_type='user_preferences',
            total_records=0,
            is_syncing='idle'
        )

        session.add(animation_sync)
        session.add(user_sync)

        session.commit()
        print("샘플 데이터 생성 완료")

    except Exception as e:
        session.rollback()
        print(f"샘플 데이터 생성 실패: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    """
    모델 테스트 및 샘플 데이터 생성
    """
    from database import init_database

    print("데이터베이스 모델을 테스트하는 중...")

    # 데이터베이스 초기화
    init_database()

    # 샘플 데이터 생성
    create_sample_data()

    print("모델 테스트 완료!")