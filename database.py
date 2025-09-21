"""
데이터베이스 설정 및 연결 관리 (PostgreSQL + pgvector)
Spring 서버의 메인 DB와 연동하여 추천 및 검색 전용 서비스
"""

import os
from typing import AsyncGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base 클래스
Base = declarative_base()

# 데이터베이스 설정
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "anime_recommendation_db"),
    "username": os.getenv("DB_USERNAME", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
}

# 동기식 연결 URL
SYNC_DATABASE_URL = (
    f"postgresql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# 비동기식 연결 URL
ASYNC_DATABASE_URL = (
    f"postgresql+asyncpg://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
)

# 엔진 생성
sync_engine = create_engine(SYNC_DATABASE_URL, echo=False)
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)

# 세션 생성
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

def init_database():
    """
    데이터베이스 초기화 (pgvector 확장 설치 및 테이블 생성)
    """
    try:
        # pgvector 확장 설치
        with sync_engine.connect() as conn:
            # pgvector 확장이 이미 설치되어 있는지 확인
            result = conn.execute(text(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            ))

            if not result.scalar():
                logger.info("pgvector 확장을 설치하는 중...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector 확장 설치 완료")
            else:
                logger.info("pgvector 확장이 이미 설치되어 있습니다")

        # 테이블 생성
        logger.info("데이터베이스 테이블을 생성하는 중...")
        Base.metadata.create_all(bind=sync_engine)
        logger.info("데이터베이스 초기화 완료")

    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {e}")
        raise

def get_sync_db():
    """
    동기식 데이터베이스 세션 생성 (컨텍스트 매니저)
    """
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    비동기식 데이터베이스 세션 생성 (FastAPI용)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def check_database_connection():
    """
    데이터베이스 연결 상태 확인
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            if result.scalar() == 1:
                logger.info("데이터베이스 연결 성공")
                return True
    except Exception as e:
        logger.error(f"데이터베이스 연결 실패: {e}")
        return False

def test_pgvector():
    """
    pgvector 기능 테스트
    """
    try:
        with sync_engine.connect() as conn:
            # 벡터 연산 테스트
            result = conn.execute(text("SELECT '[1,2,3]'::vector(3) <-> '[4,5,6]'::vector(3)"))
            distance = result.scalar()
            logger.info(f"pgvector 테스트 성공 - 거리: {distance}")
            return True
    except Exception as e:
        logger.error(f"pgvector 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    """
    데이터베이스 설정 테스트
    """
    print("데이터베이스 설정을 테스트하는 중...")

    # 데이터베이스 초기화
    init_database()

    # pgvector 테스트
    test_pgvector()

    print("데이터베이스 설정 테스트 완료!")