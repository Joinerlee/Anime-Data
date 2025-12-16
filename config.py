"""
환경 설정 관리
- python-dotenv로 .env 파일 로드
- 모든 설정을 중앙 관리
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "anime_recommendation_db")
    DB_USERNAME = os.getenv("DB_USERNAME", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    # Kanana Model
    KANANA_MODEL = os.getenv("KANANA_MODEL", "kakaocorp/kanana-nano-2.1b-embedding")
    KANANA_DEVICE = os.getenv("KANANA_DEVICE", "cpu")

    # Batch Settings
    BATCH_EMBEDDING_HOUR = int(os.getenv("BATCH_EMBEDDING_HOUR", "3"))
    BATCH_POSTER_HOUR = int(os.getenv("BATCH_POSTER_HOUR", "4"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

    # Poster
    POSTER_DIR = os.getenv("POSTER_DIR", "./posters")
    POSTER_STATIC_URL = os.getenv("POSTER_STATIC_URL", "/static/posters")

    # Spring Server
    SPRING_SERVER_URL = os.getenv("SPRING_SERVER_URL", "http://localhost:8080")

    # Database URL
    @property
    def DATABASE_URL(self):
        return f"postgresql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def ASYNC_DATABASE_URL(self):
        return f"postgresql+asyncpg://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()
