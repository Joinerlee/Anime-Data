"""
Spring 서버 연동 모듈
- 애니메이션 데이터: Spring 플래그 신호시에만 추가/임베딩
- 유저 취향 데이터: 매일 12시 자동 업데이트
"""

import asyncio
import logging
from datetime import datetime, time
from typing import List, Dict, Any, Optional
import requests
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from database import AsyncSessionLocal, get_sync_db
from models import Animation, UserPreference, SyncStatus
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spring 서버 설정
SPRING_CONFIG = {
    "base_url": os.getenv("SPRING_SERVER_URL", "http://localhost:8080"),
    "api_key": os.getenv("SPRING_API_KEY", ""),
    "timeout": 30
}

class SpringIntegrationService:
    """Spring 서버와의 연동 서비스"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {SPRING_CONFIG['api_key']}",
            "Content-Type": "application/json"
        })

    async def process_animation_flag(self, flag_data: Dict[str, Any]) -> bool:
        """
        Spring에서 온 애니메이션 플래그 처리
        새로운 애니메이션 데이터 추가 및 임베딩 생성
        """
        try:
            logger.info(f"애니메이션 플래그 처리 시작: {flag_data}")

            # 플래그 타입 확인
            flag_type = flag_data.get("type")
            animation_ids = flag_data.get("animation_ids", [])

            if flag_type == "new_animations":
                return await self._process_new_animations(animation_ids)
            elif flag_type == "update_animations":
                return await self._process_update_animations(animation_ids)
            else:
                logger.warning(f"알 수 없는 플래그 타입: {flag_type}")
                return False

        except Exception as e:
            logger.error(f"애니메이션 플래그 처리 실패: {e}")
            return False

    async def _process_new_animations(self, animation_ids: List[int]) -> bool:
        """새로운 애니메이션 데이터 처리"""
        try:
            # Spring 서버에서 애니메이션 데이터 가져오기
            animations_data = await self._fetch_animations_from_spring(animation_ids)

            if not animations_data:
                logger.warning("Spring 서버에서 애니메이션 데이터를 가져올 수 없음")
                return False

            # 데이터베이스에 저장 및 임베딩 생성
            success_count = 0
            async with AsyncSessionLocal() as session:
                for anime_data in animations_data:
                    try:
                        # 애니메이션 모델 생성
                        animation = Animation(
                            id=anime_data["id"],
                            title_korean=anime_data.get("title_korean"),
                            title_japanese=anime_data.get("title_japanese"),
                            title_english=anime_data.get("title_english"),
                            genres="|".join(anime_data.get("genres", [])),
                            tags="|".join(anime_data.get("tags", [])),
                            synopsis=anime_data.get("synopsis"),
                            year=anime_data.get("year"),
                            director=anime_data.get("director"),
                            studio=anime_data.get("studio"),
                            popularity_score=anime_data.get("popularity_score", 0.0),
                            average_rating=anime_data.get("average_rating", 0.0),
                            total_ratings=anime_data.get("total_ratings", 0)
                        )

                        # 임베딩 생성 (여기서 Kanana 모델 온디맨드 로딩)
                        embedding = await self._generate_embedding(anime_data)
                        if embedding is not None:
                            animation.content_embedding = embedding

                        session.add(animation)
                        success_count += 1

                    except Exception as e:
                        logger.error(f"애니메이션 {anime_data.get('id')} 처리 실패: {e}")

                await session.commit()

            logger.info(f"신규 애니메이션 {success_count}/{len(animations_data)}개 처리 완료")

            # 동기화 상태 업데이트
            await self._update_sync_status("animations", success_count)

            return success_count > 0

        except Exception as e:
            logger.error(f"신규 애니메이션 처리 실패: {e}")
            return False

    async def _fetch_animations_from_spring(self, animation_ids: List[int]) -> List[Dict]:
        """Spring 서버에서 애니메이션 데이터 가져오기"""
        try:
            response = self.session.post(
                f"{SPRING_CONFIG['base_url']}/api/internal/animations/batch",
                json={"animation_ids": animation_ids},
                timeout=SPRING_CONFIG['timeout']
            )
            response.raise_for_status()
            return response.json().get("animations", [])

        except Exception as e:
            logger.error(f"Spring 서버에서 애니메이션 데이터 가져오기 실패: {e}")
            return []

    async def _generate_embedding(self, anime_data: Dict) -> Optional[List[float]]:
        """
        애니메이션 임베딩 생성 (온디맨드 Kanana 로딩)
        무거운 모델은 필요할 때만 로딩
        """
        try:
            # 임베딩 텍스트 생성
            text_parts = [
                anime_data.get("title_korean", ""),
                anime_data.get("title_japanese", ""),
                "|".join(anime_data.get("genres", [])),
                "|".join(anime_data.get("tags", [])),
                anime_data.get("synopsis", "")[:500]  # 줄거리는 500자로 제한
            ]
            combined_text = " ".join(filter(None, text_parts))

            # Kanana 모델 임베딩 생성 (온디맨드 로딩)
            from anime_recommender import KananaEmbeddingModel
            embedding_model = KananaEmbeddingModel()

            # 단일 텍스트 임베딩 생성
            embeddings = embedding_model.embed_texts([combined_text])
            if embeddings and len(embeddings) > 0:
                return embeddings[0].tolist()

            return None

        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None

    async def daily_user_preference_update(self) -> bool:
        """
        매일 12시 실행되는 유저 취향 데이터 업데이트
        Spring 서버에서 모든 사용자의 liked/disliked 데이터 가져와서 업데이트
        """
        try:
            logger.info("일일 유저 취향 업데이트 시작")

            # Spring 서버에서 업데이트된 사용자 데이터 가져오기
            user_data = await self._fetch_updated_user_preferences()

            if not user_data:
                logger.info("업데이트할 사용자 데이터가 없음")
                return True

            # 사용자 선호도 업데이트
            success_count = 0
            async with AsyncSessionLocal() as session:
                for user_pref in user_data:
                    try:
                        # 기존 사용자 선호도 찾기 또는 생성
                        result = await session.execute(
                            select(UserPreference).where(
                                UserPreference.user_id == user_pref["user_id"]
                            )
                        )
                        existing_pref = result.scalar_one_or_none()

                        if existing_pref:
                            # 기존 데이터 업데이트
                            existing_pref.liked_anime_ids = user_pref["liked_anime_ids"]
                            existing_pref.disliked_anime_ids = user_pref["disliked_anime_ids"]
                            existing_pref.updated_at = datetime.utcnow()
                            existing_pref.last_sync_at = datetime.utcnow()
                        else:
                            # 새로운 사용자 생성
                            new_pref = UserPreference(
                                user_id=user_pref["user_id"],
                                liked_anime_ids=user_pref["liked_anime_ids"],
                                disliked_anime_ids=user_pref["disliked_anime_ids"],
                                last_sync_at=datetime.utcnow()
                            )
                            session.add(new_pref)

                        success_count += 1

                    except Exception as e:
                        logger.error(f"사용자 {user_pref.get('user_id')} 처리 실패: {e}")

                await session.commit()

            logger.info(f"유저 취향 업데이트 완료: {success_count}명")

            # 동기화 상태 업데이트
            await self._update_sync_status("user_preferences", success_count)

            return True

        except Exception as e:
            logger.error(f"일일 유저 취향 업데이트 실패: {e}")
            return False

    async def _fetch_updated_user_preferences(self) -> List[Dict]:
        """Spring 서버에서 업데이트된 사용자 선호도 가져오기"""
        try:
            # 마지막 동기화 시간 확인
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(SyncStatus).where(SyncStatus.sync_type == "user_preferences")
                )
                sync_status = result.scalar_one_or_none()
                last_sync = sync_status.last_successful_sync if sync_status else None

            # Spring 서버에 요청
            params = {}
            if last_sync:
                params["since"] = last_sync.isoformat()

            response = self.session.get(
                f"{SPRING_CONFIG['base_url']}/api/internal/user-preferences/updated",
                params=params,
                timeout=SPRING_CONFIG['timeout']
            )
            response.raise_for_status()

            return response.json().get("user_preferences", [])

        except Exception as e:
            logger.error(f"사용자 선호도 데이터 가져오기 실패: {e}")
            return []

    async def _update_sync_status(self, sync_type: str, record_count: int):
        """동기화 상태 업데이트"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(SyncStatus).where(SyncStatus.sync_type == sync_type)
                )
                sync_status = result.scalar_one_or_none()

                now = datetime.utcnow()

                if sync_status:
                    sync_status.last_sync_at = now
                    sync_status.last_successful_sync = now
                    sync_status.total_records = record_count
                    sync_status.is_syncing = 'idle'
                    sync_status.updated_at = now
                else:
                    new_status = SyncStatus(
                        sync_type=sync_type,
                        last_sync_at=now,
                        last_successful_sync=now,
                        total_records=record_count,
                        is_syncing='idle'
                    )
                    session.add(new_status)

                await session.commit()

        except Exception as e:
            logger.error(f"동기화 상태 업데이트 실패: {e}")

# 글로벌 서비스 인스턴스
spring_service = SpringIntegrationService()

# 스케줄러 관련 함수들

async def schedule_daily_update():
    """매일 12시에 유저 취향 업데이트 실행"""
    while True:
        try:
            now = datetime.now()
            # 다음 12시까지의 시간 계산
            next_run = now.replace(hour=12, minute=0, second=0, microsecond=0)
            if now >= next_run:
                # 오늘 12시가 지났으면 내일 12시로
                import datetime as dt
                next_run += dt.timedelta(days=1)

            # 대기 시간 계산
            wait_seconds = (next_run - now).total_seconds()

            logger.info(f"다음 유저 취향 업데이트: {next_run} ({wait_seconds/3600:.1f}시간 후)")

            # 지정된 시간까지 대기
            await asyncio.sleep(wait_seconds)

            # 업데이트 실행
            logger.info("일일 유저 취향 업데이트 시작")
            success = await spring_service.daily_user_preference_update()

            if success:
                logger.info("일일 유저 취향 업데이트 완료")
            else:
                logger.error("일일 유저 취향 업데이트 실패")

        except Exception as e:
            logger.error(f"스케줄러 오류: {e}")
            # 오류 발생시 1시간 후 재시도
            await asyncio.sleep(3600)

def start_scheduler():
    """백그라운드에서 스케줄러 시작"""
    asyncio.create_task(schedule_daily_update())

if __name__ == "__main__":
    """테스트 실행"""
    async def test_integration():
        # 애니메이션 플래그 테스트
        test_flag = {
            "type": "new_animations",
            "animation_ids": [99999]  # 테스트용 ID
        }

        result = await spring_service.process_animation_flag(test_flag)
        print(f"애니메이션 플래그 테스트 결과: {result}")

        # 유저 취향 업데이트 테스트
        result = await spring_service.daily_user_preference_update()
        print(f"유저 취향 업데이트 테스트 결과: {result}")

    asyncio.run(test_integration())