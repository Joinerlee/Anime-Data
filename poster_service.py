"""
포스터 관리 서비스
- Jikan API에서 포스터 URL 조회
- 로컬 다운로드 및 저장
- API 응답에 포스터 URL 포함
"""
import httpx
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from config import settings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal
from models import Animation

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PosterService:
    """포스터 다운로드 및 관리 서비스"""

    JIKAN_API = "https://api.jikan.moe/v4"
    RATE_LIMIT_DELAY = 0.34  # Jikan API rate limit (초당 3요청 = 0.33초, 여유 0.34초)

    def __init__(self):
        self.poster_dir = Path(settings.POSTER_DIR)
        self.poster_dir.mkdir(exist_ok=True)
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time = 0.0

    async def get_client(self) -> httpx.AsyncClient:
        """httpx AsyncClient 반환 (재사용)"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def get_local_poster_path(self, anime_id: int) -> Optional[Path]:
        """로컬 포스터 경로 반환 (존재하는 경우)"""
        for ext in ['jpg', 'png', 'webp', 'jpeg']:
            path = self.poster_dir / f"{anime_id}.{ext}"
            if path.exists():
                return path
        return None

    async def get_poster_url(self, anime_id: int, title: str, db_session: Optional[AsyncSession] = None) -> Optional[str]:
        """포스터 URL 반환 (로컬 우선, 없으면 Jikan 조회)"""
        # 1. 로컬 파일 확인
        local_path = self.get_local_poster_path(anime_id)
        if local_path:
            return f"{settings.POSTER_STATIC_URL}/{local_path.name}"

        # 2. DB에서 URL 확인
        if db_session:
            try:
                result = await db_session.execute(
                    select(Animation).where(Animation.id == anime_id)
                )
                anime = result.scalar_one_or_none()
                if anime and anime.poster_url:
                    return anime.poster_url
            except Exception as e:
                logger.warning(f"DB 조회 실패 (anime_id={anime_id}): {e}")

        # 3. Jikan API 조회
        poster_info = await self.fetch_from_jikan(title)
        if poster_info:
            return poster_info.get('large')

        return None

    async def fetch_from_jikan(self, title: str) -> Optional[Dict[str, str]]:
        """Jikan API에서 포스터 URL 조회"""
        try:
            # Rate limiting 적용
            current_time = asyncio.get_event_loop().time()
            time_since_last_request = current_time - self._last_request_time
            if time_since_last_request < self.RATE_LIMIT_DELAY:
                await asyncio.sleep(self.RATE_LIMIT_DELAY - time_since_last_request)

            client = await self.get_client()

            # Jikan API 호출
            url = f"{self.JIKAN_API}/anime"
            params = {
                "q": title,
                "limit": 1
            }

            logger.info(f"Jikan API 조회: {title}")
            response = await client.get(url, params=params)
            self._last_request_time = asyncio.get_event_loop().time()

            response.raise_for_status()
            data = response.json()

            if data.get('data') and len(data['data']) > 0:
                anime_data = data['data'][0]
                images = anime_data.get('images', {}).get('jpg', {})

                return {
                    'large': images.get('large_image_url'),
                    'small': images.get('image_url'),
                    'mal_id': anime_data.get('mal_id')
                }
            else:
                logger.warning(f"Jikan API에서 결과를 찾을 수 없음: {title}")
                return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Jikan API HTTP 오류 (title={title}): {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Jikan API 요청 오류 (title={title}): {e}")
            return None
        except Exception as e:
            logger.error(f"Jikan API 조회 중 오류 (title={title}): {e}")
            return None

    async def download_poster(self, anime_id: int, url: str, title: str = "") -> Optional[str]:
        """포스터 다운로드 및 로컬 저장"""
        try:
            # 이미 로컬에 있는지 확인
            local_path = self.get_local_poster_path(anime_id)
            if local_path:
                logger.info(f"이미 로컬에 존재: {anime_id} -> {local_path.name}")
                return str(local_path)

            # URL에서 파일 확장자 추출
            file_ext = '.jpg'
            if '.' in url:
                ext_candidate = url.split('.')[-1].split('?')[0].lower()
                if ext_candidate in ['jpg', 'jpeg', 'png', 'webp']:
                    file_ext = f'.{ext_candidate}'

            # 다운로드
            client = await self.get_client()
            logger.info(f"포스터 다운로드 중: {title} (id={anime_id})")

            response = await client.get(url)
            response.raise_for_status()

            # 저장
            save_path = self.poster_dir / f"{anime_id}{file_ext}"
            with open(save_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"포스터 저장 완료: {save_path}")
            return str(save_path)

        except httpx.HTTPStatusError as e:
            logger.error(f"포스터 다운로드 HTTP 오류 (id={anime_id}, url={url}): {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            logger.error(f"포스터 다운로드 요청 오류 (id={anime_id}): {e}")
            return None
        except Exception as e:
            logger.error(f"포스터 다운로드 중 오류 (id={anime_id}): {e}")
            return None

    async def download_missing_posters(self, db_session: Optional[AsyncSession] = None, limit: Optional[int] = None):
        """누락된 포스터 일괄 다운로드"""
        close_session = False

        try:
            # DB 세션이 없으면 새로 생성
            if db_session is None:
                db_session = AsyncSessionLocal()
                close_session = True

            # 1. DB에서 poster_local_path가 NULL인 애니 조회
            query = select(Animation).where(Animation.poster_local_path.is_(None))
            if limit:
                query = query.limit(limit)

            result = await db_session.execute(query)
            animes = result.scalars().all()

            total = len(animes)
            logger.info(f"포스터가 없는 애니메이션 {total}개 발견")

            if total == 0:
                logger.info("다운로드할 포스터가 없습니다.")
                return

            # 2. 각각 Jikan API 조회 및 다운로드
            success_count = 0
            fail_count = 0

            for idx, anime in enumerate(animes, 1):
                try:
                    # 진행률 로깅
                    logger.info(f"[{idx}/{total}] 처리 중: {anime.title_korean} (id={anime.id})")

                    # 검색에 사용할 제목 결정
                    search_title = anime.title_english or anime.title_korean

                    # Jikan API에서 포스터 정보 조회
                    poster_info = await self.fetch_from_jikan(search_title)

                    if poster_info and poster_info.get('large'):
                        poster_url = poster_info['large']

                        # 포스터 다운로드
                        local_path = await self.download_poster(
                            anime.id,
                            poster_url,
                            anime.title_korean
                        )

                        if local_path:
                            # DB 업데이트
                            anime.poster_url = poster_url
                            anime.poster_local_path = local_path
                            anime.updated_at = datetime.utcnow()
                            await db_session.commit()

                            success_count += 1
                            logger.info(f"성공: {anime.title_korean}")
                        else:
                            fail_count += 1
                            logger.warning(f"다운로드 실패: {anime.title_korean}")
                    else:
                        fail_count += 1
                        logger.warning(f"포스터 URL을 찾을 수 없음: {anime.title_korean}")

                except Exception as e:
                    fail_count += 1
                    logger.error(f"처리 중 오류 발생 (id={anime.id}): {e}")
                    await db_session.rollback()

            # 최종 결과 로깅
            logger.info(f"포스터 다운로드 완료 - 성공: {success_count}, 실패: {fail_count}, 전체: {total}")

        except Exception as e:
            logger.error(f"포스터 일괄 다운로드 중 오류: {e}")
            if db_session:
                await db_session.rollback()
            raise
        finally:
            if close_session and db_session:
                await db_session.close()

    async def update_existing_posters_in_db(self):
        """기존 로컬 포스터 파일들을 DB에 업데이트"""
        async with AsyncSessionLocal() as session:
            try:
                # 모든 애니메이션 조회
                result = await session.execute(select(Animation))
                animes = result.scalars().all()

                updated_count = 0

                for anime in animes:
                    # 로컬 포스터 파일 확인
                    local_path = self.get_local_poster_path(anime.id)

                    if local_path and not anime.poster_local_path:
                        # DB 업데이트
                        anime.poster_local_path = str(local_path)
                        anime.updated_at = datetime.utcnow()
                        updated_count += 1

                await session.commit()
                logger.info(f"기존 포스터 {updated_count}개를 DB에 업데이트했습니다.")

            except Exception as e:
                logger.error(f"포스터 DB 업데이트 중 오류: {e}")
                await session.rollback()
                raise

    async def get_stats(self) -> Dict[str, int]:
        """포스터 통계 정보 반환"""
        async with AsyncSessionLocal() as session:
            try:
                # 전체 애니메이션 수
                result = await session.execute(select(Animation))
                total_anime = len(result.scalars().all())

                # 포스터 URL이 있는 애니메이션 수
                result = await session.execute(
                    select(Animation).where(Animation.poster_url.isnot(None))
                )
                with_url = len(result.scalars().all())

                # 로컬 파일이 있는 애니메이션 수
                result = await session.execute(
                    select(Animation).where(Animation.poster_local_path.isnot(None))
                )
                with_local = len(result.scalars().all())

                # 실제 로컬 파일 개수
                local_files = len(list(self.poster_dir.glob('*.*')))

                return {
                    'total_anime': total_anime,
                    'with_poster_url': with_url,
                    'with_local_path': with_local,
                    'local_files_count': local_files,
                    'missing_posters': total_anime - with_local
                }

            except Exception as e:
                logger.error(f"통계 조회 중 오류: {e}")
                return {}

    async def close(self):
        """클라이언트 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("HTTP 클라이언트 종료")


async def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='포스터 다운로드 및 관리')
    parser.add_argument('--download-all', action='store_true', help='누락된 포스터 전체 다운로드')
    parser.add_argument('--download-limit', type=int, help='다운로드할 최대 개수')
    parser.add_argument('--update-db', action='store_true', help='기존 로컬 포스터 파일을 DB에 업데이트')
    parser.add_argument('--stats', action='store_true', help='포스터 통계 정보 출력')
    parser.add_argument('--test', type=str, help='특정 제목으로 Jikan API 테스트')

    args = parser.parse_args()

    service = PosterService()

    try:
        if args.stats:
            # 통계 출력
            stats = await service.get_stats()
            logger.info("=" * 50)
            logger.info("포스터 통계 정보")
            logger.info("=" * 50)
            logger.info(f"전체 애니메이션: {stats.get('total_anime', 0)}")
            logger.info(f"포스터 URL 보유: {stats.get('with_poster_url', 0)}")
            logger.info(f"로컬 경로 보유: {stats.get('with_local_path', 0)}")
            logger.info(f"로컬 파일 개수: {stats.get('local_files_count', 0)}")
            logger.info(f"누락된 포스터: {stats.get('missing_posters', 0)}")
            logger.info("=" * 50)

        elif args.update_db:
            # 기존 로컬 포스터를 DB에 업데이트
            logger.info("기존 로컬 포스터를 DB에 업데이트합니다...")
            await service.update_existing_posters_in_db()

        elif args.test:
            # Jikan API 테스트
            logger.info(f"Jikan API 테스트: {args.test}")
            poster_info = await service.fetch_from_jikan(args.test)
            if poster_info:
                logger.info(f"결과: {poster_info}")
            else:
                logger.warning("결과를 찾을 수 없습니다.")

        elif args.download_all:
            # 누락된 포스터 다운로드
            logger.info("누락된 포스터를 다운로드합니다...")
            await service.download_missing_posters(limit=args.download_limit)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
