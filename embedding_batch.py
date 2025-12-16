"""
배치 임베딩 생성 스크립트
- CPU에서 Kanana 모델 실행
- 새벽 시간에 스케줄 실행
- 임베딩 결과를 PostgreSQL에 저장
"""
import argparse
import logging
from datetime import datetime
from typing import List, Optional
import numpy as np
from config import settings
from database import SyncSessionLocal, sync_engine
from models import Animation
import torch
from transformers import AutoTokenizer, AutoModel
from sqlalchemy import select, text

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KananaEmbeddingModel:
    """
    Kanana 임베딩 모델 (CPU 전용)
    anime_recommender.py의 KananaEmbeddingModel을 참고하여 배치 처리에 최적화
    """
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name or settings.KANANA_MODEL
        self.device = torch.device(device)

        logger.info(f"Loading Kanana embedding model... ({self.model_name})")
        logger.info(f"Device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Kanana model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Kanana model: {e}")
            raise

    def encode_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기

        Returns:
            numpy array of embeddings (shape: [len(texts), 768])
        """
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # 진행률 로깅
                if i % (batch_size * 10) == 0:
                    logger.info(f"Processing batch {i // batch_size + 1} / {len(texts) // batch_size + 1}")

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


class BatchEmbeddingGenerator:
    """
    배치 임베딩 생성 클래스
    - 임베딩 없는 애니메이션 조회
    - 배치로 임베딩 생성
    - PostgreSQL에 저장
    """
    def __init__(self, device: str = None):
        # Kanana 모델 로드 (CPU 강제)
        device = device or settings.KANANA_DEVICE
        logger.info(f"Initializing BatchEmbeddingGenerator with device: {device}")

        self.embedding_model = KananaEmbeddingModel(device=device)
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

    def get_animes_without_embedding(self, session) -> List[Animation]:
        """
        임베딩 없는 애니메이션 조회 (content_embedding IS NULL)

        Args:
            session: SQLAlchemy session

        Returns:
            List of Animation objects without embeddings
        """
        try:
            stmt = select(Animation).where(Animation.content_embedding.is_(None))
            result = session.execute(stmt)
            animes = result.scalars().all()
            logger.info(f"Found {len(animes)} animations without embeddings")
            return animes
        except Exception as e:
            logger.error(f"Error fetching animations without embedding: {e}")
            raise

    def get_all_animes(self, session) -> List[Animation]:
        """
        모든 애니메이션 조회

        Args:
            session: SQLAlchemy session

        Returns:
            List of all Animation objects
        """
        try:
            stmt = select(Animation)
            result = session.execute(stmt)
            animes = result.scalars().all()
            logger.info(f"Found {len(animes)} total animations")
            return animes
        except Exception as e:
            logger.error(f"Error fetching all animations: {e}")
            raise

    def prepare_text_for_embedding(self, anime: Animation) -> str:
        """
        애니메이션 객체를 임베딩용 텍스트로 변환
        anime_recommender.py의 combined_features와 동일한 방식

        Args:
            anime: Animation object

        Returns:
            Combined text for embedding
        """
        combined_text = " ".join([
            anime.title_korean or "",
            anime.title_japanese or "",
            anime.title_english or "",
            anime.genres or "",
            anime.tags or "",
            anime.synopsis or ""
        ])
        return combined_text

    def generate_missing_embeddings(self):
        """
        임베딩 없는 애니메이션 처리
        1. 임베딩 없는 애니 조회
        2. 배치로 임베딩 생성 (batch_size from config)
        3. pgvector 형식으로 DB 저장
        4. 진행률 로깅
        """
        session = SyncSessionLocal()
        try:
            # 1. 임베딩 없는 애니메이션 조회
            animes = self.get_animes_without_embedding(session)

            if not animes:
                logger.info("No animations without embeddings found")
                return

            logger.info(f"Starting embedding generation for {len(animes)} animations")

            # 2. 배치로 임베딩 생성
            total_processed = 0
            total_animes = len(animes)

            for i in range(0, total_animes, self.batch_size):
                batch_animes = animes[i:i + self.batch_size]

                # 텍스트 준비
                texts = [self.prepare_text_for_embedding(anime) for anime in batch_animes]

                # 임베딩 생성
                logger.info(f"Generating embeddings for batch {i // self.batch_size + 1} / {(total_animes + self.batch_size - 1) // self.batch_size}")
                embeddings = self.embedding_model.encode_texts(texts, batch_size=self.batch_size)

                # 3. DB에 저장
                for anime, embedding in zip(batch_animes, embeddings):
                    self.save_embedding(anime.id, embedding, session)
                    total_processed += 1

                # 4. 진행률 로깅
                progress = (total_processed / total_animes) * 100
                logger.info(f"Progress: {total_processed}/{total_animes} ({progress:.1f}%)")

                # 배치마다 커밋
                session.commit()

            logger.info(f"Successfully generated embeddings for {total_processed} animations")

        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def update_all_embeddings(self):
        """
        전체 임베딩 재생성 (모델 변경 시)
        """
        session = SyncSessionLocal()
        try:
            # 모든 애니메이션 조회
            animes = self.get_all_animes(session)

            if not animes:
                logger.warning("No animations found in database")
                return

            logger.info(f"Starting full embedding regeneration for {len(animes)} animations")

            # 배치로 임베딩 생성
            total_processed = 0
            total_animes = len(animes)

            for i in range(0, total_animes, self.batch_size):
                batch_animes = animes[i:i + self.batch_size]

                # 텍스트 준비
                texts = [self.prepare_text_for_embedding(anime) for anime in batch_animes]

                # 임베딩 생성
                logger.info(f"Regenerating embeddings for batch {i // self.batch_size + 1} / {(total_animes + self.batch_size - 1) // self.batch_size}")
                embeddings = self.embedding_model.encode_texts(texts, batch_size=self.batch_size)

                # DB에 저장
                for anime, embedding in zip(batch_animes, embeddings):
                    self.save_embedding(anime.id, embedding, session)
                    total_processed += 1

                # 진행률 로깅
                progress = (total_processed / total_animes) * 100
                logger.info(f"Progress: {total_processed}/{total_animes} ({progress:.1f}%)")

                # 배치마다 커밋
                session.commit()

            logger.info(f"Successfully regenerated embeddings for {total_processed} animations")

        except Exception as e:
            logger.error(f"Error during full embedding regeneration: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def save_embedding(self, anime_id: int, embedding: np.ndarray, session):
        """
        단일 임베딩 저장 + embedding_updated_at 업데이트

        Args:
            anime_id: Animation ID
            embedding: numpy array of embedding
            session: SQLAlchemy session
        """
        try:
            # 애니메이션 조회
            stmt = select(Animation).where(Animation.id == anime_id)
            result = session.execute(stmt)
            anime = result.scalar_one_or_none()

            if not anime:
                logger.warning(f"Animation with id {anime_id} not found")
                return

            # 임베딩 저장 (pgvector 형식)
            # numpy array를 list로 변환 (pgvector가 자동으로 처리)
            anime.content_embedding = embedding.tolist()
            anime.embedding_updated_at = datetime.utcnow()

            logger.debug(f"Saved embedding for anime_id={anime_id}")

        except Exception as e:
            logger.error(f"Error saving embedding for anime_id={anime_id}: {e}")
            raise


def main():
    """
    CLI 메인 함수
    """
    parser = argparse.ArgumentParser(
        description='배치 임베딩 생성 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 누락된 임베딩만 생성 (기본)
  python embedding_batch.py --missing

  # 전체 임베딩 재생성
  python embedding_batch.py --all

  # GPU 사용 (CUDA 사용 가능 시)
  python embedding_batch.py --missing --device cuda
        """
    )
    parser.add_argument(
        '--missing',
        action='store_true',
        help='누락된 임베딩만 생성 (기본 동작)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='전체 임베딩 재생성 (모델 변경 시 사용)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='사용할 디바이스 (기본: config에서 로드)'
    )

    args = parser.parse_args()

    # 인자가 없으면 --missing을 기본으로 설정
    if not args.missing and not args.all:
        args.missing = True

    try:
        # 시작 시간 기록
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Batch Embedding Generation Started")
        logger.info(f"Start Time: {start_time}")
        logger.info("=" * 60)

        # 생성기 초기화
        generator = BatchEmbeddingGenerator(device=args.device)

        # 실행
        if args.all:
            logger.info("Mode: Full regeneration")
            generator.update_all_embeddings()
        else:
            logger.info("Mode: Missing embeddings only")
            generator.generate_missing_embeddings()

        # 종료 시간 기록
        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 60)
        logger.info("Batch Embedding Generation Completed")
        logger.info(f"End Time: {end_time}")
        logger.info(f"Duration: {duration}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
