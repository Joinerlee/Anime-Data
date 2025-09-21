# -*- coding: utf-8 -*-
"""
하이브리드 추천 시스템 검증 스크립트 (Windows 호환)
"""

import sys
import os

def check_code_syntax():
    """코드 구문 검증"""
    print("코드 구문 검증 중...")

    files_to_check = [
        "anime_recommender.py",
        "api_server.py"
    ]

    for file_name in files_to_check:
        try:
            print(f"   {file_name} 구문 검사...")

            # 파일 읽기
            with open(file_name, 'r', encoding='utf-8') as f:
                code = f.read()

            # 구문 검사
            compile(code, file_name, 'exec')
            print(f"   OK {file_name}: 구문 정상")

        except SyntaxError as e:
            print(f"   ERROR {file_name}: 구문 오류 - {e}")
            return False
        except FileNotFoundError:
            print(f"   WARNING {file_name}: 파일 없음")
        except Exception as e:
            print(f"   ERROR {file_name}: 오류 - {e}")
            return False

    return True

def analyze_hybrid_recommendation_logic():
    """하이브리드 추천 로직 분석"""
    print("\n하이브리드 추천 로직 분석 중...")

    try:
        # anime_recommender.py 파일 읽기
        with open("anime_recommender.py", 'r', encoding='utf-8') as f:
            code = f.read()

        # 주요 함수들 확인
        functions_to_check = [
            "content_based_recommend",
            "item_based_collaborative_recommend",
            "hybrid_recommend",
            "_apply_diversity_filter",
            "_calculate_genre_similarity",
            "_calculate_preference_score",
            "_get_detailed_scores",
            "_generate_recommendation_reason"
        ]

        for func_name in functions_to_check:
            if f"def {func_name}" in code:
                print(f"   OK {func_name} 함수 존재")
            else:
                print(f"   ERROR {func_name} 함수 없음")
                return False

        # 개선 사항 확인
        improvements = [
            ("점수 정규화", "content_max = max"),
            ("다양성 필터", "_apply_diversity_filter"),
            ("방법론 구분", "recommendation_method"),
            ("하이브리드 보너스", "method == 'hybrid'"),
            ("장르 다양성", "diversity_bonus"),
            ("선호도 기반 점수", "genre_prefs"),
            ("장르 유사성", "genre_similarity"),
            ("추천 이유", "recommendation_reason")
        ]

        print("\n   개선 사항 확인:")
        for improvement, pattern in improvements:
            if pattern in code:
                print(f"   OK {improvement}: 구현됨")
            else:
                print(f"   WARNING {improvement}: 패턴 '{pattern}' 확인 필요")

        return True

    except Exception as e:
        print(f"   ERROR 분석 실패: {e}")
        return False

def check_api_integration():
    """API 통합 검증"""
    print("\nAPI 통합 검증 중...")

    try:
        with open("api_server.py", 'r', encoding='utf-8') as f:
            api_code = f.read()

        # 새로운 엔드포인트 확인
        endpoints = [
            "trigger-batch-update",
            "batch-complete",
            "/api/animations",
            "trigger-global-update",
            "recommendations/{user_id}"
        ]

        for endpoint in endpoints:
            if endpoint in api_code:
                print(f"   OK {endpoint} 엔드포인트 존재")
            else:
                print(f"   ERROR {endpoint} 엔드포인트 없음")
                return False

        # 백그라운드 작업 로직 확인
        bg_features = [
            ("배치 처리", "process_batch_recommendations"),
            ("Redis 저장", "redis_client.setex"),
            ("Spring 콜백", "send_callback_to_spring"),
            ("에러 처리", "except Exception"),
            ("로깅", "logger.info"),
            ("데이터 검증", "if not user_update.liked_anime_ids")
        ]

        print("\n   백그라운드 작업 기능:")
        for feature, pattern in bg_features:
            if pattern in api_code:
                print(f"   OK {feature}: 구현됨")
            else:
                print(f"   WARNING {feature}: 확인 필요")

        return True

    except Exception as e:
        print(f"   ERROR API 검증 실패: {e}")
        return False

def validate_data_models():
    """데이터 모델 검증"""
    print("\n데이터 모델 검증 중...")

    try:
        with open("api_server.py", 'r', encoding='utf-8') as f:
            api_code = f.read()

        # Pydantic 모델 확인
        models = [
            "UserLikesDislikesUpdate",
            "BatchUpdateRequest",
            "BatchCompleteRequest",
            "NewAnimeRequest",
            "RecommendationResponse"
        ]

        for model in models:
            if f"class {model}" in api_code:
                print(f"   OK {model} 모델 정의됨")
            else:
                print(f"   ERROR {model} 모델 없음")
                return False

        # 새로운 필드 확인
        new_fields = [
            "genre_similarity",
            "preference_score",
            "anime_genres",
            "user_top_genres",
            "matched_genres",
            "recommendation_reason",
            "recommendation_method"
        ]

        print("\n   새로운 응답 필드 확인:")
        for field in new_fields:
            if field in api_code:
                print(f"   OK {field}: 존재")
            else:
                print(f"   WARNING {field}: 확인 필요")

        return True

    except Exception as e:
        print(f"   ERROR 모델 검증 실패: {e}")
        return False

def check_requirements():
    """의존성 확인"""
    print("\n의존성 확인 중...")

    try:
        with open("requirements.txt", 'r', encoding='utf-8') as f:
            requirements = f.read()

        new_deps = [
            "redis",
            "requests"
        ]

        for dep in new_deps:
            if dep in requirements:
                print(f"   OK {dep}: requirements.txt에 추가됨")
            else:
                print(f"   ERROR {dep}: requirements.txt에 없음")
                return False

        return True

    except Exception as e:
        print(f"   ERROR 의존성 확인 실패: {e}")
        return False

if __name__ == "__main__":
    print("애니메이션 하이브리드 추천 시스템 검증")
    print("=" * 60)

    # 전체 검증 수행
    syntax_ok = check_code_syntax()
    logic_ok = analyze_hybrid_recommendation_logic()
    api_ok = check_api_integration()
    model_ok = validate_data_models()
    deps_ok = check_requirements()

    print("\n" + "=" * 60)
    print("검증 결과 요약:")
    print(f"   코드 구문: {'OK' if syntax_ok else 'FAIL'}")
    print(f"   추천 로직: {'OK' if logic_ok else 'FAIL'}")
    print(f"   API 통합: {'OK' if api_ok else 'FAIL'}")
    print(f"   데이터 모델: {'OK' if model_ok else 'FAIL'}")
    print(f"   의존성: {'OK' if deps_ok else 'FAIL'}")

    print("\n새로운 기능 요약:")
    print("   - 장르 유사성 점수 (genre_similarity)")
    print("   - 종합 선호도 점수 (preference_score)")
    print("   - 애니메이션 장르 목록 (anime_genres)")
    print("   - 사용자 선호 장르 (user_top_genres)")
    print("   - 일치하는 장르 (matched_genres)")
    print("   - 추천 이유 (recommendation_reason)")
    print("   - 추천 방법 구분 (recommendation_method)")

    print("\nAPI 응답 예시:")
    print("""
    {
        "id": 12345,
        "title": "나 혼자만 레벨업",
        "final_score": 0.85,
        "content_score": 0.45,
        "collab_score": 0.40,
        "genre_similarity": 0.78,
        "preference_score": 4.2,
        "anime_genres": ["액션", "판타지", "드라마"],
        "user_top_genres": ["액션", "판타지", "SF"],
        "matched_genres": ["액션", "판타지"],
        "recommendation_reason": "'액션', '판타지' 장르를 선호하시는 취향과 일치",
        "recommendation_method": "hybrid"
    }
    """)

    if all([syntax_ok, logic_ok, api_ok, model_ok, deps_ok]):
        print("\n모든 검증 통과! 하이브리드 추천 시스템이 준비되었습니다.")
        print("\n실행 방법:")
        print("   python api_server.py")
        print("   또는")
        print("   uvicorn api_server:app --host 0.0.0.0 --port 8000")
        sys.exit(0)
    else:
        print("\n일부 검증 실패. 문제를 해결해주세요.")
        sys.exit(1)