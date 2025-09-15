#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
애니메이션 추천 시스템 데모
"""

from anime_recommender import AnimeRecommendationSystem
import pandas as pd
import json

def main():
    print("=" * 60)
    print("🎌 애니메이션 AI 추천 시스템 데모 🎌")
    print("=" * 60)
    
    # 추천 시스템 초기화
    recommender = AnimeRecommendationSystem()
    
    # 데이터 로드
    csv_path = "anilife_data_20250915_214030.csv"
    recommender.load_data(csv_path)
    
    # 콘텐츠 기반 특성 구축
    recommender.build_content_features()
    
    print("\n🔍 시스템 초기화 완료!")
    print(f"📊 총 {len(recommender.anime_data)}개의 애니메이션 데이터 로드")
    
    # 샘플 유저 생성
    create_sample_users(recommender)
    
    # 데모 실행
    demo_recommendations(recommender)
    
    print("\n" + "=" * 60)
    print("🎉 데모 완료! 추천 시스템이 준비되었습니다.")
    print("=" * 60)

def create_sample_users(recommender):
    """샘플 유저 데이터 생성"""
    print("\n👥 샘플 유저 프로필 생성 중...")
    
    # 액션 애니메이션을 좋아하는 유저
    action_fan_anime = [129, 102, 116, 121, 113]  # 실제 데이터에서 액션 애니메이션 ID
    action_fan_ratings = [5.0, 4.5, 4.8, 4.2, 4.7]
    recommender.create_user_profile("action_fan", action_fan_anime, action_fan_ratings)
    
    # 로맨스/일상 애니메이션을 좋아하는 유저
    romance_fan_anime = [126, 108, 117, 124]  # 로맨스/일상 애니메이션 ID
    romance_fan_ratings = [4.8, 4.3, 4.6, 4.4]
    recommender.create_user_profile("romance_fan", romance_fan_anime, romance_fan_ratings)
    
    # SF/미스터리를 좋아하는 유저
    sf_fan_anime = [116, 118, 122, 119]  # SF/미스터리 애니메이션 ID
    sf_fan_ratings = [5.0, 4.7, 4.4, 4.1]
    recommender.create_user_profile("sf_fan", sf_fan_anime, sf_fan_ratings)
    
    print("✅ 3명의 샘플 유저 프로필 생성 완료")

def demo_recommendations(recommender):
    """추천 데모 실행"""
    users = ["action_fan", "romance_fan", "sf_fan"]
    user_names = ["액션 매니아", "로맨스 팬", "SF 애호가"]
    
    for user_id, user_name in zip(users, user_names):
        print(f"\n{'='*50}")
        print(f"📺 {user_name}님을 위한 추천")
        print(f"{'='*50}")
        
        # 유저 취향 분석 결과 출력
        show_user_preferences(recommender, user_id, user_name)
        
        # 콘텐츠 기반 추천
        print(f"\n🎯 콘텐츠 기반 추천 (상위 5개):")
        content_recs = recommender.content_based_recommend(user_id, 5)
        display_recommendations(content_recs, "content")
        
        # 협업 필터링 추천
        print(f"\n🤝 협업 필터링 추천 (상위 5개):")
        collab_recs = recommender.item_based_collaborative_recommend(user_id, 5)
        display_recommendations(collab_recs, "collaborative")
        
        # 하이브리드 추천 (최종 추천)
        print(f"\n⭐ 하이브리드 최종 추천 (상위 5개):")
        hybrid_recs = recommender.hybrid_recommend(user_id, 5)
        display_recommendations(hybrid_recs, "hybrid")

def show_user_preferences(recommender, user_id, user_name):
    """유저 취향 분석 결과 출력"""
    user_profile = recommender.user_profiles[user_id]
    preferences = user_profile['preferences']
    
    print(f"\n📊 {user_name}님의 취향 분석:")
    print(f"   평균 평점: {preferences['avg_rating']:.1f}/5.0")
    
    # 선호 장르 (상위 3개)
    top_genres = list(preferences['genre_preferences'].items())[:3]
    if top_genres:
        print("   선호 장르:", ", ".join([f"{genre}({score:.1f})" for genre, score in top_genres]))
    
    # 선호 태그 (상위 3개)
    top_tags = list(preferences['tag_preferences'].items())[:3]
    if top_tags:
        print("   선호 태그:", ", ".join([f"{tag}({score:.1f})" for tag, score in top_tags[:3]]))

def display_recommendations(recommendations, rec_type):
    """추천 결과 출력"""
    if not recommendations:
        print("   추천할 애니메이션이 없습니다.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        title = rec['title'] if rec['title'] else "제목 없음"
        
        if rec_type == "hybrid":
            score_info = f"종합점수: {rec['final_score']:.3f}"
        else:
            score_info = f"유사도: {rec['similarity_score']:.3f}"
        
        print(f"   {i}. {title}")
        print(f"      {score_info}")
        
        if rec.get('genres'):
            print(f"      장르: {rec['genres']}")
        
        if rec.get('year') and str(rec['year']) != 'nan':
            print(f"      제작년도: {int(rec['year'])}")
        
        synopsis = rec.get('synopsis', '')
        if synopsis and synopsis != "등록된 줄거리가 없습니다.":
            print(f"      줄거리: {synopsis}")
        print()

def demo_trending_anime(recommender):
    """트렌딩 애니메이션 데모"""
    print("\n🔥 최신 트렌딩 애니메이션 (2020-2025):")
    trending = recommender.get_trending_anime(year_range=(2020, 2025), n_recommendations=5)
    display_recommendations(trending, "trending")

if __name__ == "__main__":
    main()