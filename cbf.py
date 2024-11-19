# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import random
# import pickle
# from typing import List, Optional

# # 데이터 불러오기 (맛집 데이터셋)
# file_path = "model/food_emotion.xlsx"
# df = pd.read_excel(file_path)


# # 1. 장르 유사도 (음식 이름)
# count_vectorizer = CountVectorizer()
# genre_matrix = count_vectorizer.fit_transform(df['음식이름'])
# genre_similarity = cosine_similarity(genre_matrix, genre_matrix)

# # 2. 감정 기반 유사도 (감정)
# tfidf_vectorizer = TfidfVectorizer()
# emotion_matrix = tfidf_vectorizer.fit_transform(df['감정'])
# emotion_similarity = cosine_similarity(emotion_matrix, emotion_matrix)

# # 3. 카테고리 유사도
# category_vectorizer = CountVectorizer()
# category_matrix = category_vectorizer.fit_transform(df['카테고리'])
# category_similarity = cosine_similarity(category_matrix, category_matrix)

# # 4. 가중치를 부여한 전체 유사도 계산
# # 감정: 0.5, 음식 이름(장르): 0.3, 카테고리: 0.2
# weighted_similarity = (
#     genre_similarity * 0.3 +
#     emotion_similarity * 0.5 +
#     category_similarity * 0.2
# )

# # 유사도 결과를 데이터프레임으로 정리
# df_similarity = pd.DataFrame(weighted_similarity, index=df['음식이름'], columns=df['음식이름'])

# # 5. 맛집 추천 로직 (사용자 감정 및 유사도 기반 추천)
# def recommend_restaurant(user_emotion, top_n=5):
#     # 사용자의 감정을 numpy 배열로 변환
#     user_emotion_vector = tfidf_vectorizer.transform([user_emotion]).toarray()

#     # 감정 유사도를 모든 음식에 대해 계산
#     emotion_similarities = cosine_similarity(user_emotion_vector, emotion_matrix)[0]

#     # 감정 유사도가 큰 순서대로 정렬 후 상위 n개 추출
#     sorted_indices = np.argsort(-emotion_similarities)

#     # 상위 n개 중 랜덤 5개 제공
#     top_indices = sorted_indices[:top_n * 2]
#     final_indices = random.sample(list(top_indices), top_n)
#     final_recommendations = df.iloc[final_indices]

#     return final_recommendations
    

# # 입력부분
# user_emotion = input("추천 받고 싶은 감정을 입력하세요: ")


# recommended_restaurants = recommend_restaurant(user_emotion)


# print(f"\n사용자 감정 '{user_emotion}'과 유사한 상위 {len(recommended_restaurants)}개의 음식 추천:")
# print(recommended_restaurants[['음식이름', '감정', '카테고리']])



# # 추천 시스템의 모든 요소를 딕셔너리로 묶기
# recommendation_system = {
#     "dataframe": df,
#     "count_vectorizer": count_vectorizer,
#     "tfidf_vectorizer": tfidf_vectorizer,
#     "genre_similarity": genre_similarity,
#     "emotion_matrix": emotion_matrix,
#     "emotion_similarity": emotion_similarity,
#     "category_similarity": category_similarity,
#     "weighted_similarity": weighted_similarity,
#     "recommend_function": recommend_restaurant
# }

# # pickle 파일로 저장
# with open("model/recommendation_system.pkl", "wb") as f:
#     pickle.dump(recommendation_system, f)

# print("추천 시스템이 recommendation_system.pkl 파일로 저장되었습니다.")


# cbf.py

import os
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 파일 경로 설정
file_path = "model/food_emotion.xlsx"
pkl_path = "model/recommendation_system.pkl"

# 데이터 및 모델 로드 또는 초기화
if not os.path.exists(pkl_path):
    # 데이터 불러오기
    df = pd.read_excel(file_path)
    
    # 유사도 계산
    count_vectorizer = CountVectorizer()
    genre_matrix = count_vectorizer.fit_transform(df['음식이름'])
    genre_similarity = cosine_similarity(genre_matrix, genre_matrix)
    
    tfidf_vectorizer = TfidfVectorizer()
    emotion_matrix = tfidf_vectorizer.fit_transform(df['감정'])
    emotion_similarity = cosine_similarity(emotion_matrix, emotion_matrix)
    
    category_vectorizer = CountVectorizer()
    category_matrix = category_vectorizer.fit_transform(df['카테고리'])
    category_similarity = cosine_similarity(category_matrix, category_matrix)
    
    # 가중치 적용한 종합 유사도
    weighted_similarity = (
        genre_similarity * 0.3 +
        emotion_similarity * 0.5 +
        category_similarity * 0.2
    )

    # 추천 시스템 객체 생성 및 저장
    recommendation_system = {
        "dataframe": df,
        "tfidf_vectorizer": tfidf_vectorizer,
        "emotion_matrix": emotion_matrix,  # Ensure emotion_matrix is stored
        "weighted_similarity": weighted_similarity
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(recommendation_system, f)
else:
    with open(pkl_path, "rb") as f:
        recommendation_system = pickle.load(f)

# 추천 함수
def recommend_restaurant(user_emotion, top_n=5):
    # 모델 데이터 로드
    df = recommendation_system["dataframe"]
    tfidf_vectorizer = recommendation_system["tfidf_vectorizer"]
    emotion_matrix = recommendation_system["emotion_matrix"]

    # 사용자의 감정을 TF-IDF 벡터로 변환
    user_emotion_vector = tfidf_vectorizer.transform([user_emotion]).toarray()

    # 모든 음식에 대해 감정 유사도 계산
    emotion_similarities = cosine_similarity(user_emotion_vector, emotion_matrix)[0]

    # 유사도 기준 상위 n개의 음식 선택
    sorted_indices = np.argsort(-emotion_similarities)
    top_indices = sorted_indices[:top_n * 2]
    final_indices = random.sample(list(top_indices), top_n)

    # 추천 음식 데이터프레임 반환
    final_recommendations = df.iloc[final_indices]
    return final_recommendations