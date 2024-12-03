# main.py

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import load_model, predict # 감정 예측 부분(KoBERT)-CBF 연동
from ncf.ncf_recommend import recommend_restaurants # NCF-recommend 부분
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity # 유사도 계산
from cbf import recommend_restaurant # cbf
from fastapi.middleware.cors import CORSMiddleware # CORS 오류 해결

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인을 허용.
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용.
    allow_headers=["*"],  # 모든 HTTP 헤더를 허용.
)

# 모델 로드
model_path = "model/model_statedict.pkl"
cbf_model_path = "model/recommendation_system.pkl"
model = load_model(model_path)

with open(cbf_model_path, "rb") as f:
    recommendation_system = pickle.load(f)

class TextRequest(BaseModel):
    text: str
    
class RecommendRequest(BaseModel):
    food_category: str

def recommend_restaurant(user_emotion, top_n=5):
    df = recommendation_system["dataframe"]
    tfidf_vectorizer = recommendation_system["tfidf_vectorizer"]
    emotion_matrix = recommendation_system["emotion_matrix"]

    user_emotion_vector = tfidf_vectorizer.transform([user_emotion]).toarray()
    emotion_similarities = cosine_similarity(user_emotion_vector, emotion_matrix)[0]
    sorted_indices = np.argsort(-emotion_similarities)
    top_indices = sorted_indices[:top_n * 2]
    final_indices = random.sample(list(top_indices), top_n)
    final_recommendations = df.iloc[final_indices]
    
    return final_recommendations[["음식이름", "감정", "카테고리"]].to_dict(orient="records")

@app.get("/")
async def root():
    return {"message": "Welcome to the Recommendation API!"}

# KOBERT+CBF
@app.post("/predict-and-recommend")
async def predict_and_recommend(request: TextRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # 감정 예측
    emotion_result = predict(model, text)
    user_emotion = emotion_result["emotion"]
    
    # 추천 결과
    recommendations = recommend_restaurant(user_emotion)
    
    return {"emotion": emotion_result, "recommendations": recommendations}

# NCF 추천 엔드포인트
@app.post("/recommend-ncf")
async def recommend_ncf(request: RecommendRequest):
    recommendations = recommend_restaurants(request.food_category)
    if recommendations is None:
        raise HTTPException(status_code=404, detail=f"'{request.food_category}'에 해당하는 맛집을 찾을 수 없습니다.")
    
    return {"recommendations": recommendations}

# 서버 시작
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 카카오 API 키::

# REST API(FASTAPI) - "2e32d79471d4fd0df0d926d502d796b7"
# JAVASCRIPTS(REACT) - "1a861a21e933ebb79b537a5418598c75"



# 아래서부터 카카오 API 연구 VV

# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from model import load_model, predict  # KoBERT - CBF 연동
# from ncf.ncf_recommend import recommend_restaurants  # NCF-recommend 부분
# from cbf import recommend_restaurant # cbf
# import pickle
# import random
# from sklearn.metrics.pairwise import cosine_similarity  # 유사도 계산
# from fastapi.middleware.cors import CORSMiddleware  # CORS 오류 해결
# import requests

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 모든 도메인을 허용.
#     allow_credentials=True,
#     allow_methods=["*"],  # 모든 HTTP 메서드를 허용.
#     allow_headers=["*"],  # 모든 HTTP 헤더를 허용.
# )

# # 모델 로드
# model_path = "model/model_statedict.pkl"
# cbf_model_path = "model/recommendation_system.pkl"
# model = load_model(model_path)

# with open(cbf_model_path, "rb") as f:
#     recommendation_system = pickle.load(f)

# KAKAO_API_KEY = "2e32d79471d4fd0df0d926d502d796b7"  # 자신의 카카오 API 키로 대체


# def get_restaurants_nearby(lat, lng, radius=10000):
#     """
#     카카오맵 API를 사용해 반경 내 음식점 정보를 가져옵니다.
#     """
#     url = "https://dapi.kakao.com/v2/local/search/category.json"
#     headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
#     params = {
#         "category_group_code": "FD6",  # 음식점 카테고리
#         "x": lng,  # 경도
#         "y": lat,  # 위도
#         "radius": radius,  # 반경 (단위: 미터)
#         "sort": "distance"
#     }

#     response = requests.get(url, headers=headers, params=params)
#     if response.status_code == 200:
#         return response.json()["documents"]
#     else:
#         raise Exception(f"API 호출 실패: {response.status_code}, {response.text}")


# def filter_recommendations_by_distance(recommendations, user_lat, user_lng):
#     """
#     NCF 추천 결과에서 카카오맵 API의 반경 내 음식점만 필터링합니다.
#     """
#     nearby_restaurants = get_restaurants_nearby(user_lat, user_lng)
#     nearby_ids = {res["id"] for res in nearby_restaurants}

#     filtered_recommendations = [
#         rec for rec in recommendations if rec["restaurant_id"] in nearby_ids
#     ]
#     return filtered_recommendations


# class TextRequest(BaseModel):
#     text: str


# class RecommendRequest(BaseModel):
#     food_category: str
#     latitude: float
#     longitude: float


# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Recommendation API!"}


# # KoBERT + CBF
# @app.post("/predict-and-recommend")
# async def predict_and_recommend(request: TextRequest):
#     text = request.text
#     if not text:
#         raise HTTPException(status_code=400, detail="Text is required")

#     # 감정 예측
#     emotion_result = predict(model, text)
#     user_emotion = emotion_result["emotion"]

#     # 추천 결과
#     recommendations = recommend_restaurant(user_emotion)

#     return {"emotion": emotion_result, "recommendations": recommendations}


# # NCF + 카카오맵 연동 추천 엔드포인트
# @app.post("/recommend-with-map")
# async def recommend_with_map(request: RecommendRequest):
#     # NCF 추천
#     ncf_recommendations = recommend_restaurants(request.food_category)
#     if ncf_recommendations is None:
#         raise HTTPException(status_code=404, detail="추천 결과가 없습니다.")

#     # 거리 필터링
#     filtered_recommendations = filter_recommendations_by_distance(
#         ncf_recommendations, request.latitude, request.longitude
#     )

#     return {"recommendations": filtered_recommendations}


# # 서버 시작
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)