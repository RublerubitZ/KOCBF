# # main.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from model import load_model, predict

# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# from typing import Optional
# from fastapi.exceptions import RequestValidationError
# from fastapi import Request
# from fastapi.responses import JSONResponse


# app = FastAPI()

# # 모델 경로
# MODEL_PATH = "model/model_statedict.pkl"  # 실제 모델 저장 경로로 변경 필요

# # 모델 로드
# model = load_model(MODEL_PATH)

# class TextRequest(BaseModel):
#     text: str

# @app.post("/predict-emotion")
# async def predict_emotion(request: TextRequest):
#     text = request.text
#     if not text:
#         raise HTTPException(status_code=400, detail="Text is required")

#     result = predict(model, text)
#     return result

# # 서버 시작
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# main.py

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import load_model, predict # 감정 예측 부분(KoBERT)
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity # 유사도 계산
from cbf import recommend_restaurant # cbf
from fastapi.middleware.cors import CORSMiddleware # CORS 오류 해결

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인을 허용. 특정 도메인만 허용하려면 ["http://example.com"]과 같이 작성.
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


# 서버 시작
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)