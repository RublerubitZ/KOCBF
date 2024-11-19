# # from inference_nsmc import model_fn, input_fn, predict_fn, output_fn

# # def run_inference(model_input_data):
# #     model = model_fn()
# #     transformed_inputs = input_fn(model_input_data)
# #     predicted_classes_jsonlines = predict_fn(transformed_inputs, model)
# #     model_outputs = output_fn(predicted_classes_jsonlines)
# #     return model_outputs[0]

# # test.py
# import requests

# url = "http://127.0.0.1:8000/predict-emotion"
# text = "나 너무 화나요"

# response = requests.post(url, json={"text": text})

# if response.status_code == 200:
#     print("예측 결과:", response.json())
# else:
#     print("Error:", response.status_code, response.text)


import pickle

with open("backend/model/recommendation_system.pkl", "rb") as f:
    recommendation_system = pickle.load(f)

# 출력
print(recommendation_system.keys())