# model.py
import torch
import numpy as np
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import gluonnlp as nlp
from torch import nn

class BERTClassifier(nn.Module):
    def __init__(self,
                bert,
                hidden_size=768,
                num_classes=5,  # 클래스 수 조정
                dr_rate=0.5,    # 기본 드롭아웃 확률을 0.5로 설정
                params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:  # dr_rate가 None이 아닐 때만 Dropout 정의
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                            attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        
        if self.dr_rate:  # dr_rate가 설정된 경우에만 Dropout 적용
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)

# Tokenizer와 KoBERT 모델 초기화
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# 저장된 모델 로드 함수
def load_model(model_path):
    model = BERTClassifier(bertmodel).to("cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()
    return model

# 감정 예측 함수
def predict(model, text, max_len=64):
    # Tokenize input text
    tokenized = tokenizer.tokenize(text)
    
    # Convert tokenized text to token IDs
    token_ids = [vocab[token] for token in tokenized]
    token_ids = token_ids[:max_len]  # max_len보다 길면 자르기
    segment_ids = [0] * len(token_ids)  # KoBERT는 단일 문장 처리이므로 모두 0으로 설정
    valid_length = len(token_ids)

    # 패딩 적용
    if len(token_ids) < max_len:
        token_ids += [vocab['[PAD]']] * (max_len - len(token_ids))
        segment_ids += [0] * (max_len - len(segment_ids))

    # Tensor로 변환
    token_ids = torch.tensor([token_ids])
    valid_length = torch.tensor([valid_length])
    segment_ids = torch.tensor([segment_ids])

    with torch.no_grad():
        output = model(token_ids, valid_length, segment_ids)
        softmax_output = torch.nn.functional.softmax(output, dim=1).squeeze().cpu().numpy()
        
        # 감정 라벨 및 예측 확률 출력
        emotion_labels = ["놀람", "분노", "슬픔", "중립", "행복"]
        # numpy.float32 값을 float으로 변환
        emotion_scores = {label: float(score * 100) for label, score in zip(emotion_labels, softmax_output)}
        predicted_emotion = emotion_labels[np.argmax(softmax_output)]
        
    return {"emotion": predicted_emotion, "scores": emotion_scores}