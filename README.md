## 프로젝트 개요
이 프로젝트는 한국어 BERT 모델을 사용하여 MBTI 유형을 예측하는 시스템을 개발하는 것을 목표로 합니다. 이 시스템은 텍스트 데이터를 전처리하고, BERT 모델을 사용하여 텍스트를 분류하며, 최종적으로 MBTI 유형을 예측합니다.

## 파일 구조
- `preprocess/preprocessing.py`: 텍스트 전처리 함수가 포함된 파일입니다.
- `train/MBTI_prediction_korBert.ipynb`: BERT 모델을 사용하여 MBTI 유형을 예측하는 Jupyter 노트북입니다.
- `model/mbti_prediction.py`: MBTI 예측 모델을 정의하는 파일입니다.

## 주요 기능

### 텍스트 전처리 (`preprocess/preprocessing.py`)
- `preprocess_text(text)`: 입력된 텍스트에서 HTML 태그, 대괄호 및 중괄호 내용을 제거하고, 특수 문자를 제거하며, 연속된 공백을 하나로 줄입니다.

### 모델 훈련 및 예측 (`train/MBTI_prediction_korBert.ipynb`)
- BERT 모델 설정 및 사전 훈련된 가중치 로드
  ```python
  from transformers import BertConfig, BertForSequenceClassification

  config = BertConfig.from_json_file('/content/drive/MyDrive/korBert/bert_config.json')
  config.num_labels = len(label_encoder.classes_)
  model = BertForSequenceClassification.from_pretrained('/content/drive/MyDrive/korBert/pytorch_model.bin', config=config)
  ```
- 텍스트 토큰화 및 데이터 증강
  ```python
  def tokenize_korean_text(text):
      tokens = okt.nouns(text)
      return tokens
  ```
- 각 MBTI 유형에 대해 빈도가 높은 단어 추출
  ```python
  def get_top_words_by_type_without_overlap(tokenized_texts, labels, mbti_types, top_n=100):
      top_words_by_type = {}
      all_text_by_type = {}

      for mbti_type in mbti_types:
          indices = [i for i, label in enumerate(labels) if label == mbti_type]
          # ... (이하 생략)
  ```

### MBTI 예측 (`model/mbti_prediction.py`)
- 텍스트 파일에서 데이터를 읽고, 딕셔너리에 저장한 후, 데이터프레임으로 변환하여 CSV 파일로 저장합니다.
  ```python
    def predict_mbti(self, input_text):
      preprocess_input_text = preprocess_text(input_text)
      input_ids, attention_masks = self.prepare_data_for_bert(preprocess_input_text)
      with torch.no_grad():
          outputs = self.model(input_ids.to(self.device), attention_mask=attention_masks.to(self.device))
          logits = outputs.logits
          probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
          predicted_label_idx = np.argmax(probabilities, axis=1)[0]
          probability_percent = probabilities[0][predicted_label_idx] * 100
          predicted_label = self.label_encoder.inverse_transform([predicted_label_idx])[0]
          return {'mbti': predicted_label, 'probability': probability_percent}
  ```

## 사용 방법
1. `preprocess/preprocessing.py` 파일을 사용하여 텍스트 데이터를 전처리합니다.
2. `train/MBTI_prediction_korBert.ipynb` 노트북을 실행하여 BERT 모델을 훈련시키고, MBTI 유형을 예측합니다.
3. `model/mbti_prediction.py` 파일을 사용하여 예측 결과를 저장하고, 필요한 경우 데이터를 로드합니다.

## 참고 사항
- 이 프로젝트는 한국어 텍스트 데이터를 다루기 때문에, 한국어 전처리 및 토큰화에 특화된 라이브러리를 사용합니다.
- [BERT 모델 링크](https://aiopen.etri.re.kr/bertModel)


## 라이선스
이 프로젝트는 Apache 2.0 라이선스에 따라 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
