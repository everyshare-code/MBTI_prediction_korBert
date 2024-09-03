
# MBTI 예측 및 텍스트 처리 도구 모음

## 개요

이 프로젝트는 MBTI 성격 유형을 예측하고 여행지를 추천하는 Flask RESTful API와 자연어 처리 작업을 위한 텍스트 처리 도구 모음을 결합한 종합 솔루션입니다. 이 도구 모음은 텍스트 토크나이제이션, 구두점 처리, BERT 토크나이제이션 기능을 포함하고 있어 BERT 모델과 함께 작업할 때 특히 유용합니다.

## 목차

- [MBTI 예측 및 여행지 추천 API](#mbti-예측-및-여행지-추천-api)
- [텍스트 처리 도구 모음](#텍스트-처리-도구-모음)
- [BERT 토크나이제이션 및 증강](#bert-토크나이제이션-및-증강)
- [설치](#설치)
- [사용법](#사용법)
- [API 엔드포인트](#api-엔드포인트)
- [기여](#기여)
- [라이선스](#라이선스)
- [감사의 글](#감사의-글)
- [문의](#문의)

---

## MBTI 예측 및 여행지 추천 API

### 개요

이 API는 사용자의 입력을 기반으로 MBTI 성격 유형을 예측하고 해당 성격 유형에 맞춘 여행지를 추천하는 엔드포인트를 제공합니다. 한국어 텍스트 분류를 위해 BERT 모델을 사용한 머신러닝 기술을 활용합니다.

### 주요 기능

- **MBTI 예측**: 사용자가 제공한 텍스트 입력을 기반으로 MBTI 유형을 예측합니다.
- **여행지 추천**: 예측된 MBTI 유형 또는 사용자가 정의한 검색 쿼리에 따라 여행지를 추천합니다.
- **에러 처리**: 잘못된 요청이나 누락된 데이터에 대한 견고한 에러 처리를 제공합니다.
- **로그 기록**: 이벤트, 에러 및 디버그 정보에 대한 포괄적인 로그 기록.

### 설치

#### 필수 요건

- Python 3.x
- Flask
- Pandas
- `requirements.txt`에 명시된 기타 의존성

#### 설치 방법

1. 리포지토리 클론:
   ```bash
   git clone https://github.com/everyshare-code/mbti-api.git
   cd mbti-api
   ```

2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. 데이터셋 파일 `mbti_recommend_destination.csv`가 적절한 디렉토리에 있는지 확인합니다.

### API 엔드포인트

#### 1. MBTI 예측 엔드포인트

- **URL**: `/predict`
- **Method**: `POST`
- **요청 본문**:
  ```json
  {
      "texts": ["여기에 입력 텍스트"]
  }
  ```
- **응답**:
  - **200 OK**: 예측된 MBTI 유형과 추천 결과를 반환합니다.
  - **400 Bad Request**: JSON 데이터가 없거나 'texts' 필드가 누락된 경우.
  - **404 Not Found**: 예측된 MBTI 유형에 대한 추천 결과가 없는 경우.

#### 2. 여행지 추천 엔드포인트

- **URL**: `/recommend`
- **Method**: `POST`
- **요청 본문**:
  ```json
  {
      "query": "여기에 검색 쿼리"
  }
  ```
- **응답**:
  - **200 OK**: 추천된 여행지를 반환합니다.
  - **400 Bad Request**: JSON 데이터가 없거나 'query' 필드가 누락된 경우.
  - **404 Not Found**: 제공된 쿼리에 대한 추천 결과가 없는 경우.

## 텍스트 처리 도구 모음

### 개요

텍스트 처리 도구 모음은 텍스트 토크나이제이션과 구두점 및 한자 처리를 쉽게 할 수 있도록 설계된 파이썬 라이브러리입니다. 이 도구 모음은 특히 BERT 모델과 함께 작업할 때 자연어 처리 작업에 유용합니다.

### 기능

- **구두점 처리**: 구두점에 따라 텍스트를 세그먼트로 분할합니다.
- **한자 토크나이제이션**: 한자 주변에 공백을 추가하여 토크나이제이션을 개선합니다.
- **어휘 관리**: 파일에서 어휘를 로드하고 관리합니다.
- **공백 토크나이제이션**: 기본적인 텍스트 청소 및 공백 기반의 토크나이제이션.
- **텍스트 전처리**: HTML 태그, 특수 문자 및 과도한 공백을 제거하여 텍스트를 청소합니다.
- **BERT 토크나이제이션**: BERT 모델을 위한 토큰화 프로세스를 제공하며, 서브 토큰화 및 ID 변환을 포함합니다.

### 설치

텍스트 처리 도구 모음을 사용하려면, 시스템에 Python이 설치되어 있어야 합니다. pip을 사용하여 라이브러리를 설치할 수 있습니다:
```bash
pip install text-processing-toolkit
```

### 사용법

각 기능에 대한 자세한 사용 예제는 도구 모음의 문서를 참조하세요.

## BERT 토크나이제이션 및 증강

### 개요

이 프로젝트는 BERT 모델을 위한 텍스트 데이터를 토크나이징하는 `BertTokenizer` 클래스와, 무작위 마스킹 기법을 사용하여 텍스트 다양성을 향상시키는 `BERT_Augmentation` 클래스를 구현한 것입니다.

### 설치

이 프로젝트를 사용하려면 Python과 다음 라이브러리가 설치되어 있어야 합니다:
```bash
pip install pandas transformers
```

### 사용법

`BertTokenizer` 및 `BERT_Augmentation` 클래스의 사용 예제는 프로젝트 문서를 참조하세요.

## 기여

기여는 언제나 환영입니다! 개선 사항이나 버그 수정을 위한 풀 리퀘스트 또는 이슈를 제출해 주세요.

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 감사의 글

- [Flask](https://flask.palletsprojects.com/)
- [Flask-RESTful](https://flask-restful.readthedocs.io/)
- [CORS](https://flask-cors.readthedocs.io/)
- [Uvicorn](https://www.uvicorn.org/)
- [dotenv](https://pypi.org/project/python-dotenv/)
- [Requests](https://docs.python-requests.org/en/master/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- BERT 모델과 OpenAI API의 개발자들에게 이 프로젝트의 기반 도구들을 제공해 주신 것에 대해 특별히 감사드립니다.

## 문의

문의 사항이나 피드백이 있으시면 프로젝트 유지 관리자인 [park20542040@gmail.com](mailto:park20542040@gmail.com)으로 연락해 주세요.
