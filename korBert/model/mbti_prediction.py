from korBert.preprocess.preprocessing import preprocess_text
import joblib, torch, os
from transformers import BertConfig, BertForSequenceClassification
from korBert.preprocess.tokenization import BertTokenizer
import numpy as np

class MBTI_korBert():
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_dir = os.path.join(project_root, 'model', 'config')

        label_encoder_path = os.path.join(config_dir, 'label_encoder_01.pkl')
        self.label_encoder = joblib.load(label_encoder_path)

        config_file = os.path.join(config_dir, 'bert_config.json')
        config = BertConfig.from_json_file(config_file)
        config.num_labels = len(self.label_encoder.classes_)

        vocab_file = os.path.join(config_dir, 'vocab.korean.rawtext.list')
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)
        self.device = torch.device("cpu")

        model_path = os.path.join(config_dir, 'pytorch_model.bin')
        # 모델을 from_pretrained로 로드
        self.model = BertForSequenceClassification.from_pretrained(model_path, config=config)

        # 최적의 state_dict 로드
        # state_dict_path = os.path.join(config_dir, 'best_model_state_02.bin')
        # state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        #
        # # 분류 레이어 가중치만 로드하지 않고, 나머지 가중치만 로드
        # model_dict = self.model.state_dict()
        # pretrained_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        # model_dict.update(pretrained_dict)
        #
        # self.model.load_state_dict(model_dict)

        self.model = BertForSequenceClassification.from_pretrained(model_path, config=config)
        state_dict = os.path.join(config_dir, 'best_model_state_02.bin')
        self.model.load_state_dict(torch.load(state_dict, map_location=torch.device('cpu')), strict=False)
        self.model.eval()

    def prepare_data_for_bert(self, input_text, max_len=512):
        tokens = self.tokenizer.tokenize(input_text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_len - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * padding_length
        token_ids += [0] * padding_length

        input_ids = torch.tensor([token_ids])
        attention_masks = torch.tensor([attention_mask])

        return input_ids, attention_masks

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

    def get_embedding(self, text):
        preprocessed_text = preprocess_text(text)
        input_ids, attention_masks = self.prepare_data_for_bert(preprocessed_text)
        with torch.no_grad():
            outputs = self.model.bert(input_ids, attention_mask=attention_masks)
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            return cls_embedding

class MBTIKorBERTSingleTone:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MBTI_korBert()
        return cls._instance





#
# input_text='''
# '사람',
#   '경우',
#   '것',
#   '편이',
#   '이',
#   '일',
#   '수',
#   '등',
#   '연예인',
#   '매우',
#   '때',
#   '성격',
#   '다른',
#   '가장',
#   '친구',
#   '즐거움',
#   '또한',
#   '자신',
#   '감정',
#   '유형',
#   '분위기',
#   '불',
#   '열정',
#   '계획',
#   '때문',
#   '데',
#   '관심',
#   '사교',
#   '주변',
#   '수도'
# '''
# predict=model.predict_mbti(input_text)
#
# print(predict)
# from transformers import AutoTokenizer
# def split_and_expand_descriptions(df, max_length=64):
#     tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-generator')
#     expanded_rows = []  # 확장된 행들을 저장할 리스트
#
#     for _, row in df.iterrows():
#         mbti = row['mbti']
#         text = row['description']
#
#         tokens = tokenizer.tokenize(text)
#         # 토큰의 총 길이가 최대 길이 이하이면, 해당 행을 그대로 추가
#         if len(tokens) <= max_length:
#             expanded_rows.append({'mbti': mbti, 'description': text})
#         else:
#             # 최대 길이를 초과하면, 쪼개어 추가
#             start = 0
#             while start < len(tokens):
#                 split_tokens = tokens[start:start+max_length]
#                 split_text = tokenizer.convert_tokens_to_string(split_tokens)
#                 expanded_rows.append({'mbti': mbti, 'description': split_text})
#                 start += max_length
#
#     # 새로운 데이터프레임 생성
#     expanded_df = pd.DataFrame(expanded_rows)
#     return expanded_df
#
# # 데이터 프레임 korBERT 모델로 임베딩
# def generate_embeddings(dataframe,column_name):
#     embeddings = []
#     for text in dataframe[column_name]:
#         embeddings.append(model.get_embedding(text))
#     return np.vstack(embeddings)

# mbti_data=pd.read_csv('../datasets/mbti_descriptions.csv',encoding='utf-8-sig')
# # 여행지 설명에 대한 임베딩 생성 및 저장
# mbti_data=split_and_expand_descriptions(mbti_data)
# mbti_embeddings = generate_embeddings(mbti_data, 'description')
# np.save('../datasets/embeddings/base_mbti_embeddings.npy', mbti_embeddings)

# MBTI 추천 목적지 설명에 대한 임베딩 생성 및 저장
# mbti_embeddings = generate_embeddings(mbti_recommend_data, 'description')
# np.save('../datasets/embeddings/mbti_embeddings.npy', mbti_embeddings)

# import glob,os
# 특정 디렉토리의 모든 txt 파일 목록을 가져옵니다.
# file_list = glob.glob('../analysis/mbti_top_words_200/*.txt')
#
# result_dict = {}

# 목록을 순회하며 각 파일을 읽습니다.
# for file_path in file_list:
#     # 파일 경로에서 파일명만 추출 (예: 'enfj_top_words.txt')
#     file_name = os.path.basename(file_path)
#     # 파일명에서 키 추출 (예: 'enfj')
#     key = file_name.split('_')[0]
#     with open(file_path, 'r', encoding='utf-8') as file:
#         # 파일의 모든 줄을 읽고 처음 20줄만 선택
#         lines = [line.strip() for line in file.readlines()]
#         result_dict[key] = lines
    # 수정된 내용을 딕셔너리에 저장

# dict={}
# for key,value in result_dict.items():
#     value=' '.join(value)
#     dict[key]=value
#
# mbti_df = pd.DataFrame(list(dict.items()), columns=['mbti', 'description'])
# split_df=split_and_expand_descriptions(mbti_df)
# # mbti_embeddings = generate_embeddings(split_df, 'description')
# split_df.to_csv('../datasets/word_base_mbti_datasets.csv')
# np.save('../datasets/embeddings/word_base_mbti_embeddings.npy', mbti_embeddings)