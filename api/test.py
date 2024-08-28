import os
import pandas as pd



# 파일 존재 여부 확인

current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 경로 계산
project_root = os.path.dirname(current_dir)
# 프로젝트 루트를 기준으로 데이터셋 파일 경로 설정
mbti_datasets_path = os.path.join(project_root, 'korBert', 'datasets', 'mbti_recommend_destination.csv')
if not os.path.exists(mbti_datasets_path):
    raise FileNotFoundError(f"No such file: '{mbti_datasets_path}'")
# 파일 읽기
mbti_datasets = pd.read_csv(mbti_datasets_path, encoding='utf-8-sig')

mbti='INTP'
filtered_data = mbti_datasets[mbti_datasets['mbti'] == mbti]


result = filtered_data.to_dict(orient='records')[0]

print(result)