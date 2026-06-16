"""
[Job 01] 피부 상담 데이터 전처리 및 TF-IDF 초기 학습
=======================================================
실행 순서: 1번 (가장 먼저 실행)
입력: datasets/02.문제성 피부 메이크업 추천 데이터/ (AI Hub 원본 zip 파일들)
출력: datasets/skin_data.csv
      models/tfidf.pkl
      models/Tfidf_skin_data.mtx

※ 이 파일로 생성된 skin_data.csv는 이후 수동으로 편집하여
   skin_data_final.csv 로 정리되었고, 모델도 rebuild_skin_tfidf.py로
   재학습(tfidf_rebuild.pkl / Tfidf_skin_data_rebuild.mtx)되었습니다.
"""

import os
import json
import zipfile
import pandas as pd
from konlpy.tag import Okt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.io import mmwrite

# ─────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────
# AI Hub에서 받은 원본 데이터셋 루트 경로
dataset_base_path = './datasets/02.문제성 피부 메이크업 추천 데이터/3.개방데이터/1.데이터'

# 학습(Training) / 검증(Validation) 라벨 경로
training_label_path = os.path.join(dataset_base_path, 'Training/02.라벨링데이터')
validation_label_path = os.path.join(dataset_base_path, 'Validation/02.라벨링데이터')

# 불용어 파일 경로 (영화 리뷰 프로젝트와 공유)
stopwords_path = '../movie_review/datasets/stopwords.csv'

# ─────────────────────────────────────────────────────────
# 데이터 추출
# ─────────────────────────────────────────────────────────
data_list = []  # 추출된 행을 담을 리스트

def extract_data_from_zip(zip_dir):
    """
    지정된 폴더 안의 모든 .zip 파일을 열어 JSON 데이터를 추출한다.
    각 JSON 파일은 한 건의 피부 상담 데이터를 담고 있으며,
    필요한 필드만 꺼내 data_list에 추가한다.
    """
    if not os.path.exists(zip_dir):
        print(f"Directory not found: {zip_dir}")
        return

    for zip_file in os.listdir(zip_dir):
        if zip_file.endswith('.zip'):
            full_path = os.path.join(zip_dir, zip_file)
            with zipfile.ZipFile(full_path, 'r') as z:
                for file_name in z.namelist():
                    if file_name.endswith('.json'):
                        with z.open(file_name) as f:
                            data = json.load(f)

                            # 인적 정보: 성별, 연령대
                            h_info = data.get('Human_info', {})
                            # 피부 정보: 피부 타입
                            s_info = data.get('Skin_info', {})
                            # 상담 텍스트: 질문 / 답변 / 추천성분 / 주의성분
                            user_question     = data.get('Annotation_info', {}).get('User Question', '')
                            makeup_response   = data.get('Annotation_info', {}).get('Makeup Response', '')
                            rec_ingredients   = data.get('Annotation_info', {}).get('Recommended Ingredients', '')
                            avoid_ingredients = data.get('Annotation_info', {}).get('Ingredients to Avoid', '')

                            # User Question이 있는 데이터만 사용
                            if user_question:
                                data_list.append({
                                    'Gender':              h_info.get('Gender', ''),
                                    'Age':                 h_info.get('Age', ''),
                                    'Skin Type':           s_info.get('Skin condition category', ''),
                                    'User Question':       user_question,
                                    'Makeup Response':     makeup_response,
                                    'Recommended Ingredients': rec_ingredients,
                                    'Ingredients to Avoid':    avoid_ingredients,
                                })

print("Extracting data from Training...")
extract_data_from_zip(training_label_path)
print("Extracting data from Validation...")
extract_data_from_zip(validation_label_path)

if not data_list:
    print("No data extracted. Check dataset paths.")
    exit()

df = pd.DataFrame(data_list)
print(f"Total data extracted: {len(df)}")

# ─────────────────────────────────────────────────────────
# 텍스트 전처리 (KoNLPy Okt 형태소 분석)
# ─────────────────────────────────────────────────────────
okt = Okt()

# 불용어 로드 — 파일이 없으면 기본 불용어 목록 사용
try:
    df_stopwords = pd.read_csv(stopwords_path)
    stopwords = df_stopwords['stopword'].tolist()
except:
    stopwords = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희',
                 '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게']

def preprocess_text(text):
    """
    User Question을 TF-IDF 학습에 적합한 형태로 정제한다.
    1. 한글 이외 문자 제거
    2. Okt 형태소 분석 (어간 추출 포함)
    3. 명사·동사·형용사만 남기고, 2글자 미만 및 불용어 제거
    """
    text = re.sub('[^가-힣]', ' ', text)          # 한글만 남김
    tokens = okt.pos(text, stem=True)              # 형태소 분석 + 어간 추출
    words = []
    for word, pos in tokens:
        if pos in ['Noun', 'Verb', 'Adjective']:  # 의미 있는 품사만
            if len(word) > 1 and word not in stopwords:
                words.append(word)
    return ' '.join(words)

print("Preprocessing text...")
df['cleaned_question'] = df['User Question'].apply(preprocess_text)

# 전처리 후 빈 행 제거
df.dropna(subset=['cleaned_question'], inplace=True)
df = df[df['cleaned_question'] != '']

# ─────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────
os.makedirs('./datasets', exist_ok=True)
df.to_csv('./datasets/skin_data.csv', index=False)
print("Saved skin_data.csv")

# ─────────────────────────────────────────────────────────
# TF-IDF 벡터화 및 모델 저장
# ─────────────────────────────────────────────────────────
print("Generating TF-IDF matrix...")
# sublinear_tf=True: TF 값에 log(1+tf)를 적용해 빈도 폭발 방지
tfidf = TfidfVectorizer(sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(df['cleaned_question'])

os.makedirs('./models', exist_ok=True)

# TF-IDF 벡터라이저 객체 저장 (추후 transform에 재사용)
with open('./models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# TF-IDF 행렬을 Matrix Market 형식(.mtx)으로 저장 — 희소행렬에 적합
mmwrite('./models/Tfidf_skin_data.mtx', tfidf_matrix)
print("Saved models (tfidf.pkl, Tfidf_skin_data.mtx)")
