import os
import json
import zipfile
import pandas as pd
from konlpy.tag import Okt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy.io import mmwrite

# 경로 설정
dataset_base_path = './datasets/02.문제성 피부 메이크업 추천 데이터/3.개방데이터/1.데이터'
training_label_path = os.path.join(dataset_base_path, 'Training/02.라벨링데이터')
validation_label_path = os.path.join(dataset_base_path, 'Validation/02.라벨링데이터')
stopwords_path = '../movie_review/datasets/stopwords.csv'

# 데이터 추출
data_list = []

def extract_data_from_zip(zip_dir):
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
                            # 필요한 정보 추출
                            h_info = data.get('Human_info', {})
                            s_info = data.get('Skin_info', {})
                            user_question = data.get('Annotation_info', {}).get('User Question', '')
                            makeup_response = data.get('Annotation_info', {}).get('Makeup Response', '')
                            rec_ingredients = data.get('Annotation_info', {}).get('Recommended Ingredients', '')
                            avoid_ingredients = data.get('Annotation_info', {}).get('Ingredients to Avoid', '')
                            
                            if user_question:
                                data_list.append({
                                    'Gender': h_info.get('Gender', ''),
                                    'Age': h_info.get('Age', ''),
                                    'Skin Type': s_info.get('Skin condition category', ''),
                                    'User Question': user_question,
                                    'Makeup Response': makeup_response,
                                    'Recommended Ingredients': rec_ingredients,
                                    'Ingredients to Avoid': avoid_ingredients
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

# 전처리
okt = Okt()
try:
    df_stopwords = pd.read_csv(stopwords_path)
    stopwords = df_stopwords['stopword'].tolist()
except:
    stopwords = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게']

def preprocess_text(text):
    text = re.sub('[^가-힣]', ' ', text)
    tokens = okt.pos(text, stem=True)
    words = []
    for word, pos in tokens:
        if pos in ['Noun', 'Verb', 'Adjective']:
            if len(word) > 1 and word not in stopwords:
                words.append(word)
    return ' '.join(words)

print("Preprocessing text...")
df['cleaned_question'] = df['User Question'].apply(preprocess_text)
df.dropna(subset=['cleaned_question'], inplace=True)
df = df[df['cleaned_question'] != '']

# 저장
os.makedirs('./datasets', exist_ok=True)
df.to_csv('./datasets/skin_data.csv', index=False)
print("Saved skin_data.csv")

# TF-IDF 변환
print("Generating TF-IDF matrix...")
tfidf = TfidfVectorizer(sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(df['cleaned_question'])

os.makedirs('./models', exist_ok=True)
with open('./models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
mmwrite('./models/Tfidf_skin_data.mtx', tfidf_matrix)
print("Saved models (tfidf.pkl, Tfidf_skin_data.mtx)")
