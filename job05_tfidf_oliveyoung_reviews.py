"""
[Job 05] 올리브영 리뷰 TF-IDF 학습
======================================
실행 순서: 5번 (job04 이후) — 데이터 파이프라인 마지막 단계
입력: datasets/oliveyoung_reviews_preprocessed.csv  (job04 결과)
출력: models/tfidf_reviews.pkl
      models/Tfidf_reviews.mtx

학습된 모델은 web_app.py의 '리뷰 위주 제품 찾기(review mode)'에서
사용자 입력과 제품 리뷰 간 코사인 유사도 계산에 사용된다.

행렬 구조:
- 행(row): 제품 한 개 (product_name 단위로 합쳐진 리뷰)
- 열(col): TF-IDF 어휘(vocabulary) 단어 수
- 값:      각 단어의 TF-IDF 가중치 (희소행렬)
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite
import pickle
import os


def generate_review_tfidf():
    input_file = './datasets/oliveyoung_reviews_preprocessed.csv'

    if not os.path.exists(input_file):
        print("Preprocessed file not found.")
        return

    df = pd.read_csv(input_file)

    # ── TF-IDF 벡터화 ──────────────────────────────────────
    # sublinear_tf=True: TF에 log(1+tf) 적용 — 자주 등장하는 단어의 가중치 폭발 방지
    # cleaned_review가 NaN인 경우 빈 문자열로 대체
    print("Generating TF-IDF matrix for reviews...")
    tfidf        = TfidfVectorizer(sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_review'].fillna(''))

    # ── 모델 및 행렬 저장 ──────────────────────────────────
    os.makedirs("./models", exist_ok=True)

    # 벡터라이저 저장 — 사용자 입력을 동일한 어휘 공간으로 변환할 때 필요
    with open('./models/tfidf_reviews.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    # TF-IDF 행렬 저장 — Matrix Market 형식(희소행렬에 최적화된 텍스트 포맷)
    mmwrite('./models/Tfidf_reviews.mtx', tfidf_matrix)

    print(f"Saved: tfidf_reviews.pkl, Tfidf_reviews.mtx "
          f"(Shape: {tfidf_matrix.shape})")
    # Shape 예: (384, 어휘수) — 행=제품 수, 열=학습된 단어 수


if __name__ == "__main__":
    generate_review_tfidf()
