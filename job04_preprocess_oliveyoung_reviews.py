"""
[Job 04] 올리브영 리뷰 전처리
================================
실행 순서: 4번 (job03 이후)
입력: datasets/oliveyoung_reviews.csv  (job03 결과 — 제품별 개별 리뷰 행)
출력: datasets/oliveyoung_reviews_preprocessed.csv
      컬럼: product_name, review(합쳐진 원문), cleaned_review(전처리 결과)

처리 방식:
1. 같은 제품의 모든 리뷰를 하나의 긴 텍스트로 합침 (product_name 기준 groupby)
2. KoNLPy Okt로 형태소 분석 — 명사/동사/형용사 추출
3. cleaned_review 열로 저장 → job05에서 TF-IDF 학습에 사용

※ 제품 단위로 리뷰를 합치는 이유:
   리뷰 기반 추천(review mode)은 "어떤 제품의 리뷰가 사용자 고민과 가장 유사한가"를
   TF-IDF 코사인 유사도로 계산한다. 제품 하나당 행 하나여야
   유사도 계산 결과가 제품 단위 추천으로 이어진다.
"""

import pandas as pd
from konlpy.tag import Okt
import re
import os


def preprocess_reviews():
    input_file  = './datasets/oliveyoung_reviews.csv'
    output_file = './datasets/oliveyoung_reviews_preprocessed.csv'

    if not os.path.exists(input_file):
        print("Review file not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reviews.")

    # ── 1. 제품별로 리뷰 합치기 ────────────────────────────
    # 같은 product_name을 가진 모든 리뷰 텍스트를 공백으로 이어붙임
    # 결과: 제품 한 개당 한 행 (리뷰 개수 → 제품 수로 축소)
    df_merged = df.groupby('product_name')['review'].apply(
        lambda x: ' '.join(x)
    ).reset_index()
    print(f"Merged into {len(df_merged)} unique products.")

    # ── 2. 형태소 분석 및 전처리 ───────────────────────────
    okt = Okt()

    # 불용어 로드 — 파일이 없으면 기본 불용어 사용
    try:
        df_stopwords = pd.read_csv('../movie_review/datasets/stopwords.csv')
        stopwords = df_stopwords['stopword'].tolist()
    except:
        stopwords = ['가다', '하다', '있다', '없다', '좋다', '너무', '정말']

    def clean_text(text):
        """
        합쳐진 리뷰 텍스트를 TF-IDF에 적합한 형태로 정제한다.
        1. 한글 이외 문자 제거
        2. Okt 형태소 분석 (어간 추출)
        3. 명사·동사·형용사 + 2글자 이상 + 불용어 아닌 단어만 유지
        """
        text   = re.sub('[^가-힣]', ' ', text)   # 한글만 남김
        tokens = okt.pos(text, stem=True)          # 형태소 분석 + 어간 추출
        words  = []
        for word, pos in tokens:
            if pos in ['Noun', 'Verb', 'Adjective'] and len(word) > 1:
                if word not in stopwords:
                    words.append(word)
        return ' '.join(words)

    # 데이터가 많으면 시간이 걸림 (제품당 수백~수천 단어)
    print("Preprocessing merged reviews (this may take time with large data)...")
    df_merged['cleaned_review'] = df_merged['review'].apply(clean_text)

    # ── 3. 결과 저장 ────────────────────────────────────────
    df_merged.to_csv(output_file, index=False)
    print(f"Saved preprocessed data to {output_file}")


if __name__ == "__main__":
    preprocess_reviews()
