"""
[rebuild_skin_tfidf.py] 피부 상담 TF-IDF 모델 재학습
=======================================================
실행 시점: skin_data_final.csv에 데이터를 추가한 직후
입력: datasets/skin_data_final.csv  (cleaned_question 열 포함)
출력: models/tfidf_rebuild.pkl
      models/Tfidf_skin_data_rebuild.mtx

※ web_app.py는 이 스크립트가 생성하는 *_rebuild 파일만 사용한다.
   데이터 추가 → 이 스크립트 실행 → gunicorn 재시작 순서로 갱신한다.

검증(verify_self_similarity):
  학습 완료 후 5개 샘플에 대해 "자기 자신을 가장 유사한 문서로 찾는지" 확인한다.
  self_similarity > 0.999 이어야 정상 — 낮으면 행 순서 불일치 가능성.
"""

import pickle
from pathlib import Path

import pandas as pd
from scipy.io import mmwrite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ─────────────────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_PATH   = BASE_DIR / 'datasets' / 'skin_data_final.csv'
MODEL_DIR   = BASE_DIR / 'models'
TFIDF_PATH  = MODEL_DIR / 'tfidf_rebuild.pkl'
MATRIX_PATH = MODEL_DIR / 'Tfidf_skin_data_rebuild.mtx'


def load_training_data():
    """
    skin_data_final.csv를 읽어 학습에 사용할 DataFrame을 반환한다.
    - cleaned_question 열이 반드시 있어야 함
    - NaN, 빈 문자열 행은 제거
    - web_app.py의 로드 로직과 동일한 필터링을 적용해야
      행 순서가 일치하고 인덱스가 올바르게 매핑된다.
    """
    df = pd.read_csv(DATA_PATH)
    if 'cleaned_question' not in df.columns:
        raise ValueError("Missing required column: 'cleaned_question'")

    df = df.dropna(subset=['cleaned_question']).copy()
    df['cleaned_question'] = df['cleaned_question'].astype(str).str.strip()
    df = df[df['cleaned_question'] != '']

    if df.empty:
        raise ValueError('No valid cleaned_question rows found.')
    return df


def verify_self_similarity(tfidf, tfidf_matrix, df, sample_count=5):
    """
    TF-IDF 행렬의 정합성을 검증한다.

    각 샘플 문서를 transform한 뒤 전체 행렬과 코사인 유사도를 계산하여
    자기 자신이 1위(가장 유사)로 나오는지 확인한다.

    - top_index == sample_index: 자기 자신이 최고 유사 문서
    - self_similarity > 0.999:   거의 완벽한 자기 일치
    이 두 조건을 모두 만족해야 행렬이 올바르게 생성된 것이다.
    """
    # 데이터 전체를 균등하게 커버하는 5개 인덱스 선택
    sample_indices = [0, len(df) // 4, len(df) // 2, (len(df) * 3) // 4, len(df) - 1]
    sample_indices = list(dict.fromkeys(sample_indices))[:sample_count]

    print('\nSelf similarity check:')
    all_ok = True
    for idx in sample_indices:
        sample_vec   = tfidf.transform([df.iloc[idx]['cleaned_question']])
        similarities = linear_kernel(sample_vec, tfidf_matrix)[0]
        top_index       = int(similarities.argmax())
        self_similarity = float(similarities[idx])
        ok = top_index == idx and self_similarity > 0.999
        all_ok = all_ok and ok
        print(
            f'  sample_index={idx}, top_index={top_index}, '
            f'self_similarity={self_similarity:.6f}, ok={ok}'
        )

    if not all_ok:
        raise RuntimeError('Self similarity check failed.')


def main():
    # ── 1. 데이터 로드 ─────────────────────────────────────
    df = load_training_data()
    print(f'Loaded: {DATA_PATH}')
    print(f'Rows: {len(df)}')

    # ── 2. TF-IDF 학습 ─────────────────────────────────────
    # sublinear_tf=True: log(1+tf) 적용으로 단어 빈도 편향 완화
    tfidf        = TfidfVectorizer(sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_question'])

    # ── 3. 모델 및 행렬 저장 ───────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with TFIDF_PATH.open('wb') as model_file:
        pickle.dump(tfidf, model_file)       # 벡터라이저 객체 (어휘 + 가중치 정보 포함)
    mmwrite(MATRIX_PATH, tfidf_matrix)       # 희소행렬 (Matrix Market 텍스트 포맷)

    print(f'Saved TF-IDF model:  {TFIDF_PATH}')
    print(f'Saved TF-IDF matrix: {MATRIX_PATH}')
    print(f'Matrix shape:        {tfidf_matrix.shape}')  # (문서 수, 어휘 수)
    print(f'Vocabulary size:     {len(tfidf.vocabulary_)}')

    # ── 4. 정합성 검증 ─────────────────────────────────────
    verify_self_similarity(tfidf, tfidf_matrix, df)
    print('\nRebuild complete.')


if __name__ == '__main__':
    main()
