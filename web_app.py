"""
[web_app.py] MIMO 스킨케어 추천 Flask 웹 서버
================================================
역할: 유일한 런타임 파일 — 모든 HTTP 요청을 처리한다.
실행: gunicorn -w 2 -b 0.0.0.0:5000 web_app:app

주요 기능:
  1. Expert Mode   : 피부 상담 데이터(skin_data_final.csv) 기반 TF-IDF 코사인 유사도 추천
  2. Review Mode   : 올리브영 리뷰(oliveyoung_reviews_preprocessed.csv) 기반 제품 추천
  3. 연관 제품 추천 : 추천 성분·응답 키워드로 oliveyoung_product_list.csv에서 제품 검색
  4. 회원 시스템   : 가입·로그인·검색 이력·관리자 페이지 (SQLite)
  5. 보안          : CSRF 토큰, 세션 고정 방지, 리디렉션 검증, 비밀번호 해싱
"""

import os
import pickle
import re
import secrets
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from flask import (
    Flask, abort, flash, g, jsonify, redirect, render_template, request,
    session, url_for,
)
from konlpy.tag import Okt
from scipy.io import mmread
from sklearn.metrics.pairwise import linear_kernel
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════
# Flask 앱 초기화 및 설정
# ═══════════════════════════════════════════════════════════

def load_secret_key(instance_path):
    """
    Flask 세션 암호화에 사용하는 SECRET_KEY를 가져온다.
    환경변수 SKINSCOPE_SECRET_KEY가 있으면 우선 사용하고,
    없으면 instance/secret_key 파일에서 읽는다 (최초 실행 시 자동 생성).
    """
    env_secret = os.environ.get('SKINSCOPE_SECRET_KEY')
    if env_secret:
        return env_secret
    secret_file = Path(instance_path) / 'secret_key'
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32), encoding='ascii')
    return secret_file.read_text(encoding='ascii').strip()


app = Flask(__name__, instance_relative_config=True)
app.config.update(
    SECRET_KEY=load_secret_key(app.instance_path),
    DATABASE=Path(app.instance_path) / 'users.db',
    SESSION_COOKIE_HTTPONLY=True,    # JS에서 쿠키 접근 차단 (XSS 방어)
    SESSION_COOKIE_SAMESITE='Lax',   # CSRF 방어용 SameSite 쿠키 설정
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,  # 로그인 세션 유지 시간: 8시간
)
Path(app.instance_path).mkdir(parents=True, exist_ok=True)

# 이전 버전 DB 파일(users_v9.db)이 있고 새 DB가 없으면 마이그레이션
legacy_database = Path(app.instance_path) / ('users_' + 'v' + '9.db')
if not app.config['DATABASE'].exists() and legacy_database.exists():
    shutil.copy2(legacy_database, app.config['DATABASE'])


# ═══════════════════════════════════════════════════════════
# 데이터 및 모델 로드 (서버 시작 시 1회 실행)
# ═══════════════════════════════════════════════════════════

print('Loading data and models...')

# ── 피부 상담 데이터 (Expert Mode용) ─────────────────────
df_skin      = pd.read_csv(BASE_DIR / 'datasets' / 'skin_data_final.csv')
# mmread: Matrix Market 희소행렬 파일 → CSR 포맷으로 변환 (연산 효율 최적화)
tfidf_matrix = mmread(BASE_DIR / 'models' / 'Tfidf_skin_data_rebuild.mtx').tocsr()
with open(BASE_DIR / 'models' / 'tfidf_rebuild.pkl', 'rb') as model_file:
    tfidf = pickle.load(model_file)

# 필수 컬럼 검증
required_skin_columns = {
    'Gender', 'Age', 'Skin Type', 'User Question', 'Makeup Response',
    'Recommended Ingredients', 'Ingredients to Avoid', 'cleaned_question',
}
missing_skin_columns = required_skin_columns.difference(df_skin.columns)
if missing_skin_columns:
    raise RuntimeError(f'피부 상담 데이터 필수 열이 없습니다: {sorted(missing_skin_columns)}')

# rebuild_skin_tfidf.py의 load_training_data()와 동일한 전처리를 적용해
# df_skin의 행 순서와 tfidf_matrix의 행 순서를 정확히 일치시킨다
df_skin = df_skin.dropna(subset=['cleaned_question']).copy()
df_skin['cleaned_question'] = df_skin['cleaned_question'].astype(str).str.strip()
df_skin = df_skin[df_skin['cleaned_question'] != ''].reset_index(drop=True)

# 행렬 크기 정합성 검증
if tfidf_matrix.shape[0] != len(df_skin):
    raise RuntimeError('피부 상담 데이터와 rebuild TF-IDF 행 수가 일치하지 않습니다.')
if tfidf_matrix.shape[1] != len(tfidf.get_feature_names_out()):
    raise RuntimeError('rebuild TF-IDF 모델과 행렬의 특성 수가 일치하지 않습니다.')

# ── 앱 전역 상수 ─────────────────────────────────────────
# 제품 추천에서 사용할 스킨케어 카테고리 (메이크업 카테고리 제외)
SKINCARE_CATEGORIES = {'스킨/토너', '에센스/세럼/앰플', '크림', '로션'}

# 프로필 선택 옵션 (입력값 유효성 검증에 사용)
PROFILE_OPTIONS = {
    'gender':    {'성별 선택', '남성', '여성'},
    'age':       {'연령대 선택', '20대', '30대', '40대', '50대 이상'},
    'skin_type': {'피부 타입 선택', '건성', '정상', '지성', '복합성'},
}
PROFILE_LABELS = {
    'gender':    '성별',
    'age':       '연령대',
    'skin_type': '피부 타입',
}

# ── 리뷰/제품 데이터 (Review Mode용) ─────────────────────
# 파일이 없거나 형식이 맞지 않으면 None으로 유지 (기능 비활성화)
df_product_list      = None
df_reviews           = None
tfidf_matrix_reviews = None
tfidf_reviews        = None
try:
    product_data    = pd.read_csv(BASE_DIR / 'datasets' / 'oliveyoung_product_list.csv')
    review_data     = pd.read_csv(BASE_DIR / 'datasets' / 'oliveyoung_reviews_preprocessed.csv')
    review_matrix   = mmread(BASE_DIR / 'models' / 'Tfidf_reviews.mtx').tocsr()
    with open(BASE_DIR / 'models' / 'tfidf_reviews.pkl', 'rb') as model_file:
        review_vectorizer = pickle.load(model_file)

    # 스킨케어 카테고리 제품만 필터링
    product_data = product_data[product_data['category'].isin(SKINCARE_CATEGORIES)].copy()

    # 제품 메타데이터(브랜드·링크·카테고리)를 리뷰 데이터에 조인
    product_lookup = product_data[
        ['product_name', 'product_brand', 'product_link', 'category']
    ].drop_duplicates(subset='product_name', keep='first')
    review_data = review_data.merge(product_lookup, on='product_name', how='left')

    # 리뷰 행렬에서 스킨케어 제품 행만 추출 (카테고리 매칭)
    skincare_mask = review_data['category'].isin(SKINCARE_CATEGORIES).to_numpy()

    # 정합성 검증
    if review_matrix.shape[0] != len(review_data):
        raise ValueError('리뷰 CSV와 TF-IDF 행 수가 일치하지 않습니다.')
    if review_matrix.shape[1] != len(review_vectorizer.get_feature_names_out()):
        raise ValueError('리뷰 TF-IDF 모델과 행렬의 특성 수가 일치하지 않습니다.')

    df_product_list      = product_data.reset_index(drop=True)
    df_reviews           = review_data.loc[skincare_mask].reset_index(drop=True)
    tfidf_matrix_reviews = review_matrix[skincare_mask]
    tfidf_reviews        = review_vectorizer
except (FileNotFoundError, KeyError, ValueError, pickle.UnpicklingError) as exc:
    # 리뷰 데이터가 없어도 Expert Mode는 정상 동작
    print(f'Review recommendation unavailable: {exc}')

# ── KoNLPy 형태소 분석기 초기화 ──────────────────────────
okt = Okt()

# 불용어 로드
try:
    df_stopwords = pd.read_csv(BASE_DIR / 'datasets' / 'stopwords.csv')
    stopwords = df_stopwords['stopword'].tolist()
except (FileNotFoundError, KeyError):
    stopwords = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희',
                 '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게']

print('Load complete.')


# ═══════════════════════════════════════════════════════════
# SQLite 데이터베이스 유틸리티
# ═══════════════════════════════════════════════════════════

def get_db():
    """
    현재 요청 컨텍스트(g)에서 DB 연결을 반환한다.
    같은 요청 안에서 여러 번 호출해도 연결을 재사용한다.
    row_factory=sqlite3.Row로 설정해 결과를 딕셔너리처럼 접근 가능.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')  # 외래키 제약 활성화
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    """요청이 끝날 때 DB 연결을 자동으로 닫는다."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """
    앱 시작 시 DB 테이블과 기본 관리자 계정을 생성한다.
    이미 존재하면 CREATE TABLE IF NOT EXISTS로 무시된다.

    테이블:
    - users        : 회원 정보 (프로필 포함)
    - search_history: 검색 이력 (비회원도 user_id=NULL로 저장)
    """
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'admin')),
            gender TEXT,
            age TEXT,
            skin_type TEXT,
            recommendation_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            keyword TEXT,
            search_terms TEXT,
            gender TEXT,
            age TEXT,
            skin_type TEXT,
            recommended_ingredients TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at DESC);
    ''')

    # 관리자 계정 초기 생성 (환경변수로 커스터마이징 가능)
    admin_username = os.environ.get('SKINSCOPE_ADMIN_USERNAME', 'admin').strip().lower()
    admin_password = os.environ.get('SKINSCOPE_ADMIN_PASSWORD', 'Admin1234!')
    admin_email    = os.environ.get('SKINSCOPE_ADMIN_EMAIL', 'admin@skinscope.local').strip().lower()
    existing = db.execute('SELECT id FROM users WHERE username = ?', (admin_username,)).fetchone()
    if existing is None:
        db.execute(
            '''INSERT INTO users
               (username, name, email, password_hash, role, created_at)
               VALUES (?, ?, ?, ?, 'admin', ?)''',
            (admin_username, '관리자', admin_email,
             generate_password_hash(admin_password), utc_now()),
        )
        db.commit()
        print(f"Admin account created: {admin_username}")


# ═══════════════════════════════════════════════════════════
# 유틸리티 함수
# ═══════════════════════════════════════════════════════════

def utc_now():
    """현재 UTC 시각을 ISO 8601 문자열로 반환한다 (DB 저장용)."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def csrf_token():
    """
    CSRF 방어용 토큰을 세션에 저장하고 반환한다.
    Jinja2 템플릿에서 {{ csrf_token() }} 으로 호출해 form hidden 필드에 삽입한다.
    """
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_urlsafe(32)
    return session['_csrf_token']


# Jinja2 전역 함수로 등록 — 템플릿에서 직접 호출 가능
app.jinja_env.globals['csrf_token'] = csrf_token


@app.template_filter('kst')
def format_kst(value, date_format='%Y-%m-%d %H:%M'):
    """UTC ISO 문자열을 KST(한국 표준시) 포맷으로 변환하는 Jinja2 필터."""
    if not value:
        return '-'
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(ZoneInfo('Asia/Seoul')).strftime(date_format)


def validate_csrf():
    """
    POST 요청의 CSRF 토큰을 검증한다.
    토큰이 없거나 세션 값과 다르면 400 오류를 반환한다.
    secrets.compare_digest: 타이밍 공격 방지를 위한 상수 시간 비교.
    """
    submitted = request.form.get('csrf_token', '')
    if not submitted or not secrets.compare_digest(submitted, session.get('_csrf_token', '')):
        abort(400, description='유효하지 않은 요청입니다. 페이지를 새로고침해 주세요.')


def is_safe_redirect(target):
    """
    리디렉션 대상 URL이 같은 호스트를 가리키는지 검증한다.
    외부 도메인으로의 오픈 리디렉션(Open Redirect) 공격을 방지한다.
    """
    host_url     = urlparse(request.host_url)
    redirect_url = urlparse(urljoin(request.host_url, target))
    return redirect_url.scheme in ('http', 'https') and host_url.netloc == redirect_url.netloc


# ═══════════════════════════════════════════════════════════
# 인증 데코레이터
# ═══════════════════════════════════════════════════════════

def login_required(view):
    """로그인하지 않은 사용자를 로그인 페이지로 리디렉션하는 데코레이터."""
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash('로그인이 필요한 서비스입니다.', 'info')
            return redirect(url_for('login', next=request.path))
        return view(**kwargs)
    return wrapped_view


def admin_required(view):
    """관리자(role='admin')만 접근 가능한 뷰에 적용하는 데코레이터."""
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash('관리자 로그인이 필요합니다.', 'info')
            return redirect(url_for('login', next=request.path))
        if g.user['role'] != 'admin':
            abort(403)
        return view(**kwargs)
    return wrapped_view


@app.before_request
def load_logged_in_user():
    """
    모든 요청 처리 전에 세션에서 user_id를 읽어 g.user에 저장한다.
    로그인 상태가 아니면 g.user = None.
    """
    user_id = session.get('user_id')
    g.user = None if user_id is None else get_db().execute(
        '''SELECT id, username, name, email, role, gender, age, skin_type,
                  recommendation_count, created_at, last_login
           FROM users WHERE id = ?''',
        (user_id,),
    ).fetchone()


@app.context_processor
def inject_user():
    """모든 Jinja2 템플릿에 current_user 변수를 자동 주입한다."""
    return {'current_user': g.user}


# ═══════════════════════════════════════════════════════════
# 텍스트 전처리 및 추천 핵심 로직
# ═══════════════════════════════════════════════════════════

def preprocess_input(text):
    """
    사용자 입력 텍스트를 TF-IDF 검색에 적합한 형태로 전처리한다.
    - 한글 이외 문자 제거
    - Okt 형태소 분석 (어간 추출)
    - 명사·동사·형용사 + 2글자 이상 + 불용어 제거
    반환값: 공백으로 연결된 단어 문자열 (예: "건조 각질 심해지다")
    """
    text   = re.sub('[^가-힣]', ' ', text)
    tokens = okt.pos(text, stem=True)
    words  = [word for word, part in tokens
              if part in ['Noun', 'Verb', 'Adjective'] and len(word) > 1 and word not in stopwords]
    return ' '.join(words)


def select_expert_row(user_input, cleaned, gender, age, skin_type):
    """
    사용자 입력과 가장 유사한 피부 상담 데이터 행을 선택한다.

    프로필 필터 + 코사인 유사도 3단계 폴백:
      Stage 1 — 성별·연령대·피부타입 모두 일치하는 데이터 중 최고 유사도
      Stage 2 — 성별만 일치하는 데이터 중 최고 유사도 (연령대·피부타입 완화)
      Stage 3 — 전체 데이터 중 최고 유사도 (프로필 무관)

    각 단계에서 THRESHOLD(0.20) 이상이어야 해당 결과를 채택한다.
    낮은 유사도로 엉뚱한 데이터가 선택되는 것을 방지하기 위한 최솟값.
    """
    similarities = linear_kernel(tfidf.transform([cleaned]), tfidf_matrix)[0]
    THRESHOLD = 0.20  # 유의미한 유사도 최솟값

    def best_in_mask(mask):
        """마스크가 True인 행들 중 유사도가 가장 높은 행의 인덱스와 점수를 반환."""
        s = similarities.copy()
        s[~mask] = -1          # 해당 안 되는 행은 -1로 설정해 선택 대상 제외
        idx = int(s.argmax())
        return idx, s[idx]

    # Stage 1: 프로필 전체 일치
    mask = np.ones(len(df_skin), dtype=bool)
    if gender    != '성별 선택':    mask &= df_skin['Gender'].eq(gender).to_numpy()
    if age       != '연령대 선택':  mask &= df_skin['Age'].eq(age).to_numpy()
    if skin_type != '피부 타입 선택': mask &= df_skin['Skin Type'].eq(skin_type).to_numpy()
    idx, score = best_in_mask(mask)
    if score >= THRESHOLD:
        return df_skin.iloc[idx]

    # Stage 2: 성별만 일치 (나이·피부타입 조건 완화)
    mask2 = np.ones(len(df_skin), dtype=bool)
    if gender != '성별 선택':
        mask2 &= df_skin['Gender'].eq(gender).to_numpy()
    idx, score = best_in_mask(mask2)
    if score >= THRESHOLD:
        return df_skin.iloc[idx]

    # Stage 3: 전체에서 최고 유사도 (프로필 무관)
    return df_skin.iloc[int(similarities.argmax())]


def validated_profile_value(field, value):
    """
    프로필 필드 값이 허용 목록에 있으면 그대로, 없으면 기본값('선택')을 반환한다.
    클라이언트에서 조작된 값이 들어오는 것을 방어한다.
    """
    value    = str(value)
    defaults = {
        'gender':    '성별 선택',
        'age':       '연령대 선택',
        'skin_type': '피부 타입 선택',
    }
    return value if value in PROFILE_OPTIONS[field] else defaults[field]


def validate_optional_profile(field, value):
    """
    회원가입 폼의 선택형 프로필 필드 값을 검증한다.
    빈 값이면 (None, None) 반환, 유효하지 않은 값이면 (None, 오류메시지) 반환.
    """
    value = str(value).strip()
    if not value:
        return None, None
    if value not in PROFILE_OPTIONS[field] or value.endswith('선택'):
        return None, f"올바른 {PROFILE_LABELS[field]} 값을 선택해 주세요."
    return value, None


# ═══════════════════════════════════════════════════════════
# 인기 검색어
# ═══════════════════════════════════════════════════════════

def normalize_popular_term(term):
    """
    인기 검색어 집계 시 '건조하다' → '건조' 처럼 동사형을 어근으로 정규화한다.
    같은 의미의 단어가 분리되어 집계되는 것을 방지.
    """
    if term.endswith('하다') and len(term) > 2:
        return term[:-2]
    return term


def get_popular_terms(limit=5):
    """
    최근 검색 이력 1000건에서 자주 등장하는 단어 top-N을 반환한다.
    '피부', '고민' 등 너무 일반적인 단어는 제외한다.
    """
    rows = get_db().execute(
        '''SELECT search_terms FROM search_history
           WHERE search_terms IS NOT NULL AND search_terms != ''
           ORDER BY id DESC LIMIT 1000'''
    ).fetchall()
    counter       = Counter()
    ignored_terms = {'피부', '고민', '하다', '되다', '있다', '생기다', '자주'}
    for row in rows:
        terms = {normalize_popular_term(term) for term in row['search_terms'].split()
                 if term not in ignored_terms}
        counter.update(terms)
    return counter.most_common(limit)


def get_trending_terms(limit=5):
    """
    인기 검색어를 가져오되, 부족하면 기본 키워드로 채워 항상 limit개를 반환한다.
    분석 페이지 하단 트렌딩 버튼에 사용.
    """
    ranked = get_popular_terms(limit)
    seen   = {term for term, _count in ranked}
    for term in ('건조', '트러블', '민감', '모공', '각질'):
        if len(ranked) >= limit:
            break
        if term not in seen:
            ranked.append((term, None))
            seen.add(term)
    return ranked


# ═══════════════════════════════════════════════════════════
# 검색 이력 기록
# ═══════════════════════════════════════════════════════════

def record_search(user_input, cleaned, gender, age, skin_type, row):
    """
    검색 요청을 search_history 테이블에 기록한다.
    - 로그인 회원: user_id 포함
    - 비회원: user_id = NULL
    - Expert Mode: recommended_ingredients = 추천 성분 텍스트
    - Review Mode: recommended_ingredients = '리뷰 기반 제품 추천' (row가 None)
    """
    db = get_db()
    db.execute(
        '''INSERT INTO search_history
           (user_id, keyword, search_terms, gender, age, skin_type,
            recommended_ingredients, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (g.user['id'] if g.user is not None else None,
         user_input or None,
         cleaned    or None,
         None if gender    == '성별 선택'     else gender,
         None if age       == '연령대 선택'   else age,
         None if skin_type == '피부 타입 선택' else skin_type,
         str(row['Recommended Ingredients']) if row is not None else '리뷰 기반 제품 추천',
         utc_now()),
    )
    db.commit()


# ═══════════════════════════════════════════════════════════
# 라우트 — 페이지
# ═══════════════════════════════════════════════════════════

@app.route('/')
def index():
    """메인 페이지: 로그인 회원의 최근 검색 이력 5건과 인기 검색어를 전달한다."""
    recent_history = []
    if g.user is not None:
        recent_history = get_db().execute(
            '''SELECT keyword, gender, age, skin_type, recommended_ingredients, created_at
               FROM search_history WHERE user_id = ?
               ORDER BY id DESC LIMIT 5''',
            (g.user['id'],),
        ).fetchall()
    return render_template(
        'index.html',
        recent_history=recent_history,
        popular_terms=get_popular_terms(),
    )


@app.route('/analysis')
def analysis():
    """분석 페이지: 리뷰 모드 사용 가능 여부와 트렌딩 키워드를 전달한다."""
    return render_template(
        'analysis.html',
        review_mode_available=df_reviews is not None and len(df_reviews) > 0,
        popular_terms=get_trending_terms(),
    )


# ═══════════════════════════════════════════════════════════
# 라우트 — 회원가입 / 로그인 / 로그아웃
# ═══════════════════════════════════════════════════════════

@app.route('/register', methods=['GET', 'POST'])
def register():
    """회원가입: 폼 검증 후 비밀번호 해싱하여 DB에 저장한다."""
    if g.user is not None:
        return redirect(url_for('index'))
    if request.method == 'POST':
        validate_csrf()
        username         = request.form.get('username', '').strip().lower()
        name             = request.form.get('name', '').strip()
        email            = request.form.get('email', '').strip().lower()
        password         = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        gender,    gender_error    = validate_optional_profile('gender',    request.form.get('gender', ''))
        age,       age_error       = validate_optional_profile('age',       request.form.get('age', ''))
        skin_type, skin_type_error = validate_optional_profile('skin_type', request.form.get('skin_type', ''))

        error = validate_registration(username, name, email, password, password_confirm)
        error = error or gender_error or age_error or skin_type_error

        if error is None:
            try:
                db = get_db()
                db.execute(
                    '''INSERT INTO users
                       (username, name, email, password_hash, role, gender, age, skin_type, created_at)
                       VALUES (?, ?, ?, ?, 'user', ?, ?, ?, ?)''',
                    (username, name, email, generate_password_hash(password),
                     gender or None, age or None, skin_type or None, utc_now()),
                )
                db.commit()
            except sqlite3.IntegrityError:
                # UNIQUE 제약 위반: 아이디 또는 이메일 중복
                error = '이미 사용 중인 아이디 또는 이메일입니다.'
            else:
                flash('회원가입이 완료되었습니다. 로그인해 주세요.', 'success')
                return redirect(url_for('login'))
        flash(error, 'error')
    return render_template('auth.html', mode='register')


def validate_registration(username, name, email, password, password_confirm):
    """
    회원가입 입력값을 검증한다.
    오류가 있으면 첫 번째 오류 메시지 문자열을 반환하고, 정상이면 None을 반환한다.
    """
    if not re.fullmatch(r'[a-z0-9_]{4,20}', username):
        return '아이디는 영문 소문자, 숫자, 밑줄을 사용해 4~20자로 입력해 주세요.'
    if not 2 <= len(name) <= 30:
        return '이름은 2~30자로 입력해 주세요.'
    if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        return '올바른 이메일 주소를 입력해 주세요.'
    if len(password) < 8 or not re.search(r'[A-Za-z]', password) or not re.search(r'\d', password):
        return '비밀번호는 영문과 숫자를 포함해 8자 이상이어야 합니다.'
    if password != password_confirm:
        return '비밀번호 확인이 일치하지 않습니다.'
    return None


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    로그인: 아이디·비밀번호 확인 후 세션에 user_id를 저장한다.
    - session.clear(): 세션 고정 공격(Session Fixation) 방지
    - 관리자는 관리자 페이지로, 일반 사용자는 메인 페이지로 리디렉션
    """
    if g.user is not None:
        return redirect(url_for('index'))
    if request.method == 'POST':
        validate_csrf()
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        user     = get_db().execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

        if user is None or not check_password_hash(user['password_hash'], password):
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'error')
        else:
            session.clear()
            session.permanent     = True
            session['user_id']    = user['id']
            get_db().execute('UPDATE users SET last_login = ? WHERE id = ?', (utc_now(), user['id']))
            get_db().commit()
            next_url = request.args.get('next', '')
            if next_url and is_safe_redirect(next_url):
                return redirect(next_url)
            return redirect(url_for('admin_users' if user['role'] == 'admin' else 'index'))
    return render_template('auth.html', mode='login')


@app.post('/logout')
@login_required
def logout():
    """로그아웃: 세션 전체 삭제."""
    validate_csrf()
    session.clear()
    flash('로그아웃되었습니다.', 'success')
    return redirect(url_for('index'))


# ═══════════════════════════════════════════════════════════
# 라우트 — 관리자 페이지
# ═══════════════════════════════════════════════════════════

@app.route('/admin/users')
@admin_required
def admin_users():
    """
    관리자 전용 대시보드:
    - 전체 회원 목록
    - 집계 통계 (총 회원 수, 추천 횟수, 검색 횟수 등)
    - 최근 검색 이력 20건
    - 인기 검색어
    """
    users = get_db().execute(
        '''SELECT id, username, name, email, role, gender, age, skin_type,
                  recommendation_count, created_at, last_login
           FROM users ORDER BY id DESC'''
    ).fetchall()
    stats = get_db().execute(
        '''SELECT COUNT(*) AS total,
                  SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS members,
                  COALESCE(SUM(recommendation_count), 0) AS recommendations,
                  SUM(CASE WHEN last_login IS NOT NULL THEN 1 ELSE 0 END) AS active,
                  (SELECT COUNT(*) FROM search_history) AS searches
           FROM users'''
    ).fetchone()
    recent_searches = get_db().execute(
        '''SELECT search_history.id, search_history.keyword, search_history.gender,
                  search_history.age, search_history.skin_type,
                  search_history.recommended_ingredients, search_history.created_at,
                  users.name, users.username
           FROM search_history
           LEFT JOIN users ON users.id = search_history.user_id
           ORDER BY search_history.id DESC LIMIT 20'''
    ).fetchall()
    return render_template(
        'admin.html', users=users, stats=stats,
        recent_searches=recent_searches, popular_terms=get_popular_terms(),
    )


@app.post('/admin/users/<int:user_id>/delete')
@admin_required
def delete_user(user_id):
    """
    회원 삭제:
    - 자기 자신(현재 로그인한 관리자) 삭제 불가
    - 다른 관리자 계정 삭제 불가
    """
    validate_csrf()
    db   = get_db()
    user = db.execute(
        'SELECT id, username, name, role FROM users WHERE id = ?', (user_id,)
    ).fetchone()

    if user is None:
        flash('삭제할 사용자를 찾을 수 없습니다.', 'error')
        return redirect(url_for('admin_users'))
    if user['id'] == g.user['id']:
        flash('현재 로그인한 관리자 계정은 삭제할 수 없습니다.', 'error')
        return redirect(url_for('admin_users'))
    if user['role'] == 'admin':
        flash('관리자 계정은 사용자 목록에서 삭제할 수 없습니다.', 'error')
        return redirect(url_for('admin_users'))

    db.execute('DELETE FROM users WHERE id = ?', (user_id,))
    db.commit()
    flash(f"{user['name']}({user['username']}) 계정을 삭제했습니다.", 'success')
    return redirect(url_for('admin_users'))


# ═══════════════════════════════════════════════════════════
# 라우트 — 추천 API
# ═══════════════════════════════════════════════════════════

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    추천 요청을 처리하는 메인 API 엔드포인트.
    JSON 바디로 받은 키워드·프로필·모드에 따라 분기:
      - mode='review' → recommend_by_review()
      - mode='expert' → select_expert_row()로 상담 데이터 검색

    Expert Mode 세부 흐름:
      1. keyword가 없으면 프로필 필터만으로 첫 번째 행 선택
      2. keyword가 있으면 전처리 → TF-IDF 변환 → 코사인 유사도 계산
      3. 회원이면 프로필 업데이트 및 추천 횟수 +1
      4. 검색 이력 기록
    """
    data       = request.get_json(silent=True) or {}
    user_input = str(data.get('keyword', '')).strip()[:500]  # 입력 길이 500자 제한
    gender     = validated_profile_value('gender',    data.get('gender',    '성별 선택'))
    age        = validated_profile_value('age',       data.get('age',       '연령대 선택'))
    skin_type  = validated_profile_value('skin_type', data.get('skin_type', '피부 타입 선택'))
    mode       = str(data.get('mode', 'expert'))

    if mode == 'review':
        return recommend_by_review(user_input, gender, age, skin_type)
    if mode != 'expert':
        return jsonify({'error': '지원하지 않는 추천 방식입니다.'}), 400

    cleaned = ''
    if not user_input:
        # 텍스트 없이 프로필만 선택한 경우
        if all(value.endswith('선택') for value in (gender, age, skin_type)):
            return jsonify({'error': '피부 고민 또는 프로필을 하나 이상 입력해 주세요.'}), 400
        # 프로필 조건에 맞는 첫 번째 행 선택
        filtered = df_skin.copy()
        if gender    != '성별 선택':     filtered = filtered[filtered['Gender']    == gender]
        if age       != '연령대 선택':   filtered = filtered[filtered['Age']       == age]
        if skin_type != '피부 타입 선택': filtered = filtered[filtered['Skin Type'] == skin_type]
        if filtered.empty:
            return jsonify({'error': '조건에 맞는 데이터가 없습니다.'}), 404
        row = filtered.iloc[0]
    else:
        cleaned = preprocess_input(user_input)
        if not cleaned:
            return jsonify({'error': '입력하신 내용에서 유효한 단어를 찾을 수 없습니다.'}), 400
        row = select_expert_row(user_input, cleaned, gender, age, skin_type)

    # 회원 프로필 업데이트 및 추천 횟수 증가
    if g.user is not None:
        db = get_db()
        db.execute(
            '''UPDATE users SET gender = ?, age = ?, skin_type = ?,
                                recommendation_count = recommendation_count + 1
               WHERE id = ?''',
            (None if gender    == '성별 선택'     else gender,
             None if age       == '연령대 선택'   else age,
             None if skin_type == '피부 타입 선택' else skin_type,
             g.user['id']),
        )
        db.commit()

    record_search(user_input, cleaned, gender, age, skin_type, row)
    return build_response(row, user_input, gender, age, skin_type, cleaned)


def safe_product_url(value):
    """
    올리브영 도메인의 HTTPS URL만 허용한다.
    외부 URL이나 잘못된 링크가 응답에 포함되지 않도록 검증한다.
    """
    parsed = urlparse(str(value))
    if parsed.scheme == 'https' and parsed.hostname in {
        'www.oliveyoung.co.kr', 'oliveyoung.co.kr',
    }:
        return parsed.geturl()
    return None


def update_member_profile(gender, age, skin_type):
    """리뷰 모드 사용 시 회원 프로필과 추천 횟수를 업데이트한다."""
    if g.user is None:
        return
    db = get_db()
    db.execute(
        """UPDATE users SET gender = ?, age = ?, skin_type = ?,
                            recommendation_count = recommendation_count + 1
           WHERE id = ?""",
        (None if gender    == '성별 선택'     else gender,
         None if age       == '연령대 선택'   else age,
         None if skin_type == '피부 타입 선택' else skin_type,
         g.user['id']),
    )
    db.commit()


def recommend_by_review(user_input, gender, age, skin_type):
    """
    리뷰 기반 제품 추천 (Review Mode).

    사용자 입력 → 전처리 → 리뷰 TF-IDF 행렬과 코사인 유사도 계산
    → 유사도 높은 순으로 정렬 → 중복 제품 제거 → 상위 3개 반환

    반환 형식: { mode: 'review', results: [...] }
    각 결과 항목: name, brand, category, link, review_snippet, match_percent
    """
    if df_reviews is None or tfidf_reviews is None or tfidf_matrix_reviews is None:
        return jsonify({'error': '현재 리뷰 추천 데이터를 사용할 수 없습니다.'}), 503
    if not user_input:
        return jsonify({'error': '리뷰 추천에는 피부 고민 입력이 필요합니다.'}), 400

    cleaned = preprocess_input(user_input)
    if not cleaned:
        return jsonify({'error': '입력하신 내용에서 유효한 단어를 찾을 수 없습니다.'}), 400

    # 사용자 입력 벡터와 전체 리뷰 행렬 간 코사인 유사도 계산
    similarities   = linear_kernel(tfidf_reviews.transform([cleaned]), tfidf_matrix_reviews)[0]
    ranked_indices = similarities.argsort()[::-1]  # 높은 유사도 순 정렬

    results, seen_products = [], set()
    for index in ranked_indices:
        score = float(similarities[index])
        if score <= 0:
            break  # 유사도 0 이하는 관련 없음

        row          = df_reviews.iloc[int(index)]
        product_name = str(row['product_name'])
        if product_name in seen_products:
            continue  # 같은 제품 중복 제거

        product_url = safe_product_url(row['product_link'])
        if not product_url:
            continue  # 유효하지 않은 URL 제외

        # 리뷰 미리보기 (180자 이상이면 ... 처리)
        snippet = re.sub(r'\s+', ' ', str(row['review'])).strip()
        results.append({
            'name':           product_name,
            'brand':          str(row['product_brand']),
            'category':       str(row['category']),
            'link':           product_url,
            'review_snippet': snippet[:180] + ('...' if len(snippet) > 180 else ''),
            'match_percent':  round(min(score, 1.0) * 100),
        })
        seen_products.add(product_name)
        if len(results) == 3:
            break

    if not results:
        return jsonify({'error': '해당 고민과 연결되는 스킨케어 리뷰를 찾지 못했습니다.'}), 404

    update_member_profile(gender, age, skin_type)
    record_search(user_input, cleaned, gender, age, skin_type, None)
    return jsonify({'mode': 'review', 'results': results})


def find_related_products(user_input, recommended_ingredients, response_text=''):
    """
    추천 성분과 사용자 입력을 바탕으로 연관 제품을 검색한다.

    2단계 검색:
      1차 — user_input 명사 + 성분명(콤마 분리, 괄호 내 한글명 포함)
      2차 — 성분명 검색 결과 0개일 때 Makeup Response 명사로 폴백

    성분명 처리:
      "Luteolin (루테올린)" → ['Luteolin', '루테올린'] 으로 분리해 각각 검색
      str.contains(regex=False): 괄호 등 특수문자가 정규식으로 해석되는 오류 방지

    반환: 최대 6개 제품 리스트 [{brand, name, category, link}, ...]
    """
    if df_product_list is None:
        return []

    STOPWORDS = {'피부', '추출물', '사용', '도움', '성분', '제품', '바르', '얇게',
                 '발라', '선택', '타입', '방법', '경우', '이후', '이전', '해야', '있어', '없어'}

    def extract_ing_terms(ing_text):
        """
        성분 문자열을 검색 가능한 단어 리스트로 분리한다.
        "비타민C, Luteolin (루테올린), AHA/BHA" →
        ['비타민C', 'Luteolin', '루테올린', 'AHA', 'BHA']
        """
        raw = str(ing_text)
        if not raw or raw == 'nan':
            return []
        out = []
        for part in raw.split(','):
            # 괄호·슬래시로 추가 분리
            sub = re.split(r'[\(\)\[\]/]', part)
            out.extend(s.strip() for s in sub if len(s.strip()) >= 2)
        return out

    def search(terms, limit=6):
        """
        terms 목록의 단어를 순서대로 제품명·브랜드명에서 검색한다.
        결과가 6개 채워지면 즉시 반환한다.
        """
        results, seen_links = [], set()
        for term in terms[:limit]:
            matches = df_product_list[
                df_product_list['product_name'].str.contains(term, case=False, na=False, regex=False)
                | df_product_list['product_brand'].str.contains(term, case=False, na=False, regex=False)
            ]
            for _, product in matches.head(2).iterrows():
                product_url = safe_product_url(product['product_link'])
                if not product_url or product_url in seen_links:
                    continue
                results.append({
                    'brand':    str(product['product_brand']),
                    'name':     str(product['product_name']),
                    'category': str(product['category']),
                    'link':     product_url,
                })
                seen_links.add(product_url)
                if len(results) == 6:
                    return results
        return results

    # 1차: 사용자 입력 명사 + 성분명 검색
    terms = []
    if user_input:
        terms.extend(okt.nouns(user_input))
    terms.extend(extract_ing_terms(recommended_ingredients))
    terms   = [t for t in dict.fromkeys(terms) if len(t) >= 2 and t not in STOPWORDS]
    results = search(terms)
    if results:
        return results

    # 2차 폴백: Makeup Response 본문 명사로 검색
    # (성분이 제품 DB에 없을 때 — 예: '루테올린' 같은 희귀 성분)
    if response_text:
        fallback_terms = [
            w for w in okt.nouns(response_text)
            if len(w) >= 2 and w not in STOPWORDS
        ]
        results = search(fallback_terms, limit=10)

    return results


# ═══════════════════════════════════════════════════════════
# 응답 빌더 함수
# ═══════════════════════════════════════════════════════════

def build_analysis_basis(row, user_input, cleaned, gender, age, skin_type):
    """
    분석 근거(Analysis Basis) 딕셔너리를 생성한다.
    프론트엔드의 '상담 데이터 분석' 섹션에 표시되는 메타 정보를 담는다.

    포함 항목:
    - 데이터셋 정보 (이름, 행 수, 모델명)
    - 사용자가 선택한 프로필 vs 실제 매칭된 행의 프로필
    - 텍스트 유사도 점수 (%)
    - 일치하는 프로필 항목 수
    """
    profile_values = {'성별': gender, '연령대': age, '피부 타입': skin_type}

    # '선택' 기본값은 제외한 실제 선택된 프로필
    selected_profile = {
        label: value for label, value in profile_values.items()
        if not value.endswith('선택')
    }

    # 매칭된 행의 프로필 중 사용자 선택과 일치하는 항목
    profile_matches = [
        label for label, value in selected_profile.items()
        if str(row[{'성별': 'Gender', '연령대': 'Age', '피부 타입': 'Skin Type'}[label]]) == value
    ]

    # 텍스트 유사도 계산 (키워드가 있을 때만)
    similarity = None
    if cleaned:
        similarity = float(linear_kernel(
            tfidf.transform([cleaned]), tfidf_matrix[row.name],
        )[0, 0])

    return {
        'dataset_count':          len(df_skin),
        'dataset_name':           'skin_data_final.csv',
        'model_name':             'Tfidf_skin_data_rebuild',
        'method':                 'tfidf_profile' if cleaned else 'profile_filter',
        'input_keyword':          user_input or None,
        'extracted_terms':        cleaned.split() if cleaned else [],
        'selected_profile':       selected_profile,
        'matched_profile': {
            '성별':    str(row['Gender']),
            '연령대':  str(row['Age']),
            '피부 타입': str(row['Skin Type']),
        },
        'matched_question':       str(row['User Question']),
        'text_similarity_percent': round(similarity * 100, 1) if similarity is not None else None,
        'profile_matches':        profile_matches,
        'profile_bonus':          round(len(profile_matches) * 0.05, 2),
        'source_fields':          ['Makeup Response', 'Recommended Ingredients', 'Ingredients to Avoid'],
    }


def build_response(row, user_input, gender, age, skin_type, cleaned=''):
    """
    클라이언트에 반환할 최종 JSON 응답을 조립한다.

    포함 항목:
    - mode: 'expert'
    - profile: 선택된 프로필 요약 문자열
    - response: 메이크업 상담 답변 (Makeup Response)
    - rec_ingredients: 추천 성분
    - avoid_ingredients: 주의 성분
    - analysis_basis: 분석 근거 메타 정보
    - product_recommendations: 연관 제품 목록 (최대 6개)
    """
    profile_parts = [value for value in [gender, age, skin_type]
                     if value not in ('성별 선택', '연령대 선택', '피부 타입 선택')]
    return jsonify({
        'mode':    'expert',
        'profile': ', '.join(profile_parts) if profile_parts else '전체',
        'response':           row['Makeup Response'],
        'rec_ingredients':    row['Recommended Ingredients'],
        'avoid_ingredients':  row['Ingredients to Avoid'],
        'analysis_basis':     build_analysis_basis(row, user_input, cleaned, gender, age, skin_type),
        'product_recommendations': find_related_products(
            user_input, row['Recommended Ingredients'],
            response_text=str(row['Makeup Response']),
        ),
    })


# ═══════════════════════════════════════════════════════════
# 앱 시작
# ═══════════════════════════════════════════════════════════

# 앱 컨텍스트 내에서 DB 초기화 (테이블 생성 및 관리자 계정 확인)
with app.app_context():
    init_db()


if __name__ == '__main__':
    # 직접 실행 시 (개발 환경) — 운영 환경에서는 gunicorn 사용
    debug_enabled = os.environ.get('SKINSCOPE_DEBUG', '').lower() in {'1', 'true', 'yes'}
    app.run(
        host=os.environ.get('SKINSCOPE_HOST', '127.0.0.1'),
        port=int(os.environ.get('SKINSCOPE_PORT', '5006')),
        debug=debug_enabled,
    )
