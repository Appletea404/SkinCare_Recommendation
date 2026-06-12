import sys
import os
import pandas as pd
import pickle
import re
import numpy as np
from konlpy.tag import Okt
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5 import uic

# UI 파일 로드
form_window = uic.loadUiType('./skincare_recommendation.ui')[0]

class SkincareRecommendationApp(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 데이터 및 모델 로드
        print("Loading data and models...")
        try:
            self.df_skin = pd.read_csv('./datasets/skin_data.csv')
            self.tfidf_matrix = mmread('./models/Tfidf_skin_data.mtx').tocsr()
            with open('./models/tfidf.pkl', 'rb') as f:
                self.tfidf = pickle.load(f)
            print("Load complete.")
        except Exception as e:
            print(f"Error loading files: {e}")
            QMessageBox.critical(self, "오류", "데이터 파일을 로드하는 중 오류가 발생했습니다.")

        self.okt = Okt()
        
        # 불용어 로드
        try:
            df_stopwords = pd.read_csv('../movie_review/datasets/stopwords.csv')
            self.stopwords = df_stopwords['stopword'].tolist()
        except:
            self.stopwords = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게']

        # 버튼 이벤트 연결
        self.btn_recommend.clicked.connect(self.recommend_logic)
        self.le_keyword.returnPressed.connect(self.recommend_logic)

    def preprocess_input(self, text):
        text = re.sub('[^가-힣]', ' ', text)
        tokens = self.okt.pos(text, stem=True)
        words = []
        for word, pos in tokens:
            if pos in ['Noun', 'Verb', 'Adjective']:
                if len(word) > 1 and word not in self.stopwords:
                    words.append(word)
        return ' '.join(words)

    def recommend_logic(self):
        user_input = self.le_keyword.text().strip()
        gender = self.cb_gender.currentText()
        age = self.cb_age.currentText()
        skin_type = self.cb_skin_type.currentText()

        # 데이터 필터링
        filtered_df = self.df_skin.copy()
        
        if gender != "성별 선택":
            filtered_df = filtered_df[filtered_df['Gender'] == gender]
        if age != "연령대 선택":
            filtered_df = filtered_df[filtered_df['Age'] == age]
        if skin_type != "피부 타입 선택":
            filtered_df = filtered_df[filtered_df['Skin Type'] == skin_type]

        if filtered_df.empty:
            QMessageBox.information(self, "알림", "선택하신 조건에 맞는 데이터가 부족하여 전체 데이터에서 검색합니다.")
            filtered_df = self.df_skin
            filtered_indices = np.arange(len(self.df_skin))
        else:
            filtered_indices = filtered_df.index.values

        # 입력값 전처리
        if not user_input:
            # 키워드가 없으면 필터링된 데이터 중 랜덤하게 하나 추천하거나 안내
            if gender == "성별 선택" and age == "연령대 선택" and skin_type == "피부 타입 선택":
                QMessageBox.warning(self, "경고", "피부 고민을 입력하거나 조건을 선택해주세요.")
                return
            else:
                # 조건만 선택된 경우 해당 조건의 첫 번째 데이터 표시
                best_idx = filtered_indices[0]
                self.display_result(best_idx, is_filtered=True)
                return

        cleaned_input = self.preprocess_input(user_input)
        if not cleaned_input:
            self.lb_recommendation.setText("죄송합니다. 입력하신 내용에서 유효한 단어를 찾을 수 없습니다.")
            return

        # TF-IDF 변환
        input_vec = self.tfidf.transform([cleaned_input])

        # 필터링된 행렬 추출
        sub_tfidf_matrix = self.tfidf_matrix[filtered_indices]

        # 코사인 유사도 계산
        cosine_sim = linear_kernel(input_vec, sub_tfidf_matrix)

        # 가장 유사도가 높은 인덱스 찾기
        best_sub_idx = cosine_sim[0].argmax()
        best_idx = filtered_indices[best_sub_idx]
        
        self.display_result(best_idx)

    def display_result(self, idx, is_filtered=False):
        row = self.df_skin.iloc[idx]
        response = row['Makeup Response']
        rec_ingredients = row['Recommended Ingredients']
        avoid_ingredients = row['Ingredients to Avoid']
        
        profile_info = f"<small style='color: #888;'>프로필: {row['Gender']}, {row['Age']}, {row['Skin Type']}</small><br>"

        result_text = profile_info
        result_text += f"<b>[추천 솔루션]</b><br>{response}<br><br>"
        result_text += f"<b>✨ 추천 성분:</b> {rec_ingredients}<br>"
        result_text += f"<b>⚠️ 주의 성분:</b> {avoid_ingredients}"

        self.lb_recommendation.setText(result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SkincareRecommendationApp()
    window.show()
    sys.exit(app.exec_())
