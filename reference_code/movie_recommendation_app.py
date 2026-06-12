# #앱만들기 기본
#
# import sys
# from PyQt5.QtWidgets import *
# from PyQt5 import uic
# import pandas as pd
# from sklearn.metrics.pairwise import linear_kernel
# from gensim.models import Word2Vec
# from scipy.io import mmread
# import pickle
# from PyQt5.QtCore import QStringListModel
#
# form_window = uic.loadUiType('./movie_recommendation.ui')[0]
#
#
# class Exam(QWidget, form_window):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     mainWindow = Exam()
#     mainWindow.show()
#     sys.exit(app.exec_())
# #여기까지 기본

# import sys
# from PyQt5.QtWidgets import *
# from PyQt5 import uic
# import pandas as pd
# from sklearn.metrics.pairwise import linear_kernel
# from gensim.models import Word2Vec
# from scipy.io import mmread
# import pickle
# from PyQt5.QtCore import QStringListModel, Qt
#
# form_window = uic.loadUiType('./movie_recommendation.ui')[0]
#
# class Exam(QWidget,form_window):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)
#
#         self.Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
#         with open('./models/tfidf.pkl', 'rb') as f:
#             self.Tfidf = pickle.load(f)
#         self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')
#
#         self.df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv')
#         self.titles = list(self.df_reviews.titles)
#         self.titles.sort()
#         for title in self.titles:
#             self.cb_title.addItem(title)
#
#
#         model = QStringListModel()
#         model.setStringList((self.titles))
#         completer = QCompleter()
#         completer.setModel(model)
#
#         completer.setCompletionMode(QCompleter.PopupCompletion)
#         completer.setFilterMode(Qt.MatchContains)
#
#         self.le_keyword.setCompleter(completer)
#
#         self.cb_title.currentIndexChanged.connect(self.combobox_slot)
#         self.btn_recommend.clicked.connect(self.btn_keywords_clicked)
#
#     def btn_keywords_clicked(self):
#         keyword = self.le_keyword.text()
#         if keyword in self.titles:
#             recommendations = self.recommendation_by_title(keyword)
#         else:
#             recommendations = self.recommendation_by_keyword(keyword)
#         self.lb_recommendation.setText(recommendations)
#
#     def getRecommendation(self, cosine_sim):
#         simScore = list(enumerate(cosine_sim[-1]))
#         simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
#         simScore = simScore[:11]
#         movieIdx = [i[0] for i in simScore]
#         recmovieList = self.df_reviews.iloc[movieIdx, 0]
#         return recmovieList[:11]
#
#     def combobox_slot(self):
#         title = self.cb_title.currentText()
#         # print(title)
#         recommendations = self.recommendation_by_title(title)
#         self.lb_recommendation.setText(recommendations)
#
#     def recommendation_by_title(self, title):
#         movieIdx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
#         cosine_sim = linear_kernel(self.Tfidf_matrix[movieIdx], self.Tfidf_matrix)
#         recommendations = self.getRecommendation(cosine_sim)
#         recommendations = '\n'.join(recommendations[1:])
#         return recommendations
#
#     def recommendation_by_keyword(self, keyword):
#         try:
#             sim_word = self.embedding_model.wv.most_similar(keyword, topn = 10)
#         except:
#             return '제가 모르는 단어에요 ㅠㅠ'
#         sentence = [keyword] * 11
#         count = 10
#         for word, _ in sim_word:
#             sentence = sentence + [word] * count
#             count = count - 1
#         # print(sentence)
#         sentence = ' '.join(sentence)
#         # print(sentence)
#         sentence_vec = self.Tfidf.transform([sentence])
#         cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
#         recommendations = self.getRecommendation(cosine_sim)
#         recommendations = '\n'.join(recommendations[:10])
#         return recommendations
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     mainWindow = Exam()
#     mainWindow.show()
#     sys.exit(app.exec_())

import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from scipy.io import mmread
import pickle
from PyQt5.QtCore import Qt  # 💡 Qt 임포트 확인

form_window = uic.loadUiType('./movie_recommendation.ui')[0]


class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
        with open('./models/tfidf.pkl', 'rb') as f:
            self.Tfidf = pickle.load(f)
        self.embedding_model = Word2Vec.load('./models/word2vec_movie_review.model')

        self.df_reviews = pd.read_csv('./datasets/reviews_2017_2022.csv')
        self.titles = list(self.df_reviews.titles)
        self.titles.sort()
        for title in self.titles:
            self.cb_title.addItem(title)

        # 💡 [QCompleter 대체] 수동 자동완성용 리스트 위젯 동적 생성
        self.completer_list = QListWidget(self)
        self.completer_list.hide()  # 처음엔 숨김
        # 입력창 바로 아래에 배치 (위치와 크기는 UI에 맞게 자동 조절 가능)
        self.completer_list.setGeometry(self.le_keyword.x(),
                                        self.le_keyword.y() + self.le_keyword.height(),
                                        self.le_keyword.width(), 150)

        # 💡 이벤트 시그널 연결
        self.le_keyword.textChanged.connect(self.show_completion)
        self.completer_list.itemClicked.connect(self.select_completion)

        self.cb_title.currentIndexChanged.connect(self.combobox_slot)
        self.btn_recommend.clicked.connect(self.btn_keywords_clicked)

    # 💡 [새로 추가] 글자가 입력될 때마다 실시간으로 영화 제목 필터링해서 띄우는 함수
    def show_completion(self, text):
        if not text:
            self.completer_list.hide()
            return

        # 💡 [핵심 패치] 입력된 텍스트에 완성된 한글(가~힣)이 하나라도 있는지 검사합니다.
        # 낱개 자음/모음(ㅇ, ㅁ, ㅓ)이나 영어 임시 버퍼일 때는 자동완성을 패스합니다.
        has_complete_korean = any('가' <= char <= '힣' for char in text)

        if not has_complete_korean:
            self.completer_list.hide()
            return

        # 입력한 글자가 포함된 영화 제목 필터링 (최대 7개)
        matched_titles = [title for title in self.titles if text in title][:7]

        if matched_titles:
            self.completer_list.clear()
            self.completer_list.addItems(matched_titles)
            self.completer_list.show()
            self.completer_list.raise_()
        else:
            self.completer_list.hide()

    # 💡 [새로 추가] 자동완성 리스트에서 영화를 클릭하면 입력창에 쏙 들어가게 하는 함수
    def select_completion(self, item):
        self.le_keyword.setText(item.text())
        self.completer_list.hide()

    def btn_keywords_clicked(self):
        self.completer_list.hide()  # 검색 버튼 누르면 리스트 닫기
        keyword = self.le_keyword.text()
        if keyword in self.titles:
            recommendations = self.recommendation_by_title(keyword)
        else:
            recommendations = self.recommendation_by_keyword(keyword)
        self.lb_recommendation.setText(recommendations)

    def getRecommendation(self, cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))
        simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
        simScore = simScore[:11]
        movieIdx = [i[0] for i in simScore]
        recmovieList = self.df_reviews.iloc[movieIdx, 0]
        return recmovieList[:11]

    def combobox_slot(self):
        title = self.cb_title.currentText()
        recommendations = self.recommendation_by_title(title)
        self.lb_recommendation.setText(recommendations)

    def recommendation_by_title(self, title):
        movieIdx = self.df_reviews[self.df_reviews['titles'] == title].index[0]
        cosine_sim = linear_kernel(self.Tfidf_matrix[movieIdx], self.Tfidf_matrix)
        recommendations = self.getRecommendation(cosine_sim)
        recommendations = '\n'.join(recommendations[1:])
        return recommendations

    def recommendation_by_keyword(self, keyword):
        try:
            if keyword not in self.embedding_model.wv.key_to_index:
                return '제가 모르는 단어에요 ㅠㅠ'
            sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)
        except:
            return '제가 모르는 단어에요 ㅠㅠ'
        sentence = [keyword] * 11
        count = 10
        for word, _ in sim_word:
            sentence = sentence + [word] * count
            count = count - 1
        sentence = ' '.join(sentence)
        sentence_vec = self.Tfidf.transform([sentence])
        cosine_sim = linear_kernel(sentence_vec, self.Tfidf_matrix)
        recommendations = self.getRecommendation(cosine_sim)
        recommendations = '\n'.join(recommendations[:10])
        return recommendations


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())