import pandas as pd
from konlpy.tag import Okt
import re
df = pd.read_csv('datasets/reviews_2017_2022.csv')
df.info()

df_stopwords = pd.read_csv('datasets/stopwords.csv')
stopwords = df_stopwords['stopword'].tolist()
#추가
stopwords = stopwords + ['가다', '감독', '연출', '연기', '배우', '하다', '모르다', '보여주다', '주연', '많다', '좋다']

okt = Okt()
print(df.titles[0])
print(df.reviews[0])

cleaned_sentences = []
# for review in df.reviews:
for review in df.reviews[:5]:
    review = re.sub('[^가-힣]', ' ', review)
    tokened_review = okt.pos(review, stem=True)
    df_token = pd.DataFrame(tokened_review, columns=['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb')  |
                        (df_token['class'] == 'Adjective')]
    words = []
    for word in df_token['word']:
        if len(word) > 1:
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
# df.reviews = cleaned_sentences
df.reviews[:5] = cleaned_sentences
df.dropna(inplace=True)
df.info()
df.to_csv('datasets/reviews_2017_2022_test_5.csv', index=False)