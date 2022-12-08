# pip install bs4

# pip install konlpy

from konlpy.tag import Twitter

twitter = Twitter()

url='https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=205027&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page={}'

import requests 
from bs4 import BeautifulSoup

page = 1
response=requests.get(url.format(1))
response

response.text

soup = BeautifulSoup(response.text , 'html.parser')
soup.find_all('li')

for li in soup.find('div', {'class':'score_result'}).find_all('li'):
  print("점수:", li.em.text)
  print("댓글:", li.p.text)

def get_reple(page = 1):
  response = requests.get(url.format(page))
  soup = BeautifulSoup(response.text , 'html.parser')

  s, t = [],[]

  for li in soup.find('div', {'class':'score_result'}).find_all('li'):
    #print("점수:", li.em.text)
    #print("댓글:", li.p.text)
      if int(li.em.text) >= 8:
       s.append(1)
       t.append(li.p.text)
      elif int(li.em.text) <=5:
        s.append(0)
        t.append(li.p.text)

  return s, t

get_reple(1)


score, text = [], []
import time
for i in range(1,34):
  time.sleep(1)
  print(i,end='\r')
  s,t= get_reple(i)
  score +=s
  text +=t

len(score),len(text)

import pandas as pd
df=pd.DataFrame([text,score]).T
df.columns =['text','score']

df.to_csv('test.csv')

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y =train_test_split(text,score,test_size=0.2,random_state=0)

len(train_x), len(train_y)

len(test_x), len(test_y)

train_x

from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer(tokenizer=twitter.morphs, ngram_range=(1,2), min_df=3, max_df=0.9)
tfv.fit(train_x)
tfv_train_x=tfv.transform(train_x)
tfv_train_x

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

clf=LogisticRegression(random_state=0)
params={'C':[1,3,5,7,9]}
grid_cv=GridSearchCV(clf,param_grid=params, cv=4,scoring='accuracy',verbose=1)
grid_cv.fit(tfv_train_x,train_y)

grid_cv.best_params_

grid_cv.best_score_

tfv_test_x = tfv.transform(test_x)
grid_cv.best_estimator_.score(tfv_test_x,test_y)

a = ['아 너무 재밌어요 꼭 보세요','핵노잼 너무 재미없어 절대 보지마', '영화 너무 훌륭해요!','영화가 지루해요','보지마']
my_review = tfv.transform(a)
grid_cv.best_estimator_.predict(my_review)

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')

# #%config InlineBackend.figure_format='retina'

# #!apt -qq -y install fonts-nanum

# import matplotlib.font_manager as fm
# fontpath = 'C:\Users\82105\AppData\Local\Microsoft\Windows\Fonts\NanumBarunGothic.ttf'
# font = fm.FontProperties(fname=fontpath, size =10)
# plt.rc('font', family='NanumBarunGothic')
# mpl.font_manager._rebuild()

# #!set -x \
# #&& pip install konlpy \
# #&& curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -x

# #pip install pandas

# import urllib.request

# raw = urllib.request.urlopen('https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt').readlines()
# print(raw[:5])

# raw = [x.decode() for x in raw[1:]]

# reviews = []
# for i in raw:
#   reviews.append(i.split('\t')[1])

# print(reviews[:5])

# from konlpy.tag import Mecab
# tagger = Mecab()

# nouns = []
# for review in reviews:
#   for noun in tagger.nouns(review):
#     nouns.append(noun)

# nouns[:10]

# #pip install konlpy

# stop_words = "영화 전 난 일 걸 뭐 줄 만 건 분 개 끝 잼 이거 번 중 듯 때 게 내 말 나 수 거 점 것"
# stop_words = stop_words.split(' ')
# print(stop_words)

# nouns = []
# for review in reviews:
#   for noun in tagger.nouns(review):
#     if noun not in stop_words:
#       nouns.append(noun)

# nouns[:10]

# from collections import Counter

# nouns_counter = Counter(nouns)
# top_nouns = dict(nouns_counter.most_common(50))
# top_nouns

# import numpy as np

# y_pos = np.arange(len(top_nouns))

# plt.figure(figsize=(12,12))
# plt.barh(y_pos, top_nouns.values())
# plt.title('Word Count')
# plt.yticks(y_pos,top_nouns.keys())
# plt.show()

# #!pip install wordcloud

# from wordcloud import WordCloud

# wc = WordCloud(background_color='white',font_path='C:\Users\82105\AppData\Local\Microsoft\Windows\Fonts/NanumBarunGothic.ttf')
# wc.generate_from_frequencies(top_nouns)

# figure = plt.figure(figsize=(12,12))
# ax = figure.add_subplot(1,1,1)
# ax.axis('off')
# ax.imshow(wc)
# plt.show()

# #!pip install squarify

# import squarify

# norm = mpl.colors.Normalize(vmin = min(top_nouns.values()),
#                             vmax = max(top_nouns.values()))
# colors = [mpl.cm.Blues(norm(value)) for value in top_nouns.values()]

# squarify.plot(label=top_nouns.keys(),
#               sizes=top_nouns.values(),
#               color=colors,
#               alpha=.7);