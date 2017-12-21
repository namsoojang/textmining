
# coding: utf-8

# In[ ]:

'''
Insta Crwaling Analyze  / 2017.12.19 

'''

# from selenium import webdriver
# import time
# import random
# from bs4 import BeautifulSoup
# import re
from datetime import datetime
import pandas as pd
import seaborn as sns
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud
from itertools import combinations
import networkx as nx
import sys
import gensim
from gensim.models import Word2Vec

get_ipython().magic('matplotlib inline')


# In[ ]:

# 크롤링 데이터 불러오기


# In[ ]:

def load_crwal_datas(fname):

    datas = pd.read_csv(fname)
	
    return datas

# In[1]:

# 시계열 트렌드 살펴보기


# In[ ]:

def insta_trends_analyser(datas):
    # datas: insta 크롤 결과 pandas 자료형 형태
    datas['day'] = [date[:10] for date in datas['datetime']]    # #월/일 값만 추가  2017-11-25형태
    
    trends = pd.DataFrame()
    trends['article'] = datas.groupby(datas['day'])['article'].count()  #  일자별 인스타 포스팅수
    trends['replies'] = datas.groupby(datas['day'])['replys_count'].sum() # 일자별 댓글 작성수
    trends['likes'] = datas.groupby(datas['day'])['likes'].sum()     # 일자별 좋아요 수
    trends['sum'] = trends.replies + trends.article + trends.likes
    
    return trends


# In[ ]:

# 태그 빈도수 계산하기


# In[ ]:

def tag_counter(tag_lists, STOP=[]):
    # 인스타 태그 수집결과를 입력하면, 태그 빈도수 카운트가 높은 순으로 보여줌, STOP 불용어는 제외함
    
    # tag_lists = ["'#라인1태그1'",    "'라인2태그1', '라인2태그2', '라인2태그3'" ] 형식
    # 태그 빈도수 카운트
    tag_counts = Counter()
    for tags in tag_lists:
        try: tag_counts.update(tags.replace("'",'').replace(' ','').strip().split(','),)
        except: continue
    tag_counts = tag_counts.most_common()  # 정열하기
    
    # STOP 제외하기
    tag_counts_s = []
    for word, count in tag_counts:
        if word not in STOP:
            tag_counts_s.append((word,count))
    
    return tag_counts_s


# In[ ]:

def tag_counts_sellector(tag_counts, STOP):
    # STOP 단어를 제외하여  리스트 형태로 출력함(카운터 --> 리스트 형태 변경됨)
    tag_counts_s = []
    for word, count in tag_counts:
        if word not in STOP:
            tag_counts_s.append((word,count))
    return tag_counts_s


# In[ ]:




# In[ ]:

# 상위 태그 바차트로 보여주기


# In[ ]:

def tag_counts_chart(tag_counts, n):
    # matplotlib 한글 사용하기
    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    rc('font',family=font_name)
    
    labels, values = [],[]
    for tags in reversed(tag_counts[:n]):   # 가로 그래프일 경우에는 reversed 로 순서 변경해준다
        label, value = tags
        labels.append(label)
        values.append(value)

    indexes = np.arange(len(labels))
    width = 0.9
    fig = plt.figure(figsize = (8,6))
    plt.barh(indexes, values, width)  # 수평 바 그래프
#     plt.yticks(indexes + width * 0.5, labels, rotation='0')
    plt.yticks(indexes, labels, rotation='0')

    # 세로그래프일 경우
    # plt.bar(indexes, values, width)
    # plt.xticks(indexes + width * 0.5, labels, rotation='90') 
    plt.show()


# In[ ]:

# STOPWORD 제외한 태그 빈도수


# In[ ]:

# def tag_counts_sellector(tag_counts, STOP):
#     # STOP 단어를 제외하여  리스트 형태로 출력함(카운터 --> 리스트 형태 변경됨)
#     tag_counts_s = []
#     for word, count in tag_counts:
#         if word not in STOP:
#             tag_counts_s.append((word,count))
#     return tag_counts_s


# In[ ]:

# 워드클라우드 그리기


# In[ ]:

def tag_wordcloud(tag_counts, word_num):
    tmp = dict(tag_counts[:word_num])
    wordcloud=WordCloud(font_path="c:/Windows/Fonts/malgun.ttf", relative_scaling=0.5, 
                        background_color="white", max_words=100).generate_from_frequencies(tmp)    #font_step=5
    plt.figure(figsize=(8,6))
    plt.imshow(wordcloud)
    plt.axis('off')


# In[ ]:

# 게시글별 태그리스트 정리한다


# In[ ]:

def tag_lists_selector(tag_lists, STOP=[]):
    # 인스타 태그 검색결과 중 일부
    
    tag_lists_sel = []
    for tags in tag_lists:
        tag_sel = tags.replace("'",'').replace(' ','').split(',')
        tag_lists_sel.append([tag for tag in tag_sel if tag not in STOP])
    return tag_lists_sel


# In[ ]:

# 단어 관계 행렬 만들기


# In[ ]:

def word_matrix(tag_lists, stop=None, must=None):
    # 매트릭스 만들기
    word_cooc_mat=Counter()
    for line in tag_lists:
        for word1, word2 in combinations(line,2):
            if stop!=None and (word1 in stop or word2 in stop): continue
            if must!=None and (word1 not in must and word2 not in must): continue
            if word1 == word2: continue  #동일한 단어간의 벡터는 계산하지 않음
            elif word_cooc_mat[(word2,word1)]>=1:word_cooc_mat[(word2,word1)]+=1
            else: word_cooc_mat[(word1,word2)]+=1    
    
    word_coocs = []
    for words, count in word_cooc_mat.items():
        word_coocs.append((words[0],words[1],count))
    
    sorted_word_coocs = sorted(word_coocs, key=lambda x: x[2], reverse=True)   # 정렬하기
    
    return sorted_word_coocs


# In[ ]:

# 태그간 SNA 그래프


# In[ ]:

def word_sna_graph(word_matrix, n, fname=None):
    G= nx.Graph()
    for word1, word2, count in word_matrix[:n]:    #상위 n개로만 그림 그리기
        G.add_edge(word1, word2, weight=count)
    T = nx.minimum_spanning_tree(G)
    nodes = nx.nodes(T)
    degrees = nx.degree(T)
    node_size = []

    for node in nodes:
        ns = degrees[node]*200
        node_size.append(ns)

    if sys.platform in ["win32", "win64"]: font_name = "malgun gothic"
    elif sys.platform == "darwin": fornt_name = "AppleGothic"
    plt.figure(figsize=(12,10))    
    nx.draw(T,
           pos=nx.fruchterman_reingold_layout(G, k=0.5),
           node_size=node_size,
           node_color="#42FC0A",   #노란색: "#FFE27F"
            # http://www.colourlovers.com/palettes/add  에서 색상 가져올 수 있음
           font_family=font_name,
           label_pos=0, #0=head, 0.5=center, 1=tail
            with_labels=True,
            font_size=10 )
    if fname!=None: 
        plt.savefig(fname)
        print('{}에 저장하였습니다'.format(fname))
    plt.axis("off")

    plt.show()    


# In[ ]:

# 특정단어 트랜드 분석


# In[ ]:

def insta_trends_word(datas, word):
    datas['day'] = [date[:10] for date in datas['datetime']]    # #월/일 값만 추가  2017-11-25형태
    check = [word in str(tags) for tags in datas.tags]

    trends = pd.DataFrame()
    trends['article'] = datas[check].groupby(datas['day'])['article'].count()  #  일자별 인스타 포스팅수
    trends['replies'] = datas[check].groupby(datas['day'])['replys_count'].sum() # 일자별 댓글 작성수
    trends['likes'] = datas[check].groupby(datas['day'])['likes'].sum()     # 일자별 좋아요 수
    trends['sum'] = trends.replies + trends.article + trends.likes
    
    return trends


# In[ ]:




# In[ ]:

#워드임베딩 모델 생성하기


# In[ ]:

def make_word2vec(tags_raw, STOP=[]):
    tag_lists_selected = tag_lists_selector(tags_raw, STOP=STOP)  #stop 제외한 태그리스트
    model = Word2Vec(tag_lists_selected, size=100, window=10, min_count=5)
    return model


# In[ ]:

#워드임베딩 그래프로 표시하기


# In[ ]:

def word2vec_similar(model, word, topn=20):
    # matplotlib 한글 사용하기
    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    rc('font',family=font_name)
    
    labels, values = [],[]
    for tags in reversed(model.wv.most_similar(word, topn=topn)):   # 가로 그래프일 경우에는 reversed 로 순서 변경해준다
        label, value = tags
        labels.append(label)
        values.append(value)
        
    indexes = np.arange(len(labels))
    width = 0.9
    fig = plt.figure(figsize = (5,4))
    plt.barh(indexes, values, width)  # 수평 바 그래프
    plt.yticks(indexes, labels, rotation='0')
    low = min(values)
    high = max(values)
    plt.xlim([(low-1.5*(high-low)),1])
    plt.title('{}와 함께 사용된 단어 Top{}'.format(word,topn))

