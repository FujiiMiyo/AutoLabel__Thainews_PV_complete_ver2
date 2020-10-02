# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 07:36:22 2020


"""

import numpy as np
import pymysql
import pandas as pd
#import NLPS as nlp
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import common
from pandas import Series,DataFrame
#import time
from gensim import models
from gensim import corpora,matutils
from gensim.models import word2vec
import json

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import re

from pythainlp.util import isthai
from pythainlp.util import num_to_thaiword
from pythainlp.tokenize import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.spell import NorvigSpellChecker
from pythainlp.corpus import tnc

from pythainlp.spell import *

conn = pymysql.connect(host = 'localhost', user = 'root', passwd = None, db = 'news', charset = 'utf8')
cur = conn.cursor()
cur.execute("USE news")

class doc_labeling(object):
    
    ####### Get data #######
    def get_body_eco(self,limit):
        genre = []
        body = []
        
        cur.execute("SELECT genre FROM article WHERE genre LIKE '%economic%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            genre.append(k)
        
        cur.execute("SELECT body FROM article WHERE genre LIKE '%economic%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            body.append(k)
        #print(body)
        
        df = pd.DataFrame({'genre':genre,'body':body})
        #print(df)
        return df
    
    
    def get_body_edu(self,limit):
        genre = []
        body = []
        
        cur.execute("SELECT genre FROM article WHERE genre LIKE '%education%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            genre.append(k)
        
        cur.execute("SELECT body FROM article WHERE genre LIKE '%education%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            body.append(k)
        #print(body)
        
        df = pd.DataFrame({'genre':genre,'body':body})
        #print(df)
        return df
    
    
    def get_body_ent(self,limit):
        genre = []
        body = []
        
        cur.execute("SELECT genre FROM article WHERE genre LIKE '%entertainment%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            genre.append(k)
        
        cur.execute("SELECT body FROM article WHERE genre LIKE '%entertainment%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            body.append(k)
        #print(body)
        
        df = pd.DataFrame({'genre':genre,'body':body})
        #print(df)
        return df
    
    
    def get_body_fore(self,limit):
        genre = []
        body = []
        
        cur.execute("SELECT genre FROM article WHERE genre LIKE '%foreign%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            genre.append(k)
        
        cur.execute("SELECT body FROM article WHERE genre LIKE '%foreign%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            body.append(k)
        #print(body)
        
        df = pd.DataFrame({'genre':genre,'body':body})
        #print(df)
        return df
    
    
    def get_body_it(self,limit):
        genre = []
        body = []
        
        cur.execute("SELECT genre FROM article WHERE genre LIKE '%it%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            genre.append(k)
        
        cur.execute("SELECT body FROM article WHERE genre LIKE '%it%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            body.append(k)
        #print(body)
        
        df = pd.DataFrame({'genre':genre,'body':body})
        #print(df)
        return df
    
    
    def get_body_spo(self,limit):
        genre = []
        body = []
        
        cur.execute("SELECT genre FROM article WHERE genre LIKE '%sports%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            genre.append(k)
        
        cur.execute("SELECT body FROM article WHERE genre LIKE '%sports%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            body.append(k)
        #print(body)
        
        df = pd.DataFrame({'genre':genre,'body':body})
        #print(df)
        return df
    
    
    
    ####### Labeling #######
    def LabeledLineSentence(self,documents,label_name):
        sentences = []
        
        for uid,line in enumerate(np.ndenumerate(documents)):
            #print(line[1])
            doc = self.Tokenize_word(line[1])
            mod = models.doc2vec.LabeledSentence(words = doc, tags = ['%s_%s' %(label_name,uid)])
            sentences.append(mod)
            #print(doc,['%s_%s' %(label_name,uid)])
        return sentences
    
    
    
    ####### Word segmention #######    
    def Tokenize_word(self,text):
        
        ######## Thai word segment ########
        ''''sent = text[0].replace("'","")
        word = word_tokenize(sent, engine='deepcut') # use this method
        wword = [x.replace('.',' ').replace('=',' ').replace('-',' ').replace("("," ").replace(")"," ").replace("/"," ").replace('สำหรับ',' ').replace('%',' ').strip(' ') for x in word]
        words =[]
        for w in wword:
            if w not in common.thai_stopwords():
                words = [str for str in words if str]
                words.append(w)
        return words'''
    
        ######## Thai word segment ######## ver.2 -> stopwords, type of words
        sent = text[0].replace("'","")
        word = word_tokenize(sent, engine='deepcut') # use this method
        wword = [x.replace('.',' ').replace('%',' ').replace('=',' ').replace('-',' ').replace("("," ").replace(")"," ").replace("/"," ").strip(' ') for x in word]
        th_no_stopwords =[]
        eng_no_stopwords =[]
        th_correct_words =[]
        eng_correct_words =[]
        mix_correct_words =[]
        mix1_correct_words =[]
        all_correct_words =[]
        all_correct_words_final =[]
        check_thai_list = []
        for w in wword:
            thai = isthai(w)
            #number = c.isnumeric()
            if thai:
                if w not in common.thai_stopwords():
                    #th_no_stopwords = [str for str in th_no_stopwords if str]        
                    th_no_stopwords.append(w)
                    #print("thai = ", th_correct)
            elif not thai:
                if w not in stopwords.words('english'):
                    #eng_no_stopwords = [str for str in eng_no_stopwords if str]        
                    #eng_no_stopwords.append(w)
                    no_num = w.isalpha()
                    match1 = re.findall('\D', w) #Return ถ้าไม่พบตัวเลข 0-9 ใน string
                    if no_num:
                        eng = w
                        eng_no_stopwords.append(eng)
                        #print("eng = ", eng_correct)
                    elif match1:
                        mix = w
                        mix_correct_words.append(mix)
                        #print("mix = ", mix)
                    else:
                        num = w #No return
                        #print("num = ", num)
            

        #print("th_correct_words = ", th_correct_words)
        #print("eng_correct_stopwords = ", eng_correct_words)
        
        all_correct_words = th_no_stopwords + eng_no_stopwords + mix_correct_words
        all_correct_words = [x.replace('น.','').replace(':',' ').replace('=',' ').replace('–',' ').replace("("," ").replace(")"," ").replace("/"," ").strip(" ") for x in all_correct_words]
        all_correct_words_final = list(filter(None, all_correct_words))
        #print("words = ", all_correct_words)
        return all_correct_words_final
    
        
        ######## Eng word segment ########
        '''word = text[0]
        words = []
        for i in word.split(' '):
            words = [str for str in words if str]
            words.append(i)
        return words'''



    ####### Doc2vec ####### 
    def doc_to_vec(self,documents):
        model = models.doc2vec.Doc2Vec(dm = 1, alpha=0.025, min_alpha=0.025) #use fixed learning rate
        model.build_vocab(documents)
        
        '''
        for epoch in range(10):
            model.train(documents, total_examples=10000, epochs=10)
            model.alpha -= 0.002 #decrease the learning rate
            model.min_alpha = model.alpha #fix the learning rate, no dacay            
        '''
        
        %time model.train(documents, total_examples=12000, epochs=10) #*** training data 80%
        
        %time model.save('model_deepcut_test1')  #*** 14,880 articles


if __name__ == '__main__':
        
    doclb = doc_labeling()
    a = doclb.get_body_eco(2480) #*** 2,480 articles
    b = doclb.get_body_edu(2480) #*** 2,480 articles
    c = doclb.get_body_ent(2480) #*** 2,480 articles
    d = doclb.get_body_fore(2480) #*** 2,480 articles
    e = doclb.get_body_it(2480) #*** 2,480 articles
    f = doclb.get_body_spo(2480) #*** 2,480 articles
    
    
    
    eco = a['body'].dropna().values
    %time eco_sent = doclb.LabeledLineSentence(eco,'economic') 
    edu = b['body'].dropna().values
    %time edu_sent = doclb.LabeledLineSentence(edu,'education')
    ent = c['body'].dropna().values
    %time ent_sent = doclb.LabeledLineSentence(ent,'entertainment')
    fore = d['body'].dropna().values
    %time fore_sent = doclb.LabeledLineSentence(fore,'foreign')
    itec = e['body'].dropna().values
    %time it_sent = doclb.LabeledLineSentence(itec,'it')
    spo = f['body'].dropna().values
    %time spo_sent = doclb.LabeledLineSentence(spo,'sports')
 
    
    all_list_vec = []
    all_list_vec.extend(eco_sent)
    all_list_vec.extend(edu_sent)
    all_list_vec.extend(ent_sent)
    all_list_vec.extend(fore_sent)
    all_list_vec.extend(it_sent)
    all_list_vec.extend(spo_sent)
    print(all_list_vec)
    with open('json_all_list_vec_deepcut_test1.txt','w', encoding='utf-8') as f:
        json.dump(all_list_vec,f,ensure_ascii=False)
    
    doc = doclb.doc_to_vec(all_list_vec)
    #print(doc)
    
    n_sent = len(all_list_vec)
    print(n_sent)