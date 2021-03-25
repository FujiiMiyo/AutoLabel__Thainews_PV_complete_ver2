# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:35:35 2020


"""

import numpy as np
import pymysql
import pandas as pd
#import NLPS as nlp
from pythainlp.tokenize import word_tokenize, Tokenizer
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

conn = pymysql.connect(host = 'localhost', user = 'root', passwd = None, db = 'news', charset = 'utf8')
cur = conn.cursor()
cur.execute("USE news")

class doc_labeling(object):
    
    ####### Get data #######  
    
    def get_body_it(self,limit):
        genre = []
        body = []
        
        cur.execute("SELECT genre FROM article_it_300 WHERE genre LIKE '%it%' LIMIT " + str(limit))
        rows = cur.fetchall()
        for k in rows:
            genre.append(k)
        
        cur.execute("SELECT body FROM article_it_300 WHERE genre LIKE '%it%' LIMIT " + str(limit))
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
        
        ######## Thai word segment ######## ver1
        '''sent = text[0].replace("'","")
        word = word_tokenize(sent, engine='deepcut') # use this method
        wword = [x.replace('.',' ').replace('=',' ').replace('-',' ').replace("("," ").replace(")"," ").replace("/"," ").replace('สำหรับ',' ').replace('%',' ').strip(' ') for x in word]
        words =[]
        for w in wword:
            if w not in common.thai_stopwords():
                words = [str for str in words if str]
                words.append(w)
        return words'''
    
        ######## Thai word segment ######## ver2 -> stopwords, type of words, check spell(Eng & Thai)
        sent = text[0].replace("'","")    
        word = word_tokenize(sent, engine='deepcut') # use this method
        #wword = [x.replace('=',' ').replace('-',' ').replace("("," ").replace(")"," ").replace("/"," ").strip(' ') for x in word]
        th_no_stopwords =[]
        all_no_stopwords =[]
        th_correct_words =[]
        eng_correct_words =[]
        mix_correct_words =[]
        mix1_correct_words =[]
        all_correct_words =[]
        all_correct_words_final =[]
        check_thai_list = []
        #for tw in wword:
        for tw in word:
            if tw not in common.thai_stopwords():
                th_no_stopwords = [str for str in th_no_stopwords if str]
                th_no_stopwords.append(tw)
        #print("th_no_stopwords = ", th_no_stopwords)
        for ew in th_no_stopwords:
            if ew not in stopwords.words('english'):
                all_no_stopwords = [str for str in all_no_stopwords if str]        
                all_no_stopwords.append(ew)
        #print("all_no_stopwords = ", all_no_stopwords)
        for c in all_no_stopwords:
            thai = isthai(c)
            number = c.isnumeric()
            if not thai:
                no_num = c.isalpha()
                match1 = re.findall('\D', c) #Return ถ้าไม่พบตัวเลข 0-9 ใน string
                if no_num:
                    spell = SpellChecker()
                    eng_correct = spell.correction(c) #pn
                    eng_correct_words.append(eng_correct)
                    #print("eng = ", eng_correct)
                elif match1:
                    mix = c
                    mix_correct_words.append(mix)
                    #print("mix = ", mix)
                else:
                    num = c #No return
                    #print("num = ", num)
            elif thai:
                checker = NorvigSpellChecker(custom_dict=tnc.word_freqs()) #pn
                th_correct = checker.correct(c)
                th_correct_words.append(th_correct)
                #print("thai = ", th_correct)
              
        all_correct_words = th_correct_words + eng_correct_words + mix_correct_words
        all_correct_words = [x.replace('น.','').replace(':',' ').replace('=',' ').replace('–',' ').replace("("," ").replace(")"," ").replace("/"," ").strip(" ") for x in all_correct_words]
        all_correct_words_final = list(filter(None, all_correct_words))
        #print("words = ", all_correct_words_final)  
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
        
        %time model.train(documents, total_examples=240, epochs=10) #*** training data 80%
        
        %time model.save('model_deepcut_test2_it_300')  #*** 300 articles


if __name__ == '__main__':
        
    doclb = doc_labeling()
    e = doclb.get_body_it(1) #*** 300 articles
    
    itec = e['body'].dropna().values
    %time it_sent = doclb.LabeledLineSentence(itec,'it')
 
    
    all_list_vec = []
    all_list_vec.extend(it_sent)
    print(all_list_vec)
    with open('json_all_list_vec_deepcut_test2_it_300.txt','w', encoding='utf-8') as f:
        json.dump(all_list_vec,f,ensure_ascii=False)
    
    doc = doclb.doc_to_vec(all_list_vec)
    #print(doc)
    
    n_sent = len(all_list_vec)
    print(n_sent)