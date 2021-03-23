# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:37:57 2020


"""

from gensim import models
#import pymysql
import pandas as pd
#import MeCab
#from progressbar import ProgressBar
#import time
#from pandas import Series,DataFrame
#from gensim import corpora,matutils
#from gensim.models import word2vec
#import math

#https://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/


if __name__ == '__main__':
    #model = models.doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)
    
    ####### Load model ########
    model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_deepcut_test1')
    model_loaded_attacut = models.doc2vec.Doc2Vec.load('model_attacut_test1')
    print(model_loaded_deepcut)
    print(model_loaded_attacut)
    
    
    #--- Old version for Dailynews ---#
    '''######### ! 1. USE TARGET WORD VECTOR --> Similar words ###########
    #x = model_loaded.most_similar(["Cybersecurity"]) #--- Cybersecurity ---#
    x = model_loaded.most_similar(["เอไอ"]) #--- Cybersecurity ---#
    print("USE TARGET WORD VECTOR = ", x)
    ###################################################################'''
    
    ########## Paragraph vector #############
    #vec1 = model_loaded.docvecs['it_245']
    #vec2 = model_loaded.docvecs['it_464']
    """vec3 = model_loaded.docvecs['sports_1865']
    vec4 = model_loaded.docvecs['sports_782']
    vec5 = model_loaded.docvecs['sports_1463']
    vec6 = model_loaded.docvecs['sports_1830']"""
    #vec7 = model_loaded.docvecs['it_876'] ***************
    """vec8 = model_loaded.docvecs['it_622']
    vec9 = model_loaded.docvecs['it_1116']
    vec10 = model_loaded.docvecs['it_228']
    vec11 = model_loaded.docvecs['it_270']
    vec12 = model_loaded.docvecs['education_759']"""
    
    
    #----------------------------------------------------------------------------------------#
    
    #--- Cybersecurity ---#
    ######### ! 1. USE TARGET WORD VECTOR --> Similar words ###########
    '''x = model_loaded.most_similar(["เอไอ"]) #--- AI ---#
    print("USE TARGET WORD VECTOR = ", x)
    
    vec13 = model_loaded.docvecs['it_310'] #target vector
    vec14 = model_loaded.docvecs['it_1607']
    vec15 = model_loaded.docvecs['it_308']
    vec16 = model_loaded.docvecs['it_1953']
    vec17 = model_loaded.docvecs['it_3311']
    #vec14 = model_loaded.docvecs['sports_782']
    
    #--- Find Similar paragraph vector for Feature vector ---#
    print("Find Similar paragraph vector = ", model_loaded.docvecs.most_similar(["it_310"]))
    ############################################
       
    ###### ! 2. USE TARGET PARAGRAPH VECTOR --> Similar words ######
    tasu = (vec13)
    y = model_loaded.similar_by_vector(tasu, topn=10, restrict_vocab=None)
    print("USE TARGET PARAGRAPH VECTOR = ", y)
        
    ###### ! 3. USE FEATURE VECTOR --> Similar words ####### 
    tasu1 = (vec13+vec14+vec15) #--- Cybersecurity ---#
    z = model_loaded.similar_by_vector(tasu1, topn=10, restrict_vocab=None)
    print("USE FEATURE VECTOR = ", z)
    '''
    
    #--- Black Hole ---#
    ######### ! 1. USE TARGET WORD VECTOR --> Similar words ###########
    '''x = model_loaded.most_similar(["ฟูลมูน"]) #--- Full moon ---#
    print("USE TARGET WORD VECTOR = ", x)
    
    vec13 = model_loaded.docvecs['it_2528'] #target vector
    vec14 = model_loaded.docvecs['it_2703']
    vec15 = model_loaded.docvecs['it_302']
    vec16 = model_loaded.docvecs['it_1506']
    vec17 = model_loaded.docvecs['it_2931']
    #vec14 = model_loaded.docvecs['sports_782']
    
    #--- Find Similar paragraph vector for Feature vector ---#
    print("Find Similar paragraph vector = ", model_loaded.docvecs.most_similar(["it_310"]))
    ############################################
       
    ###### ! 2. USE TARGET PARAGRAPH VECTOR --> Similar words ######
    tasu = (vec13)
    y = model_loaded.similar_by_vector(tasu, topn=10, restrict_vocab=None)
    print("USE TARGET PARAGRAPH VECTOR = ", y)
        
    ###### ! 3. USE FEATURE VECTOR --> Similar words ####### 
    tasu1 = (vec13+vec14+vec15) #--- Cybersecurity ---#
    z = model_loaded.similar_by_vector(tasu1, topn=10, restrict_vocab=None)
    print("USE FEATURE VECTOR = ", z)]'''
    
    
    #--- Test ---#
    ######### ! 1. USE TARGET WORD VECTOR --> Similar words ###########
    %time dx = model_loaded_deepcut.wv.most_similar(["หลุม"])  #--- Medicine ---#
    print("USE TARGET WORD VECTOR [DeepCut] = ", dx)
    print('')
    %time ax = model_loaded_attacut.wv.most_similar(["หลุม"])     #--- Medicine ---#
    print("USE TARGET WORD VECTOR [AttaCut] = ", ax)
    print('')
    print('-----')
    
    
    '''ddoc0 = 'entertainment_1990'
    ddoc1 = 'entertainment_1612'
    ddoc2 = 'entertainment_953'
    ddoc3 = 'entertainment_1937'
    
    adoc0 = 'entertainment_1990'
    adoc1 = 'entertainment_1612'
    adoc2 = 'entertainment_1937'
    adoc3 = 'entertainment_223' '''
    
    
    #--- Black Hole ---#  
    '''ddoc0 = 'it_1241'
    ddoc1 = 'it_314'
    ddoc2 = 'it_183'
    ddoc3 = 'it_1721'
    
    adoc0 = 'it_1241'
    adoc1 = 'it_458'
    adoc2 = 'it_314'
    adoc3 = 'it_183' '''
    #----------#
    
    
    #--- รถยนต์ไฟฟ้า ---#  
    ddoc0 = 'it_2172'
    ddoc1 = 'it_225'
    ddoc2 = 'it_378'
    ddoc3 = 'it_2376'
    ddoc4 = 'it_917'
    ddoc5 = 'it_886'
    ddoc6 = 'it_2365'
    ddoc7 = 'it_1690'
    
    adoc0 = 'it_2172'
    adoc1 = 'it_225'
    adoc2 = 'it_2376'
    adoc3 = 'it_378'
    adoc4 = 'it_917'
    adoc5 = 'it_886'
    adoc6 = 'it_923'
    adoc7 = 'it_888'
    #----------#
    
    
    dvec12 = model_loaded_deepcut.docvecs[ddoc0]
    dvec13 = model_loaded_deepcut.docvecs[ddoc1]
    dvec14 = model_loaded_deepcut.docvecs[ddoc2]
    dvec15 = model_loaded_deepcut.docvecs[ddoc3]
    dvec16 = model_loaded_deepcut.docvecs[ddoc4]
    dvec17 = model_loaded_deepcut.docvecs[ddoc5]
    dvec18 = model_loaded_deepcut.docvecs[ddoc6]
    dvec19 = model_loaded_deepcut.docvecs[ddoc7]
    
    avec12 = model_loaded_attacut.docvecs[adoc0]
    avec13 = model_loaded_attacut.docvecs[adoc1]
    avec14 = model_loaded_attacut.docvecs[adoc2]
    avec15 = model_loaded_attacut.docvecs[adoc3]
    avec16 = model_loaded_attacut.docvecs[adoc4]
    avec17 = model_loaded_attacut.docvecs[adoc5]
    avec18 = model_loaded_attacut.docvecs[adoc6]
    avec19 = model_loaded_attacut.docvecs[adoc7]
    
    
    #--- Find Similar paragraph vector for Feature vector ---#
    %time dd = model_loaded_deepcut.docvecs.most_similar([ddoc0])
  
    print("Find Similar paragraph vector [DeepCut] = ", dd)
    print('')
    %time ad = model_loaded_attacut.docvecs.most_similar([adoc0])
    print("Find Similar paragraph vector [AttaCut] = ", ad)
    print('')
    print('-----')
    ############################################
       
    ###### ! 2. USE TARGET PARAGRAPH VECTOR --> Similar words ######
    dtasu = (dvec12)
    atasu = (avec12)
    %time dy = model_loaded_deepcut.wv.similar_by_vector(dtasu, topn=10, restrict_vocab=None)
    print("USE TARGET PARAGRAPH VECTOR [DeepCut] = ", dy)
    print('')
    %time ay = model_loaded_attacut.wv.similar_by_vector(atasu, topn=10, restrict_vocab=None)
    print("USE TARGET PARAGRAPH VECTOR [AttaCut] = ", ay)
    print('')
    print('-----')
        
    ###### ! 3. USE FEATURE VECTOR --> Similar words ####### 
    #dtasu1 = (dvec12+dvec13+dvec14+dvec15+dvec16+dvec17+dvec18+dvec19)
    #atasu1 = (avec12+avec13+avec14+avec15+avec16+avec17+avec18+avec19)
    dtasu1 = (dvec12+dvec13)
    atasu1 = (avec12+avec13)
    %time dz = model_loaded_deepcut.wv.similar_by_word(dtasu1, topn=10, restrict_vocab=None)
    print("USE FEATURE VECTOR [DeepCut] = ", dz)
    print('')
    %time az = model_loaded_attacut.wv.similar_by_word(atasu1, topn=10, restrict_vocab=None)
    print("USE FEATURE VECTOR [AttaCut] = ", az)
    print('')
    print('')
    
    #note: it_627, 1044, 1241, 172* -> ดาราศาสตร์
    #it_715 -> ทีวีดิจิตอล
    #it_498 -> IoT (061)
    #it_527 -> IoT car *
    #it_529,1215 -> cyber
    #it_2307 -> รถยนต์ไฟฟ้า
    #it_2349 = it_344 -> BTS
    #it_2413,it_426 -> work home
    #it_1500 -> มัลแวร์ (All)
