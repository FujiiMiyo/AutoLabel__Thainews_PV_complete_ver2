# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:53:09 2021


"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:27:35 2020


"""

from gensim import models ### use
from sklearn.manifold import TSNE ### use

import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure
from bokeh.io import show, output_notebook

import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt ### use
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D ### use
from sklearn import manifold


#http://csmoon-ml.com/index.php/2019/02/15/tutorial-doc2vec-and-t-sne/
#https://mlexplained.com/2018/09/14/paper-dissected-visualizing-data-using-t-sne-explained/
######https://leightley.com/visualizing-tweets-with-word2vec-and-t-sne-in-python/


class visaulization_vector_space_model(object):        
    
    def pattern_docs(self,genre_g,no_d,n):
        patterns = genre_g + '_' + str(no_d)
        ddoc_vect = model_loaded_deepcut.docvecs[patterns]
        adoc_vect = model_loaded_attacut.docvecs[patterns]
        dsim_words_cos = model_loaded_deepcut.wv.similar_by_vector(ddoc_vect, topn=n, restrict_vocab=None)
        asim_words_cos = model_loaded_attacut.wv.similar_by_vector(adoc_vect, topn=n, restrict_vocab=None)            
        return patterns, ddoc_vect, adoc_vect, dsim_words_cos, asim_words_cos #, dfirst_match, afirst_match, sim_dword_vect, sim_aword_vect
    
    def list_doc_id(self, list_genre,limit_docs,n):
        list_doc_id0 = []
        list_doc_id1 = []
        list_doc_id2 = []
        list_doc_id3 = []
        list_doc_id4 = []
        list_doc_id5 = []
        for g, lgenre in enumerate(list_genre):
            genre_g = list_genre[g]
            for no_d in range(limit_docs):
                #print(no_d)
                pattern_docs = vec_model.pattern_docs(genre_g, no_d,n)
                if lgenre == list_genre[0]:
                    list_doc_id0.append(pattern_docs[0])
                elif lgenre == list_genre[1]:
                    list_doc_id1.append(pattern_docs[0])
                elif lgenre == list_genre[2]:
                    list_doc_id2.append(pattern_docs[0])
                elif lgenre == list_genre[3]:
                    list_doc_id3.append(pattern_docs[0])
                elif lgenre == list_genre[4]:
                    list_doc_id4.append(pattern_docs[0])
                elif lgenre == list_genre[5]:
                    list_doc_id5.append(pattern_docs[0])
        #print('list_doc_id0 = ', list_doc_id0)
        #print('list_doc_id1 = ', list_doc_id1)
        #print('list_doc_id2 = ', list_doc_id2)
        #print('list_doc_id3 = ', list_doc_id3)
        #print('list_doc_id4 = ', list_doc_id4)
        #print('list_doc_id5 = ', list_doc_id5)
        return list_doc_id0,list_doc_id1,list_doc_id2,list_doc_id3,list_doc_id4,list_doc_id5    
        
    def list_ddoc_vec(self, list_genre,limit_docs,n):
        list_ddoc_vec0 = []
        list_ddoc_vec1 = []
        list_ddoc_vec2 = []
        list_ddoc_vec3 = []
        list_ddoc_vec4 = []
        list_ddoc_vec5 = []
            
        for g, lgenre in enumerate(list_genre):
            genre_g = list_genre[g]
            for no_d in range(limit_docs):
                #print(no_d)
                pattern_docs = vec_model.pattern_docs(genre_g, no_d,n)
                if lgenre == list_genre[0]:
                    list_ddoc_vec0.append(pattern_docs[1])
                elif lgenre == list_genre[1]:
                    list_ddoc_vec1.append(pattern_docs[1])
                elif lgenre == list_genre[2]:
                    list_ddoc_vec2.append(pattern_docs[1])
                elif lgenre == list_genre[3]:
                    list_ddoc_vec3.append(pattern_docs[1])
                elif lgenre == list_genre[4]:
                    list_ddoc_vec4.append(pattern_docs[1])
                elif lgenre == list_genre[5]:
                    list_ddoc_vec5.append(pattern_docs[1])
        #print('list_ddoc_vec0 = ', list_ddoc_vec0)
        #print('list_ddoc_vec1 = ', list_ddoc_vec1)
        #print('list_ddoc_vec2 = ', list_ddoc_vec2)
        #print('list_ddoc_vec3 = ', list_ddoc_vec3)
        #print('list_ddoc_vec4 = ', list_ddoc_vec4)
        #print('list_ddoc_vec5 = ', list_ddoc_vec5)
        return list_ddoc_vec0,list_ddoc_vec1,list_ddoc_vec2,list_ddoc_vec3,list_ddoc_vec4,list_ddoc_vec5
    
    def list_adoc_vec(self,list_genre,limit_docs,n):
        list_adoc_vec0 = []
        list_adoc_vec1 = []
        list_adoc_vec2 = []
        list_adoc_vec3 = []
        list_adoc_vec4 = []
        list_adoc_vec5 = []
            
        for g, lgenre in enumerate(list_genre):
            genre_g = list_genre[g]
            for no_d in range(limit_docs):
                #print(no_d)
                pattern_docs = vec_model.pattern_docs(genre_g, no_d,n)
                if lgenre == list_genre[0]:
                    list_adoc_vec0.append(pattern_docs[2])
                elif lgenre == list_genre[1]:
                    list_adoc_vec1.append(pattern_docs[2])
                elif lgenre == list_genre[2]:
                    list_adoc_vec2.append(pattern_docs[2])
                elif lgenre == list_genre[3]:
                    list_adoc_vec3.append(pattern_docs[2])
                elif lgenre == list_genre[4]:
                    list_adoc_vec4.append(pattern_docs[2])
                elif lgenre == list_genre[5]:
                    list_adoc_vec5.append(pattern_docs[2])
        #print('list_adoc_vec0 = ', list_adoc_vec0)
        #print('list_adoc_vec1 = ', list_adoc_vec1)
        #print('list_adoc_vec2 = ', list_adoc_vec2)
        #print('list_adoc_vec3 = ', list_adoc_vec3)
        #print('list_adoc_vec4 = ', list_adoc_vec4)
        #print('list_adoc_vec5 = ', list_adoc_vec5)
        return list_adoc_vec0,list_adoc_vec1,list_adoc_vec2,list_adoc_vec3,list_adoc_vec4,list_adoc_vec5
    
    def list_dword_sim(self,list_genre,limit_docs,n):
        list_dword_sim0 = []
        list_dword_sim1 = []
        list_dword_sim2 = []
        list_dword_sim3 = []
        list_dword_sim4 = []
        list_dword_sim5 = []
            
        for g, lgenre in enumerate(list_genre):
            genre_g = list_genre[g]
            for no_d in range(limit_docs):
                #print(no_d)
                pattern_docs = vec_model.pattern_docs(genre_g, no_d,n)
                if lgenre == list_genre[0]:
                    list_dword_sim0.append(pattern_docs[3])
                elif lgenre == list_genre[1]:
                    list_dword_sim1.append(pattern_docs[3])
                elif lgenre == list_genre[2]:
                    list_dword_sim2.append(pattern_docs[3])
                elif lgenre == list_genre[3]:
                    list_dword_sim3.append(pattern_docs[3])
                elif lgenre == list_genre[4]:
                    list_dword_sim4.append(pattern_docs[3])
                elif lgenre == list_genre[5]:
                    list_dword_sim5.append(pattern_docs[3])
        #print('list_dword_sim0 = ', list_ddoc_vec0)
        #print('list_dword_sim1 = ', list_ddoc_vec1)
        #print('list_dword_sim2 = ', list_ddoc_vec2)
        #print('list_dword_sim3 = ', list_ddoc_vec3)
        #print('list_dword_sim4 = ', list_ddoc_vec4)
        #print('list_dword_sim5 = ', list_ddoc_vec5)
        return list_dword_sim0,list_dword_sim1,list_dword_sim2,list_dword_sim3,list_dword_sim4,list_dword_sim5
    
    def list_aword_sim(self,list_genre,limit_docs,n):
        list_aword_sim0 = []
        list_aword_sim1 = []
        list_aword_sim2 = []
        list_aword_sim3 = []
        list_aword_sim4 = []
        list_aword_sim5 = []
            
        for g, lgenre in enumerate(list_genre):
            genre_g = list_genre[g]
            for no_d in range(limit_docs):
                #print(no_d)
                pattern_docs = vec_model.pattern_docs(genre_g, no_d,n)
                if lgenre == list_genre[0]:
                    list_aword_sim0.append(pattern_docs[4])
                elif lgenre == list_genre[1]:
                    list_aword_sim1.append(pattern_docs[4])
                elif lgenre == list_genre[2]:
                    list_aword_sim2.append(pattern_docs[4])
                elif lgenre == list_genre[3]:
                    list_aword_sim3.append(pattern_docs[4])
                elif lgenre == list_genre[4]:
                    list_aword_sim4.append(pattern_docs[4])
                elif lgenre == list_genre[5]:
                    list_aword_sim5.append(pattern_docs[4])
        #print('list_aword_sim0 = ', list_adoc_vec0)
        #print('list_aword_sim1 = ', list_adoc_vec1)
        #print('list_aword_sim2 = ', list_adoc_vec2)
        #print('list_aword_sim3 = ', list_adoc_vec3)
        #print('list_aword_sim4 = ', list_adoc_vec4)
        #print('list_aword_sim5 = ', list_adoc_vec5)
        return list_aword_sim0,list_aword_sim1,list_aword_sim2,list_aword_sim3,list_aword_sim4,list_aword_sim5

    def list_dwords_sim_vec(self,each_list_dword_sim):
        dwords_sim = []
        for c, no_doc in enumerate(each_list_dword_sim):
            #print(c)
            for d, no_word in enumerate(no_doc):
                print(d)
                for a, words in enumerate(no_word):
                    most_similar_key, similarity = no_doc[d][a]
                    first_match = most_similar_key
                    dwords_sim.append(first_match)
        dwords_sim_vec = []
        for b, sim_dword in enumerate(dwords_sim):
            sim_dword_vec = model_loaded_deepcut.wv[dwords_sim[b]]
            dwords_sim_vec.append(sim_dword_vec)
        return dwords_sim,dwords_sim_vec
    
    def list_awords_sim_vec(self,each_list_aword_sim):
        awords_sim = []
        for c, no_doc in enumerate(each_list_aword_sim):
            #print(c)
            for d, no_word in enumerate(no_doc):
                print(d)
                for a, words in enumerate(no_word):
                    most_similar_key, similarity = no_doc[d][a]
                    first_match = most_similar_key
                    awords_sim.append(first_match)
        awords_sim_vec = []
        for b, sim_aword in enumerate(awords_sim):
            sim_aword_vec = model_loaded_attacut.wv[awords_sim[b]]
            awords_sim_vec.append(sim_aword_vec)
        return awords_sim,awords_sim_vec
    
    def all_list_docs_id(self,each_list_doc_id):
        doc_all = each_list_doc_id[0] + each_list_doc_id[1] + each_list_doc_id[2] + each_list_doc_id[3] + each_list_doc_id[4] + each_list_doc_id[5]
        return doc_all
    
    def all_list_ddocs_vec(self,each_list_ddoc_vec):
        ddoc_all_vec = each_list_ddoc_vec[0] + each_list_ddoc_vec[1] + each_list_ddoc_vec[2] + each_list_ddoc_vec[3] + each_list_ddoc_vec[4] + each_list_ddoc_vec[5]
        #print(d_all_vectors)
        return ddoc_all_vec
        
    def all_list_adocs_vec(self,each_list_adoc_vec):
        adoc_all_vec = each_list_adoc_vec[0] + each_list_adoc_vec[1] + each_list_adoc_vec[2] + each_list_adoc_vec[3] + each_list_adoc_vec[4] + each_list_adoc_vec[5]
        #print(a_all_vectors)
        return adoc_all_vec   
    
    def plot_list_dvec(self,all_list_doc_id,list_dwords_sim_id,dresult_X_all,p,limit_docs):
        for i, word in enumerate(list_dwords_sim_id):
            if i < p:
                #word_eco = word
                #print("Deepcut Economic word: ", word_eco)
                scatter6 = ax0.scatter3D(dresult_X_all[i, 0], dresult_X_all[i, 1], dresult_X_all[i, 2], c='#94ebd8', label=list_genre_plot[0], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
                #plt.annotate(doc, xy=(dresult_X_all[j, 0], dresult_X_all[j, 1]))
            elif i >= p and i < (p*2):
                #word_edu = word
                #print("Deepcut Education word: ", word_edu)
                scatter7 = ax0.scatter3D(dresult_X_all[i, 0], dresult_X_all[i, 1], dresult_X_all[i, 2], c='#fff8cd', label=list_genre_plot[1], alpha=1.0, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*2) and i < (p*3):
                #word_ent = word
                #print("Deepcut Entertainment word: ", word_ent)
                scatter8 = ax0.scatter3D(dresult_X_all[i, 0], dresult_X_all[i, 1], dresult_X_all[i, 2], c='#ffd5cd', label=list_genre_plot[2], alpha=0.6, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*3) and i < (p*4):
                #word_fore = word
                #print("Deepcut Foreign word: ", word_fore)
                scatter9 = ax0.scatter3D(dresult_X_all[i, 0], dresult_X_all[i, 1], dresult_X_all[i, 2], c='#d88081', label=list_genre_plot[3], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*4) and i < (p*5):
                #word_it = word
                #print("Deepcut IT word: ", word_it)
                scatter10 = ax0.scatter3D(dresult_X_all[i, 0], dresult_X_all[i, 1], dresult_X_all[i, 2], c='#f3bad6', label=list_genre_plot[4], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*5) and i < (p*6):
                #word_spo = word
                #print("Deepcut Sports word: ", word_spo)
                scatter11 = ax0.scatter3D(dresult_X_all[i, 0], dresult_X_all[i, 1], dresult_X_all[i, 2], c='#9885b2', label=list_genre_plot[5], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
                #j = j + 1 
    
        for j, doc in enumerate(all_list_doc_id):
            if j < limit_docs:
                #doc_eco = doc
                #print("Deepcut Economic: ", doc_eco)
                scatter0 = ax0.scatter3D(dresult_X_all[j, 0], dresult_X_all[j, 1], dresult_X_all[j, 2], c='#00787a', label=list_genre_plot[0], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)  
                #plt.annotate(doc, xy=(aresult_X_all[j, 0], aresult_X_all[j, 1]))
            elif j >= limit_docs and j < limit_docs*2:
                #doc_edu = doc
                #print("Deepcut Education: ", doc_edu)
                scatter1 = ax0.scatter3D(dresult_X_all[j, 0], dresult_X_all[j, 1], dresult_X_all[j, 2], c='#fbc11a', label=list_genre_plot[1], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*2 and j < limit_docs*3:
                #doc_ent = doc
                #print("Deepcut Entertainment: ", doc_ent)
                scatter2 = ax0.scatter3D(dresult_X_all[j, 0], dresult_X_all[j, 1], dresult_X_all[j, 2], c='#ff6334', label=list_genre_plot[2], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*3 and j < limit_docs*4:
                #doc_fore = doc
                #print("Deepcut Foreign: ", doc_fore)
                scatter3 = ax0.scatter3D(dresult_X_all[j, 0], dresult_X_all[j, 1], dresult_X_all[j, 2], c='#981826', label=list_genre_plot[3], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*4 and j < limit_docs*5:
                #doc_it = doc
                #print("Deepcut IT: ", doc_it)
                scatter4 = ax0.scatter3D(dresult_X_all[j, 0], dresult_X_all[j, 1], dresult_X_all[j, 2], c='#cc0e74', label=list_genre_plot[4], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*5 and j < limit_docs*6:
                #doc_spo = doc
                #print("Deepcut Sports: ", doc_spo)
                scatter5 = ax0.scatter3D(dresult_X_all[j, 0], dresult_X_all[j, 1], dresult_X_all[j, 2], c='#821752', label=list_genre_plot[5], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
                #j = j + 1
        return scatter0,scatter1,scatter2,scatter3,scatter4,scatter5,scatter6,scatter7,scatter8,scatter9,scatter10,scatter11
    
    def plot_list_avec(self,all_list_doc_id,list_awords_sim_id,aresult_X_all,p,limit_docs):
        for i, word in enumerate(list_awords_sim_id):
            if i < p:
                #word_eco = word
                #print("Deepcut Economic word: ", word_eco)
                scatter6 = ay0.scatter3D(aresult_X_all[i, 0], aresult_X_all[i, 1], aresult_X_all[i, 2], c='#94ebd8', label=list_genre_plot[0], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
                #plt.annotate(doc, xy=(aresult_X_all[j, 0], aresult_X_all[j, 1]))
            elif i >= p and i < (p*2):
                #word_edu = word
                #print("Deepcut Education word: ", word_edu)
                scatter7 = ay0.scatter3D(aresult_X_all[i, 0], aresult_X_all[i, 1], aresult_X_all[i, 2], c='#fff8cd', label=list_genre_plot[1], alpha=1.0, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*2) and i < (p*3):
                #word_ent = word
                #print("Deepcut Entertainment word: ", word_ent)
                scatter8 = ay0.scatter3D(aresult_X_all[i, 0], aresult_X_all[i, 1], aresult_X_all[i, 2], c='#ffd5cd', label=list_genre_plot[2], alpha=0.6, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*3) and i < (p*4):
                #word_fore = word
                #print("Deepcut Foreign word: ", word_fore)
                scatter9 = ay0.scatter3D(aresult_X_all[i, 0], aresult_X_all[i, 1], aresult_X_all[i, 2], c='#d88081', label=list_genre_plot[3], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*4) and i < (p*5):
                #word_it = word
                #print("Deepcut IT word: ", word_it)
                scatter10 = ay0.scatter3D(aresult_X_all[i, 0], aresult_X_all[i, 1], aresult_X_all[i, 2], c='#f3bad6', label=list_genre_plot[4], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
            elif i >= (p*5) and i < (p*6):
                #word_spo = word
                #print("Deepcut Sports word: ", word_spo)
                scatter11 = ay0.scatter3D(aresult_X_all[i, 0], aresult_X_all[i, 1], aresult_X_all[i, 2], c='#9885b2', label=list_genre_plot[5], alpha=0.5, edgecolors='none', marker='>', depthshade=False, s=80)
                #j = j + 1 
    
        for j, doc in enumerate(all_list_doc_id):
            if j < limit_docs:
                #doc_eco = doc
                #print("Deepcut Economic: ", doc_eco)
                scatter0 = ay0.scatter3D(aresult_X_all[j, 0], aresult_X_all[j, 1], aresult_X_all[j, 2], c='#00787a', label=list_genre_plot[0], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)  
                #plt.annotate(doc, xy=(aresult_X_all[j, 0], aresult_X_all[j, 1]))
            elif j >= limit_docs and j < limit_docs*2:
                #doc_edu = doc
                #print("Deepcut Education: ", doc_edu)
                scatter1 = ay0.scatter3D(aresult_X_all[j, 0], aresult_X_all[j, 1], aresult_X_all[j, 2], c='#fbc11a', label=list_genre_plot[1], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*2 and j < limit_docs*3:
                #doc_ent = doc
                #print("Deepcut Entertainment: ", doc_ent)
                scatter2 = ay0.scatter3D(aresult_X_all[j, 0], aresult_X_all[j, 1], aresult_X_all[j, 2], c='#ff6334', label=list_genre_plot[2], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*3 and j < limit_docs*4:
                #doc_fore = doc
                #print("Deepcut Foreign: ", doc_fore)
                scatter3 = ay0.scatter3D(aresult_X_all[j, 0], aresult_X_all[j, 1], aresult_X_all[j, 2], c='#981826', label=list_genre_plot[3], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*4 and j < limit_docs*5:
                #doc_it = doc
                #print("Deepcut IT: ", doc_it)
                scatter4 = ay0.scatter3D(aresult_X_all[j, 0], aresult_X_all[j, 1], aresult_X_all[j, 2], c='#cc0e74', label=list_genre_plot[4], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
            elif j >= limit_docs*5 and j < limit_docs*6:
                #doc_spo = doc
                #print("Deepcut Sports: ", doc_spo)
                scatter5 = ay0.scatter3D(aresult_X_all[j, 0], aresult_X_all[j, 1], aresult_X_all[j, 2], c='#821752', label=list_genre_plot[5], alpha=0.8, edgecolors='none', marker='o', depthshade=False, s=110)
                #j = j + 1
        return scatter0,scatter1,scatter2,scatter3,scatter4,scatter5,scatter6,scatter7,scatter8,scatter9,scatter10,scatter11
     

if __name__ == '__main__':
    
    #0 : Setting
    vec_model = visaulization_vector_space_model()
    count_gerne = 6
    n = 10 ##Note: Count Similar words
    m = 5 ##Note: Count Similar paragraph(s)
    o = 10 ##Note: Count Similar words -> plot
    limit_docs = 10 ##Note: Number of each Categories of News Articles
    list_genre = ["economic", "education", "entertainment", "foreign", "it", "sports"] #====> Setting
    p = n*limit_docs
    #P.S. n == limit_docs
    
    #1 : Load Model
    #------ Load Model ------#
    model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_deepcut_test1_3.bin')
    model_loaded_attacut = models.doc2vec.Doc2Vec.load('model_attacut_test1_3.bin')
    
    #------ List: Words & Paragraph ID ------#
    dwords = list(model_loaded_deepcut.wv.vocab) #A #22187 vectors
    awords = list(model_loaded_attacut.wv.vocab) #A #22187 vectors
    ddocs = list(model_loaded_deepcut.docvecs.doctags.keys()) #B #14880 vectors
    adocs = list(model_loaded_attacut.docvecs.doctags.keys()) #B #14880 vectors
    
    
    #2 : Categories of News Articles & Paragraph vectors
    each_list_doc_id = vec_model.list_doc_id(list_genre,limit_docs,n)
    each_list_ddoc_vec = vec_model.list_ddoc_vec(list_genre,limit_docs,n)
    each_list_adoc_vec = vec_model.list_adoc_vec(list_genre,limit_docs,n) 
    print(each_list_doc_id)
    #print(each_list_ddoc_vec)
    #print(each_list_adoc_vec)
    
    
    #3 : Similar words
    each_list_dword_sim = vec_model.list_dword_sim(list_genre,limit_docs,n)
    each_list_dwords_sim_vec = vec_model.list_dwords_sim_vec(each_list_dword_sim)
    list_dwords_sim_id = each_list_dwords_sim_vec[0]
    list_dwords_sim_vec = each_list_dwords_sim_vec[1]
    
    each_list_aword_sim = vec_model.list_aword_sim(list_genre,limit_docs,n)
    each_list_awords_sim_vec = vec_model.list_awords_sim_vec(each_list_aword_sim)
    list_awords_sim_id = each_list_awords_sim_vec[0]
    list_awords_sim_vec = each_list_awords_sim_vec[1]
    
                
    #4 : All List [Paragraph ID]
    all_list_doc_id = vec_model.all_list_docs_id(each_list_doc_id)    
    all_list_ddocs_vec = vec_model.all_list_ddocs_vec(each_list_ddoc_vec)
    all_list_adocs_vec = vec_model.all_list_adocs_vec(each_list_adoc_vec)
   
    
    #5 : All List [Paragraph vectors + Similar Word vectors] 
    list_d_id = all_list_doc_id + list_dwords_sim_id
    list_a_id = all_list_doc_id + list_awords_sim_id
    list_dvec = all_list_ddocs_vec + list_dwords_sim_vec
    list_avec = all_list_adocs_vec + list_awords_sim_vec
    
    
    #6
    #-------------- PCA ------------------#
    #pca = PCA(n_components=2)
    #dresult_X_all = pca.fit_transform(list_dvec)
    #aresult_X_all = pca.fit_transform(list_avec)
    
    
    #-------------- TSNE -----------------#
    #tsne = TSNE(n_components=2)
    #tsne = TSNE(n_components=2, random_state=None, verbose=1, perplexity=40, n_iter=300)
    #tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=40)
    tsne = TSNE(n_components=3, perplexity=40, metric='euclidean', init='pca', verbose=0, random_state=0)
    dresult_X_all = tsne.fit_transform(list_dvec)
    aresult_X_all = tsne.fit_transform(list_avec)
    
    
    #------------- Setting plot -----------------#
    word_vec = "Word Vectors"
    docs_vec = "Paragraph Vectors"
    list_genre_plot = ["Economic", "Education", "Entertainment", "Foreign", "IT", "Sports"]
    list_wovec_plot = [list_genre_plot[0]+' '+word_vec,list_genre_plot[1]+' '+word_vec,list_genre_plot[2]+' '+word_vec,list_genre_plot[3]+' '+word_vec,list_genre_plot[4]+' '+word_vec,list_genre_plot[5]+' '+word_vec]
    list_dovec_plot = [list_genre_plot[0]+' '+docs_vec,list_genre_plot[1]+' '+docs_vec,list_genre_plot[2]+' '+docs_vec,list_genre_plot[3]+' '+docs_vec,list_genre_plot[4]+' '+word_vec,list_genre_plot[5]+' '+docs_vec]
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    plt.rcParams['font.size'] = 14
    #with plt.style.context('dark_background'):
    
    #********---------------------------------------------------------********#
    
    ################################################################################
    ############# 0.1 [Plot] Vector Space Model - DOC2VEC with Deepcut #############
    ################################################################################
    fig = plt.figure() #3D
    ax0 = Axes3D(fig)
    ax0.set_yticklabels([]) #Hide ticks
    ax0.set_xticklabels([]) #Hide ticks
    ax0.set_zticklabels([]) #Hide ticks
    #ax0.set_title('[3D] Vector Space Model - DOC2VEC with Deepcut (n_words = ' + str(n*count_gerne) + ' , limit_docs = ' + str(limit_docs*count_gerne) + ')')
    ax0.text2D(0.25, 0.95, '[3D] Vector Space Model - DOC2VEC with Deepcut (n_words = ' + str(n*count_gerne) + ' , limit_docs = ' + str(limit_docs*count_gerne) + ')', transform=ax0.transAxes)
    plot_list_dvec = vec_model.plot_list_dvec(all_list_doc_id,list_dwords_sim_id,dresult_X_all,p,limit_docs) 
    
    '''for angle in range(0,360,110):
        ax0.view_init(0, angle) #0'''
        
    legend1 = ax0.legend((plot_list_dvec[0],plot_list_dvec[1],plot_list_dvec[2],plot_list_dvec[3],plot_list_dvec[4],plot_list_dvec[5]), (list_genre_plot[0],list_genre_plot[1],list_genre_plot[2],list_genre_plot[3],list_genre_plot[4],list_genre_plot[5]), scatterpoints=1, labelspacing=1, title=docs_vec, loc='center left', bbox_to_anchor=(0.85,0.37))
    legend2 = ax0.legend((plot_list_dvec[6],plot_list_dvec[7],plot_list_dvec[8],plot_list_dvec[9],plot_list_dvec[10],plot_list_dvec[11]), (list_genre_plot[0],list_genre_plot[1],list_genre_plot[2],list_genre_plot[3],list_genre_plot[4],list_genre_plot[5]), scatterpoints=1, labelspacing=1, title=word_vec, loc='center left', bbox_to_anchor=(0.85,0.63))
    ax0.add_artist(legend1)
    ax0.add_artist(legend2)
    ax0.set_xlabel('x', fontsize=20)
    ax0.set_ylabel('y', fontsize=20)
    ax0.set_zlabel('z', fontsize=20)
    # plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 0.2 [Plot] Vector Space Model - DOC2VEC with Attacut #############
    ################################################################################
    fig = plt.figure() #3D
    ay0 = Axes3D(fig)
    ay0.set_yticklabels([]) #Hide ticks
    ay0.set_xticklabels([]) #Hide ticks
    ay0.set_zticklabels([]) #Hide ticks
    #ay0.set_title('[3D] Vector Space Model - DOC2VEC with Attacut (n_words = ' + str(n*count_gerne) + ' , limit_docs = ' + str(limit_docs*count_gerne) + ')')
    ay0.text2D(0.25, 0.95, '[3D] Vector Space Model - DOC2VEC with Attacut (n_words = ' + str(n*count_gerne) + ' , limit_docs = ' + str(limit_docs*count_gerne) + ')', transform=ay0.transAxes)
    plot_list_avec = vec_model.plot_list_avec(all_list_doc_id,list_awords_sim_id,aresult_X_all,p,limit_docs) 
    
    for angle in range(0,360,110):
        ay0.view_init(0, angle) #0
    
    legend1 = ay0.legend((plot_list_avec[0],plot_list_avec[1],plot_list_avec[2],plot_list_avec[3],plot_list_avec[4],plot_list_avec[5]), (list_genre_plot[0],list_genre_plot[1],list_genre_plot[2],list_genre_plot[3],list_genre_plot[4],list_genre_plot[5]), scatterpoints=1, labelspacing=1, title=docs_vec, loc='center left', bbox_to_anchor=(0.85,0.37))
    legend2 = ay0.legend((plot_list_avec[6],plot_list_avec[7],plot_list_avec[8],plot_list_avec[9],plot_list_avec[10],plot_list_avec[11]), (list_genre_plot[0],list_genre_plot[1],list_genre_plot[2],list_genre_plot[3],list_genre_plot[4],list_genre_plot[5]), scatterpoints=1, labelspacing=1, title=word_vec, loc='center left', bbox_to_anchor=(0.85,0.63))
    ay0.add_artist(legend1)
    ay0.add_artist(legend2)
    ay0.set_xlabel('x', fontsize=20)
    ay0.set_ylabel('y', fontsize=20)
    ay0.set_zlabel('z', fontsize=20)
    #plt.grid(True)
    plt.show()