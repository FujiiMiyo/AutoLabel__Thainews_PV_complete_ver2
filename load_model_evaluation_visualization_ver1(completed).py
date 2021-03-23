# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:01:52 2021


"""

from gensim import models ### use
from sklearn.manifold import TSNE ### use

import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure
from bokeh.io import show, output_notebook

import re ### use
import gensim
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt ### use
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold


#http://csmoon-ml.com/index.php/2019/02/15/tutorial-doc2vec-and-t-sne/
#https://mlexplained.com/2018/09/14/paper-dissected-visualizing-data-using-t-sne-explained/
######https://leightley.com/visualizing-tweets-with-word2vec-and-t-sne-in-python/


class visaulization_vector_space_model(object):        
    
    def pattern_words(self,n,m,target_word,target_genre,target_no_doc):
        dword_vect = model_loaded_deepcut.wv[target_word]
        aword_vect = model_loaded_attacut.wv[target_word]
        dsim_words_cos = model_loaded_deepcut.wv.similar_by_vector(dword_vect, topn=n, restrict_vocab=None)
        asim_words_cos = model_loaded_attacut.wv.similar_by_vector(aword_vect, topn=n, restrict_vocab=None)   
        
        target_doc = target_genre + '_' + target_no_doc
        ddoc_vect = model_loaded_deepcut.docvecs[target_doc]
        adoc_vect = model_loaded_attacut.docvecs[target_doc]
        dsim_words_cos2 = model_loaded_deepcut.wv.similar_by_vector(ddoc_vect, topn=n, restrict_vocab=None)
        asim_words_cos2 = model_loaded_attacut.wv.similar_by_vector(adoc_vect, topn=n, restrict_vocab=None)
        dsim_docs_cos = model_loaded_deepcut.docvecs.most_similar([target_doc], topn=m)
        asim_docs_cos = model_loaded_attacut.docvecs.most_similar([target_doc], topn=m)
        return dword_vect, aword_vect, dsim_words_cos, asim_words_cos,ddoc_vect,adoc_vect,dsim_words_cos2,asim_words_cos2,dsim_docs_cos,asim_docs_cos,target_doc
    
    
    def list_dword_sim(self,n,m,target_word,target_genre,target_no_doc):
        pattern_words = vec_model.pattern_words(n,m,target_word,target_genre,target_no_doc)
        list_dword_sim = pattern_words[2]
        dtarget_word_vec = pattern_words[0]
        list_dword_sim2 = pattern_words[6]
        dtarget_doc_vec = pattern_words[4]
        list_ddoc_sim = pattern_words[8]
        return list_dword_sim,dtarget_word_vec,list_dword_sim2,dtarget_doc_vec,list_ddoc_sim
    
    def list_aword_sim(self,n,m,target_word,target_genre,target_no_doc):
        pattern_words = vec_model.pattern_words(n,m,target_word,target_genre,target_no_doc)
        list_aword_sim = pattern_words[3]
        atarget_word_vec = pattern_words[1]
        list_aword_sim2 = pattern_words[7]
        atarget_doc_vec = pattern_words[5]
        list_adoc_sim = pattern_words[9]
        return list_aword_sim,atarget_word_vec,list_aword_sim2,atarget_doc_vec,list_adoc_sim
    
    
    def list_dwords_sim_vec(self,dsim_words):
        dwords_sim = []
        for a, words in enumerate(dsim_words):
            most_similar_key, similarity = dsim_words[a]
            first_match = most_similar_key
            dwords_sim.append(first_match)
        dwords_sim_vec = []
        for b, sim_dword in enumerate(dwords_sim):
            sim_dword_vect = model_loaded_deepcut.wv[dwords_sim[b]]
            dwords_sim_vec.append(sim_dword_vect)
        return dwords_sim,dwords_sim_vec
    
    def list_awords_sim_vec(self,asim_words):
        awords_sim = []
        for a, words in enumerate(asim_words):
            most_similar_key, similarity = asim_words[a]
            first_match = most_similar_key
            awords_sim.append(first_match)
        awords_sim_vec = []
        for b, sim_aword in enumerate(awords_sim):
            sim_aword_vect = model_loaded_attacut.wv[awords_sim[b]]
            awords_sim_vec.append(sim_aword_vect)
        return awords_sim,awords_sim_vec
    
    
    def list_ddocs_sim_vec(self,dsim_docs,target_doc,dtarget_doc_vec,patterns):
        ddocs_sim = []
        for c, docs in enumerate(dsim_docs):
            most_similar_key, similarity = dsim_docs[c]
            first_match = most_similar_key
            ddocs_sim.append(first_match)
        ddocs_sim_vec = []
        for d, sim_ddocs in enumerate(ddocs_sim):
            sim_ddocs_vec = model_loaded_deepcut.docvecs[ddocs_sim[d]]
            ddocs_sim_vec.append(sim_ddocs_vec)
        #ex-l = dtarget_doc_vec+ddocs_sim_vec[0]+ddocs_sim_vec[1]-ddocs_sim_vec[2]-ddocs_sim_vec[3]+ddocs_sim_vec[4]+ddocs_sim_vec[5]+ddocs_sim_vec[6]-ddocs_sim_vec[7]+ddocs_sim_vec[8]-ddocs_sim_vec[9]+ddocs_sim_vec[10]-ddocs_sim_vec[11]+ddocs_sim_vec[12]+ddocs_sim_vec[13]-ddocs_sim_vec[14]-ddocs_sim_vec[15]-ddocs_sim_vec[16]-ddocs_sim_vec[17]-ddocs_sim_vec[18]-ddocs_sim_vec[19]-ddocs_sim_vec[20]-ddocs_sim_vec[21]+ddocs_sim_vec[22]+ddocs_sim_vec[23]+ddocs_sim_vec[24]+ddocs_sim_vec[25]+ddocs_sim_vec[26]+ddocs_sim_vec[27]-ddocs_sim_vec[28]+ddocs_sim_vec[29]
            
        dsimdocs_id = []
        dfeat_vec = [dtarget_doc_vec]
        for s, simdoc_id in enumerate(ddocs_sim):
            z = re.match(patterns, simdoc_id)
            if s < n:
                if z:
                    #print("addsim_ddocs = ", z.groups())
                    add_sim_id = ' + ' + simdoc_id
                    add_sim_vec = ddocs_sim_vec[s]
                    dsimdocs_id.append(add_sim_id)
                    dfeat_vec.append(add_sim_vec)
                else:
                    #print("subsim_ddocs = ", z.groups())
                    sub_sim_id = ' - ' + simdoc_id
                    sub_sim_vec = -ddocs_sim_vec[s]
                    dsimdocs_id.append(sub_sim_id)
                    dfeat_vec.append(sub_sim_vec)
        dfeat_vec_sum = sum(dfeat_vec)
        dfeat_id_sum = [target_doc] + dsimdocs_id
        dsim_words_cos3 = model_loaded_deepcut.wv.similar_by_vector(dfeat_vec_sum, topn=n, restrict_vocab=None)
        return ddocs_sim,ddocs_sim_vec,dfeat_id_sum,dfeat_vec_sum,dsim_words_cos3
    
    def list_adocs_sim_vec(self,asim_docs,target_doc,atarget_doc_vec,patterns):
        adocs_sim = []
        for c, docs in enumerate(asim_docs):
            most_similar_key, similarity = asim_docs[c]
            first_match = most_similar_key
            adocs_sim.append(first_match)
        adocs_sim_vec = []
        for d, sim_adocs in enumerate(adocs_sim):
            sim_adocs_vec = model_loaded_attacut.docvecs[adocs_sim[d]]
            adocs_sim_vec.append(sim_adocs_vec)
            
        asimdocs_id = []
        afeat_vec = [atarget_doc_vec]
        for s, simdoc_id in enumerate(adocs_sim):
            z = re.match(patterns, simdoc_id)
            if s < n:
                if z:
                    #print("addsim_adocs = ", z.groups())
                    add_sim_id = ' + ' + simdoc_id
                    add_sim_vec = adocs_sim_vec[s]
                    asimdocs_id.append(add_sim_id)
                    afeat_vec.append(add_sim_vec)
                else:
                    #print("subsim_ddocs = ", z.groups())
                    sub_sim_id = ' - ' + simdoc_id
                    sub_sim_vec = -adocs_sim_vec[s]
                    asimdocs_id.append(sub_sim_id)
                    afeat_vec.append(sub_sim_vec)
        afeat_vec_sum = sum(afeat_vec)
        afeat_id_sum = [target_doc] + asimdocs_id
        asim_words_cos3 = model_loaded_attacut.wv.similar_by_vector(afeat_vec_sum, topn=n, restrict_vocab=None)
        return adocs_sim,adocs_sim_vec,afeat_id_sum,afeat_vec_sum,asim_words_cos3
    
    
    def plot_list_dsimwords_tword(self,list_dword_id,dresult_X,o):
        for i, word in enumerate(list_dword_id):
            if i != 0:
                if i < o:
                    scatter0 = ax1.scatter(dresult_X[i, 0], dresult_X[i, 1], c='#007d92', alpha=0.8, edgecolors='none', marker='>', s=80)###
                    ax1.text(dresult_X[i, 0], dresult_X[i, 1], word)
                    #print("Similar words = ",word)
                else:
                    scatter1 = ax1.scatter(dresult_X[i, 0], dresult_X[i, 1], c='#fdd03b', alpha=0.5, edgecolors='none', marker='>', s=80)###
                    #ax1.text(dresult_X[i, 0], dresult_X[i, 1], word)
                    #print("Similar words = ",word)        
            else:
                scatter2 = ax1.scatter(dresult_X[0, 0], dresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ax1.text(dresult_X[0, 0], dresult_X[0, 1], word)
                #print("Target word = ", word)
        return scatter0,scatter1,scatter2
    
    
    def plot_list_asimwords_tword(self,list_aword_id,aresult_X,o):
        for i, word in enumerate(list_aword_id):
            if i != 0:
                if i < o:
                    scatter0 = ay1.scatter(aresult_X[i, 0], aresult_X[i, 1], c='#007d92', alpha=0.8, edgecolors='none', marker='>', s=80)###
                    ay1.text(aresult_X[i, 0], aresult_X[i, 1], word)
                    #print("Similar words = ",word)
                else:
                    scatter1 = ay1.scatter(aresult_X[i, 0], aresult_X[i, 1], c='#fdd03b', alpha=0.5, edgecolors='none', marker='>', s=80)###
                    #ay1.text(aresult_X[i, 0], aresult_X[i, 1], word)
                    #print("Similar words = ",word)        
            else:
                scatter2 = ay1.scatter(aresult_X[0, 0], aresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ay1.text(aresult_X[0, 0], aresult_X[0, 1], word)
                #print("Target word = ", word)
        return scatter0,scatter1,scatter2
    
    def plot_list_dsimwords_tdoc(self,list_dword_id,dresult_X,o):
        for i, word in enumerate(list_dword_id):
            if i != 0:
                if i < o:
                    scatter0 = ax2.scatter(dresult_X[i, 0], dresult_X[i, 1], c='#007d92', alpha=0.8, edgecolors='none', marker='>', s=80)###
                    ax2.text(dresult_X[i, 0], dresult_X[i, 1], word)
                    #print("Similar words = ",word)
                else:
                    scatter1 = ax2.scatter(dresult_X[i, 0], dresult_X[i, 1], c='#fdd03b', alpha=0.5, edgecolors='none', marker='>', s=80)###
                    #ax2.text(dresult_X[i, 0], dresult_X[i, 1], word)
                    #print("Similar words = ",word)        
            else:
                scatter2 = ax2.scatter(dresult_X[0, 0], dresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ax2.text(dresult_X[0, 0], dresult_X[0, 1], word)
                #print("Target word = ", word)
        return scatter0,scatter1,scatter2
    
    
    def plot_list_asimwords_tdoc(self,list_aword_id,aresult_X,o):
        for i, word in enumerate(list_aword_id):
            if i != 0:
                if i < o:
                    scatter0 = ay2.scatter(aresult_X[i, 0], aresult_X[i, 1], c='#4a9396', alpha=0.8, edgecolors='none', marker='>', s=80)###
                    ay2.text(aresult_X[i, 0], aresult_X[i, 1], word)
                    #print("Similar words = ",word)
                else:
                    scatter1 = ay2.scatter(aresult_X[i, 0], aresult_X[i, 1], c='#fdd03b', alpha=0.5, edgecolors='none', marker='>', s=80)###
                    #ay2.text(aresult_X[i, 0], aresult_X[i, 1], word)
                    #print("Similar words = ",word)        
            else:
                scatter2 = ay2.scatter(aresult_X[0, 0], aresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ay2.text(aresult_X[0, 0], aresult_X[0, 1], word)
                #print("Target word = ", word)
        return scatter0,scatter1,scatter2
    
    def plot_list_dsimwords_feat(self,list_dword_id,dresult_X,o):
        for i, word in enumerate(list_dword_id):
            if i != 0:
                if i < o:
                    scatter0 = ax3.scatter(dresult_X[i, 0], dresult_X[i, 1], c='#4a9396', alpha=0.8, edgecolors='none', marker='>', s=80)###
                    ax3.text(dresult_X[i, 0], dresult_X[i, 1], word)
                    #print("Similar words = ",word)
                else:
                    scatter1 = ax3.scatter(dresult_X[i, 0], dresult_X[i, 1], c='#fdd03b', alpha=0.5, edgecolors='none', marker='>', s=80)###
                    #ax3.text(dresult_X[i, 0], dresult_X[i, 1], word)
                    #print("Similar words = ",word)        
            else:
                scatter2 = ax3.scatter(dresult_X[0, 0], dresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ax3.text(dresult_X[0, 0], dresult_X[0, 1], word)
                #print("Target word = ", word)
        return scatter0,scatter1,scatter2
    
    def plot_list_asimwords_feat(self,list_aword_id,aresult_X,o):
        for i, word in enumerate(list_aword_id):
            if i != 0:
                if i < o:
                    scatter0 = ay3.scatter(aresult_X[i, 0], aresult_X[i, 1], c='#4a9396', alpha=0.8, edgecolors='none', marker='>', s=80)###
                    ay3.text(aresult_X[i, 0], aresult_X[i, 1], word)
                    #print("Similar words = ",word)
                else:
                    scatter1 = ay3.scatter(aresult_X[i, 0], aresult_X[i, 1], c='#fdd03b', alpha=0.5, edgecolors='none', marker='>', s=80)###
                    #ax3.text(aresult_X[i, 0], aresult_X[i, 1], word)
                    #print("Similar words = ",word)        
            else:
                scatter2 = ay3.scatter(aresult_X[0, 0], aresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ay3.text(aresult_X[0, 0], aresult_X[0, 1], word)
                #print("Target word = ", word)
        return scatter0,scatter1,scatter2
    
    def plot_list_dsimdocs_feat(self,list_ddoc_id,dresult_X,p,pattern):
        for j, doc in enumerate(list_ddoc_id):
            if j != 0 and j != 1:
                dz = re.search(pattern, doc)
                #print(j)
                #print(doc)
                if j < p:
                    if dz:
                        #print("addsim_adocs = ", z.groups())
                        scatter0 = ax4.scatter(dresult_X[j, 0], dresult_X[j, 1], c='#4a9396', alpha=0.8, edgecolors='none', marker='o', s=80)###
                        ax4.text(dresult_X[j, 0], dresult_X[j, 1], doc)
                        print("Top 10 Similar articles = ", doc)
                    else:
                        #print("subsim_adocs = ", z.groups())
                        scatter1 = ax4.scatter(dresult_X[j, 0], dresult_X[j, 1], c='#fdd03b', alpha=0.8, edgecolors='none', marker='o', s=80)###
                        ax4.text(dresult_X[j, 0], dresult_X[j, 1], doc)
                        print("Similar articles = ", doc)
            elif j == 1:
                scatter2 = ax4.scatter(dresult_X[1, 0], dresult_X[1, 1], c='#f3166b', alpha=0.8, edgecolors='none', marker=(5,0), s=110)###
                ax4.text(dresult_X[1, 0], dresult_X[1, 1], doc)
                #print("Target word = ", doc)
            elif j == 0:
                scatter3 = ax4.scatter(dresult_X[0, 0], dresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ax4.text(dresult_X[0, 0], dresult_X[0, 1], doc)
                #print("Feature vector = ", doc)
        return scatter0,scatter1,scatter2,scatter3
    
    def plot_list_asimdocs_feat(self,list_adoc_id,aresult_X,p,pattern):
        for j, doc in enumerate(list_adoc_id):
            if j != 0 and j != 1:
                az = re.search(pattern, doc)
                #print(j)
                #print(doc)
                if j < p:
                    if az:
                        #print("addsim_adocs = ", z.groups())
                        scatter0 = ay4.scatter(aresult_X[j, 0], aresult_X[j, 1], c='#4a9396', alpha=0.8, edgecolors='none', marker='o', s=80)###
                        ay4.text(aresult_X[j, 0], aresult_X[j, 1], doc)
                        print("Top 10 Similar articles = ", doc)
                    else:
                        #print("subsim_adocs = ", z.groups())
                        scatter1 = ay4.scatter(aresult_X[j, 0], aresult_X[j, 1], c='#fdd03b', alpha=0.8, edgecolors='none', marker='o', s=80)###
                        ay4.text(aresult_X[j, 0], aresult_X[j, 1], doc)
                        print("Similar articles = ", doc)
            elif j == 1:
                scatter2 = ay4.scatter(aresult_X[1, 0], aresult_X[1, 1], c='#f3166b', alpha=0.8, edgecolors='none', marker=(5,0), s=110)###
                ay4.text(aresult_X[1, 0], aresult_X[1, 1], doc)
                #print("Target word = ", doc)
            elif j == 0:
                scatter3 = ay4.scatter(aresult_X[0, 0], aresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ay4.text(aresult_X[0, 0], aresult_X[0, 1], doc)
                #print("Feature vector = ", doc)
        return scatter0,scatter1,scatter2,scatter3
    
    def plot_list_dsimdocs_feat_all(self,list_ddoc_id,dresult_X,o,p,pattern):
        for j, doc in enumerate(list_ddoc_id):
            if j != 0 and j != 1:
                if j >= 2 and j < (2+p):
                    dz = re.search(pattern, doc)
                    #print(j)
                    #print(doc)
                    if j < p:
                        if dz:
                            #print("addsim_adocs = ", z.groups())
                            scatter0 = ax4.scatter(dresult_X[j, 0], dresult_X[j, 1], c='#4a9396', alpha=0.8, edgecolors='none', marker='o', s=80)###
                            ax4.text(dresult_X[j, 0], dresult_X[j, 1], doc)
                            print("Top 10 Similar articles = ", doc)
                        else:
                            #print("subsim_adocs = ", z.groups())
                            scatter1 = ax4.scatter(dresult_X[j, 0], dresult_X[j, 1], c='#fdd03b', alpha=0.8, edgecolors='none', marker='o', s=80)###
                            ax4.text(dresult_X[j, 0], dresult_X[j, 1], doc)
                            print("Similar articles = ", doc)
            elif j == 1:
                scatter2 = ax4.scatter(dresult_X[1, 0], dresult_X[1, 1], c='#f3166b', alpha=0.8, edgecolors='none', marker=(5,0), s=110)###
                ax4.text(dresult_X[1, 0], dresult_X[1, 1], doc)
                #print("Target word = ", doc)
            elif j == 0:
                scatter3 = ax4.scatter(dresult_X[0, 0], dresult_X[0, 1], c='#981826', alpha=0.8, edgecolors='none', marker=(5,1), s=110)###
                ax4.text(dresult_X[0, 0], dresult_X[0, 1], doc)
                #print("Feature vector = ", doc)
        return scatter0,scatter1,scatter2,scatter3
    
    
if __name__ == '__main__':
    
    #0 : Setting
    vec_model = visaulization_vector_space_model()
    count_gerne = 6
    n = 30 ##Note: Count Similar words
    m = 30 ##Note: Count Similar paragraph(s)
    o = 10 ##Note: Count Similar words -> plot (o<n)
    p = 20 ##Note: Count Similar paragraph(s) -> plot (o<n)
    target_word = 'มัลแวร์' #1-target word
    target_genre = 'it'
    target_no_doc = '1982'
    target_doc = target_genre + '_' + target_no_doc #2-target paragraph
    feat_vec = 'Feature vector' #Fixed
    patterns = '(' + target_genre + '_\w+)'
    patterns2 = '(' + target_genre + '_\w+)+'
    
    #1 : Load Model
    #------ Load Model ------#
    model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_deepcut_test1_3.bin')
    model_loaded_attacut = models.doc2vec.Doc2Vec.load('model_attacut_test1_3.bin')
    
    
    #3 : Similar words
    
    ''' #Deepcut '''
    each_list_dword_sim = vec_model.list_dword_sim(n,m,target_word,target_genre,target_no_doc)
    dsim_words = each_list_dword_sim[0] #target word
    dtarget_word_vec = each_list_dword_sim[1]
    dsim_words2 = each_list_dword_sim[2] #target paragraph
    dtarget_doc_vec = each_list_dword_sim[3]
    dsim_docs = each_list_dword_sim[4] 
    ''' Target word vector '''
    each_list_dwords_sim_vec = vec_model.list_dwords_sim_vec(dsim_words) ###->
    list_dwords_sim_id = each_list_dwords_sim_vec[0]
    list_dwords_sim_vec = each_list_dwords_sim_vec[1]
    ''' Target paragraph vector '''
    each_list_dwords_sim_vec2 = vec_model.list_dwords_sim_vec(dsim_words2) ###->
    list_dwords_sim_id2_tdoc = [target_doc] + each_list_dwords_sim_vec2[0] 
    list_dwords_sim_vec2_tdoc = [dtarget_doc_vec] + each_list_dwords_sim_vec2[1]
    ''' Feature vector '''
    each_list_ddocs_sim_vec = vec_model.list_ddocs_sim_vec(dsim_docs,target_doc,dtarget_doc_vec,patterns)
    list_ddocs_sim_id = each_list_ddocs_sim_vec[0]
    list_ddocs_sim_vec = each_list_ddocs_sim_vec[1]
    list_dfeat_id = each_list_ddocs_sim_vec[2] ###-> Similar paragraph (Feature) + target paragraph
    list_dfeat_vec = each_list_ddocs_sim_vec[3]
    list_dfeat_sim_vec = each_list_ddocs_sim_vec[4]  ###-> Similar words (Feature)
    each_list_dwords_sim_vec3 = vec_model.list_dwords_sim_vec(list_dfeat_sim_vec) ###->
    list_dwords_sim_id3_feat = [feat_vec] + each_list_dwords_sim_vec3[0]
    list_dwords_sim_vec3_feat = [list_dfeat_vec] + each_list_dwords_sim_vec3[1]
    list_ddocs_sim_id4_feat = [feat_vec] + list_dfeat_id
    list_ddocs_sim_vec4_feat = [list_dfeat_vec] + list_ddocs_sim_vec
    list_ddocs_sim_id5_feat_all = [feat_vec] + list_dfeat_id + each_list_dwords_sim_vec3[0]
    list_ddocs_sim_vec5_feat_all = [list_dfeat_vec] + list_ddocs_sim_vec + each_list_dwords_sim_vec3[1]
    
    
    ''' #Attacut '''
    each_list_aword_sim = vec_model.list_aword_sim(n,m,target_word,target_genre,target_no_doc)
    asim_words = each_list_aword_sim[0] #target word
    atarget_word_vec = each_list_aword_sim[1]
    asim_words2 = each_list_aword_sim[2] #target paragraph
    atarget_doc_vec = each_list_aword_sim[3]
    asim_docs = each_list_aword_sim[4]
    ''' Target word vector '''
    each_list_awords_sim_vec = vec_model.list_awords_sim_vec(asim_words) ###->
    list_awords_sim_id = each_list_awords_sim_vec[0]
    list_awords_sim_vec = each_list_awords_sim_vec[1]
    ''' Target paragraph vector '''
    each_list_awords_sim_vec2 = vec_model.list_awords_sim_vec(asim_words2) ###->
    list_awords_sim_id2_tdoc = [target_doc] +  each_list_awords_sim_vec2[0]
    list_awords_sim_vec2_tdoc = [atarget_doc_vec] + each_list_awords_sim_vec2[1]
    ''' Feature vector '''
    each_list_adocs_sim_vec = vec_model.list_adocs_sim_vec(asim_docs,target_doc,atarget_doc_vec,patterns)
    list_adocs_sim_id = each_list_adocs_sim_vec[0]
    list_adocs_sim_vec = each_list_adocs_sim_vec[1]
    list_afeat_id = each_list_adocs_sim_vec[2] ###-> Similar paragraph (Feature) + target paragraph
    list_afeat_vec = each_list_adocs_sim_vec[3]
    list_afeat_sim_vec = each_list_adocs_sim_vec[4]  ###-> Similar words (Feature)
    each_list_awords_sim_vec3 = vec_model.list_awords_sim_vec(list_afeat_sim_vec) ###->
    list_awords_sim_id3_feat = [feat_vec] + each_list_awords_sim_vec3[0]
    list_awords_sim_vec3_feat = [list_afeat_vec] + each_list_awords_sim_vec3[1]
    list_adocs_sim_id4_feat = [feat_vec] + list_afeat_id
    list_adocs_sim_vec4_feat = [list_afeat_vec] + list_adocs_sim_vec
    list_adocs_sim_id5_feat_all = [feat_vec] + list_afeat_id + each_list_awords_sim_vec3[0]
    list_adocs_sim_vec5_feat_all = [list_afeat_vec] + list_adocs_sim_vec + each_list_awords_sim_vec3[1]
    
    #6
    #-------------- PCA ------------------#
    #pca = PCA(n_components=2)
    #dresult_X_all = pca.fit_transform(list_dvec)
    #aresult_X_all = pca.fit_transform(list_avec)
    
    #-------------- TSNE -----------------#
    #tsne = TSNE(n_components=2)
    #tsne = TSNE(n_components=2, random_state=None, verbose=1, perplexity=40, n_iter=300)
    #tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=40)
    tsne = TSNE(n_components=2, perplexity=40, metric='euclidean', init='pca', verbose=0, random_state=0)
    dresult_X_tword = tsne.fit_transform(list_dwords_sim_vec)
    aresult_X_tword = tsne.fit_transform(list_awords_sim_vec)
    dresult_X_tdoc = tsne.fit_transform(list_dwords_sim_vec2_tdoc)
    aresult_X_tdoc = tsne.fit_transform(list_awords_sim_vec2_tdoc)
    dresult_X_feat = tsne.fit_transform(list_dwords_sim_vec3_feat)
    aresult_X_feat = tsne.fit_transform(list_awords_sim_vec3_feat)
    dresult_X_feat_pv = tsne.fit_transform(list_ddocs_sim_vec4_feat)
    aresult_X_feat_pv = tsne.fit_transform(list_adocs_sim_vec4_feat)
    dresult_X_feat_all = tsne.fit_transform(list_ddocs_sim_vec5_feat_all)
    aresult_X_feat_all = tsne.fit_transform(list_adocs_sim_vec5_feat_all)
    
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
    ############# 1.1 [Plot] Similar words - Target Word vector with Deepcut #############
    ################################################################################
    fig, ax1 = plt.subplots()
    ax1.set_yticklabels([]) #Hide ticks
    ax1.set_xticklabels([]) #Hide ticks
    ax1.set_title('[2D] Similar words - Target Word Vector with Deepcut (n_similar words = ' + str(o) + ' , target word = ' + target_word + ')')
    plot_list_dvec = vec_model.plot_list_dsimwords_tword(list_dwords_sim_id,dresult_X_tword,o) 
    
    legend1 = ax1.legend((plot_list_dvec[0],plot_list_dvec[1],plot_list_dvec[2]), ('Top 10 Similar Words','Similar Words','Target Word'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3)
    ax1.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 1.2 [Plot] Similar words - Target Word vector with Attacut #############
    ################################################################################
    fig, ay1 = plt.subplots()
    ay1.set_yticklabels([]) #Hide ticks
    ay1.set_xticklabels([]) #Hide ticks
    ay1.set_title('[2D] Similar words - Target Word Vector with Attacut (n_similar words = ' + str(o) + ' , target word = ' + target_word + ')')
    plot_list_avec = vec_model.plot_list_asimwords_tword(list_awords_sim_id,aresult_X_tword,o) 
    
    legend1 = ay1.legend((plot_list_avec[0],plot_list_avec[1],plot_list_avec[2]), ('Top 10 Similar Words','Similar Words','Target Word'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3)
    ay1.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 2.1 [Plot] Similar words - Target paragraph vector with Deepcut #############
    ################################################################################
    fig, ax2 = plt.subplots()
    ax2.set_yticklabels([]) #Hide ticks
    ax2.set_xticklabels([]) #Hide ticks
    ax2.set_title('[2D] Similar words - Target Paragraph Vector with Deepcut (n_similar words = ' + str(o) + ' , target article = ' + target_doc + ')')
    plot_list_dvec2 = vec_model.plot_list_dsimwords_tdoc(list_dwords_sim_id2_tdoc,dresult_X_tdoc,o) 
    
    legend1 = ax2.legend((plot_list_dvec2[0],plot_list_dvec2[1],plot_list_dvec2[2]), ('Top 10 Similar Words','Similar Words','Target Article'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3)
    ax2.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 2.2 [Plot] Similar words - Target paragraph vector with Attacut #############
    ################################################################################
    fig, ay2 = plt.subplots()
    ay2.set_yticklabels([]) #Hide ticks
    ay2.set_xticklabels([]) #Hide ticks
    ay2.set_title('[2D] Similar words - Target Paragraph Vector with Attacut (n_similar words = ' + str(o) + ' , target article = ' + target_doc + ')')
    plot_list_avec2 = vec_model.plot_list_asimwords_tdoc(list_awords_sim_id2_tdoc,aresult_X_tdoc,o) 
    
    legend1 = ay2.legend((plot_list_avec2[0],plot_list_avec2[1],plot_list_avec2[2]), ('Top 10 Similar Words','Similar Words','Target Article'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3)
    ay2.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 3.1 [Plot] Similar words - Feature vector with Deepcut #############
    ################################################################################
    fig, ax3 = plt.subplots()
    ax3.set_yticklabels([]) #Hide ticks
    ax3.set_xticklabels([]) #Hide ticks
    ax3.set_title('[2D] Similar words - Feature Vector with Deepcut (n_similar words = ' + str(o) + ' , target article = ' + target_doc + ')')
    plot_list_dvec3 = vec_model.plot_list_dsimwords_feat(list_dwords_sim_id3_feat,dresult_X_feat,o) 
    
    legend1 = ax3.legend((plot_list_dvec3[0],plot_list_dvec3[1],plot_list_dvec3[2]), ('Top 10 Similar Words','Similar Words','Target Article'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3)
    ax3.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 3.2 [Plot] Similar words - Feature vector with Attacut #############
    ################################################################################
    fig, ay3 = plt.subplots()
    ay3.set_yticklabels([]) #Hide ticks
    ay3.set_xticklabels([]) #Hide ticks
    ay3.set_title('[2D] Similar words - Feature Vector with Attacut (n_similar words = ' + str(o) + ' , target article = ' + target_doc + ')')
    plot_list_avec3 = vec_model.plot_list_asimwords_feat(list_awords_sim_id3_feat,aresult_X_feat,o) 
    
    legend1 = ay3.legend((plot_list_avec3[0],plot_list_avec3[1],plot_list_avec3[2]), ('Top 10 Similar Words','Similar Words','Target Article'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3)
    ay3.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 4.1 [Plot] Similar articles - Feature vector with Deepcut #############
    ################################################################################
    fig, ax4 = plt.subplots()
    ax4.set_yticklabels([]) #Hide ticks
    ax4.set_xticklabels([]) #Hide ticks
    ax4.set_title('[2D] Similar words - Feature Vector with Deepcut (n_similar articles = ' + str(p) + ' , target article = ' + target_doc + ')')
    plot_list_dvec4 = vec_model.plot_list_dsimdocs_feat(list_ddocs_sim_id4_feat,dresult_X_feat_pv,p,patterns2) 
    
    legend1 = ax4.legend((plot_list_dvec4[0],plot_list_dvec4[1],plot_list_dvec4[2],plot_list_dvec4[3]), ('Similar Articles (same genre of news)','Similar Articles (other genre of news)','Target Article','Feature vector'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4)
    ax4.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 4.2 [Plot] Similar articles - Feature vector with Attacut #############
    ################################################################################
    fig, ay4 = plt.subplots()
    ay4.set_yticklabels([]) #Hide ticks
    ay4.set_xticklabels([]) #Hide ticks
    ay4.set_title('[2D] Similar words - Feature Vector with Attacut (n_similar articles = ' + str(p) + ' , target article = ' + target_doc + ')')
    plot_list_avec4 = vec_model.plot_list_asimdocs_feat(list_adocs_sim_id4_feat,aresult_X_feat_pv,p,patterns2) 
    
    legend1 = ay4.legend((plot_list_avec4[0],plot_list_avec4[1],plot_list_avec4[2],plot_list_avec4[3]), ('Similar Articles (same genre of news)','Similar Articles (other genre of news)','Target Article','Feature vector'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4)
    ay4.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    '''################################################################################
    ############# 5.1 [Plot] Similar articles - Feature vector with Deepcut #############
    ################################################################################
    fig, ax5 = plt.subplots()
    ax5.set_yticklabels([]) #Hide ticks
    ax5.set_xticklabels([]) #Hide ticks
    ax5.set_title('[2D] Similar words - Feature Vector with Deepcut (n_similar words = ' + str(o) + ' ,n_similar articles = ' + str(p) + ' , target article = ' + target_doc + ')')
    plot_list_dvec5 = vec_model.plot_list_dsimdocs_feat_all(list_ddocs_sim_id5_feat_all,dresult_X_feat_all,o,p,patterns2) 
    
    legend1 = ax5.legend((plot_list_dvec5[0],plot_list_dvec5[1],plot_list_dvec5[2],plot_list_dvec5[3]), ('Similar Articles (same genre of news)','Similar Articles (other genre of news)','Target Article','Feature vector'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4)
    ax5.add_artist(legend1)
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 5.2 [Plot] Similar articles - Feature vector with Attacut #############
    ################################################################################
    fig, ay5 = plt.subplots()
    ay5.set_yticklabels([]) #Hide ticks
    ay5.set_xticklabels([]) #Hide ticks
    ay5.set_title('[2D] Similar words - Feature Vector with Attacut (n_similar words = ' + str(o) + ' ,n_similar articles = ' + str(p) + ' , target article = ' + target_doc + ')')
    plot_list_avec5 = vec_model.plot_list_asimdocs_feat_all(list_adocs_sim_id5_feat_all,aresult_X_feat_all,o,p,patterns2) 
    
    legend1 = ay5.legend((plot_list_avec5[0],plot_list_avec5[1],plot_list_avec5[2],plot_list_avec5[3]), ('Similar Articles (same genre of news)','Similar Articles (other genre of news)','Target Article','Feature vector'), scatterpoints=1, labelspacing=1, title=docs_vec, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4)
    ay5.add_artist(legend1)
    plt.grid(True)
    plt.show()'''