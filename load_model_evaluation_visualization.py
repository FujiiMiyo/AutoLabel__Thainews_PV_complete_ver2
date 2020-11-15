# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:07:01 2020


"""

from gensim import models
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure
from bokeh.io import show, output_notebook

import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#http://csmoon-ml.com/index.php/2019/02/15/tutorial-doc2vec-and-t-sne/
#https://mlexplained.com/2018/09/14/paper-dissected-visualizing-data-using-t-sne-explained/
######https://leightley.com/visualizing-tweets-with-word2vec-and-t-sne-in-python/


if __name__ == '__main__':
    #model = models.doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)
    
    
    ###########################
    ####### Load model ########
    ###########################
    #model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_attacut_test_10') #test
    model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_deepcut_test1')
    model_loaded_attacut = models.doc2vec.Doc2Vec.load('model_attacut_test1')
    #print(model_loaded_deepcut)
    #print(model_loaded_attacut)
    
    #doc_tags = list(model_loaded_deepcut.docvecs.doctags.keys())
    #X = model_loaded_deepcut[doc_tags]
    #print(X)
    
    '''tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    X_tsne_df = pd.DataFrame(data=X_tsne, index=doc_tags, columns=['x', 'y'])
    #print(df)'''
    
    
    #----------------------------------------------------------------#
    
    
    ############ Deepcut ##############
    ################ test ################
    '''words = list(model_loaded_deepcut.wv.vocab)
    print(words)
    #print(model_loaded_deepcut['ดิจิทัล'])
    
    X = model_loaded_deepcut[model_loaded_deepcut.wv.vocab]
    #print(X)
    
    pca = PCA(n_components=2, random_state=0)
    result = pca.fit_transform(X)
    #print(result)
    #tsne = TSNE(n_components=2, random_state=0)
    #X_tsne = tsne.fit_transform(X)
    
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    fig, ax = plt.subplots()
    #plt.font_manager.fontManager.addfont('thsarabunnew-webfont.ttf')
    #plt.rc('font', family='TH Sarabun New', size=11)
    ax.plot(result[:, 0], result[:, 1], 'o')
    #ax.plot(X_tsne[:, 0], X_tsne[:, 1], 'o')
    ax.set_title('DOC2VEC - PCA')
    #ax.set_yticklabels([]) #Hide ticks
    #ax.set_xticklabels([]) #Hide ticks
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
    plt.show()'''
    ######################################################
    
    
    #------------------------------------------------------------------------------------#
    
    
    ######################################
    ################ Main ################
    ######################################
    dwords = list(model_loaded_deepcut.wv.vocab)
    awords = list(model_loaded_attacut.wv.vocab)
    ddocs = list(model_loaded_deepcut.docvecs.doctags.keys())   
    adocs = list(model_loaded_attacut.docvecs.doctags.keys())
    #print(dwords) #22187 vectors
    #print(ddocs) #14880 vectors
    #print(awords) #22187 vectors
    #print(adocs) #14880 vectors
    
    
    #--------- [Load] Target & Similar Word vectors ----------------#
    target_word = 'กระทรวงวัฒนธรรม'
    dtarget_word_vector = model_loaded_deepcut.wv[target_word]
    atarget_word_vector = model_loaded_attacut.wv[target_word]
    dsim_word = list(model_loaded_deepcut.wv.most_similar([target_word]))
    asim_word = list(model_loaded_attacut.wv.most_similar([target_word]))
    print(dsim_word) 
    print(asim_word) 
    
    #---# Add Similar Word vectors - Deepcut #---#
    dwords_arr = []
    for a, words in enumerate(dsim_word):
        most_similar_key, similarity = dsim_word[a]
        first_match = most_similar_key
        dwords_arr.append(first_match)
    dsim_word_vector = []
    for b, sim_dword in enumerate(dwords_arr):
        #print(sim_docword)
        sim_dword_vect = model_loaded_deepcut.wv[dwords_arr[b]]
        #print(sim_dword_vect)
        dsim_word_vector.append(sim_dword_vect)
    #print(dsim_word_vector)   
    #-----------------------------------------------------#
    
    #---# Add Similar Word vectors - Attacut #---#
    awords_arr = []
    for a, words in enumerate(asim_word):
        most_similar_key, similarity = asim_word[a]
        first_match = most_similar_key
        awords_arr.append(first_match)
    asim_word_vector = []
    for b, sim_aword in enumerate(awords_arr):
        #print(sim_aword)
        sim_aword_vect = model_loaded_attacut.wv[awords_arr[b]]
        #print(sim_aword_vect)
        asim_word_vector.append(sim_aword_vect)
    #print(asim_word_vector)   
    #-----------------------------------------------------#


    
    '''arr = np.empty((0,100), dtype='f')
    arr = np.append(arr,np.array([target_word_model]), axis=0)
    for b, wrd_score in enumerate(dsim_word):
        wrd_vector = model_loaded_deepcut[wrd_score[b]]
        dwords.append(wrd_score[b])
        arr = np.append(arr,np.array(wrd_vector), axis=0)
        print(arr)'''

    
    #------------  [Load] Vector Space Model - DOC2VEC --------------------# 
    dX_all = model_loaded_deepcut[model_loaded_deepcut.wv.vocab]
    dY_all = model_loaded_deepcut[model_loaded_deepcut.docvecs.doctags.keys()]
    aX_all = model_loaded_attacut[model_loaded_attacut.wv.vocab]
    aY_all = model_loaded_attacut[model_loaded_attacut.docvecs.doctags.keys()]    
    
    
    #------------- PCA -----------------#
    pca = PCA(n_components=2, random_state=0)
    dresult_X_all = pca.fit_transform(dX_all)
    dresult_Y_all = pca.fit_transform(dY_all)
    #dresult_X_tword = pca.fit_transform(dtarget_word_vector)
    dresult_X_simword = pca.fit_transform(dsim_word_vector)
    
    aresult_X_all = pca.fit_transform(aX_all)
    aresult_Y_all = pca.fit_transform(aY_all)   
    #aresult_X_tword = pca.fit_transform(atarget_word_vector)
    aresult_X_simword = pca.fit_transform(asim_word_vector)
    
    
    
    #-------------- TSEN -----------------#
    '''tsne = TSNE(n_components=2, random_state=0)
    dresult_X_all = tsne.fit_transform(dX_all)
    dresult_Y_all = tsne.fit_transform(dY_all)
    aresult_X_all = tsne.fit_transform(aX_all)
    aresult_Y_all = tsne.fit_transform(aY_all)'''
    
    
    #------------- Setting plot -----------------#
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    #plt.legend(('Paragraph vector (PV)', 'Word vector (WV)'),loc='lower right')
    

    
    ################################################################################
    ############# [Plot] Vector Space Model - DOC2VEC with Deepcut #################
    ################################################################################
    '''fig, ax = plt.subplots()
    ax.set_yticklabels([]) #Hide ticks
    ax.set_xticklabels([]) #Hide ticks
    ax.set_title('Vector Space Model - DOC2VEC with Deepcut')
    for i, word in enumerate(dwords):
        plt.annotate(word, xy=(dresult_X_all[i, 0], dresult_X_all[i, 1]))
        ax.plot(dresult_X_all[i, 0], dresult_X_all[i, 1], '^', color='gray', alpha=0.3)
        
    for j, doc in enumerate(ddocs):
        plt.annotate(doc, xy=(dresult_Y_all[j, 0], dresult_Y_all[j, 1]))
        ax.plot(dresult_Y_all[j, 0], dresult_Y_all[j, 1], 'o', color='hotpink', alpha=0.3)    
        
    ax.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ax.plot(dresult_Y_all[:, 0], dresult_Y_all[:, 1], 'o', color='hotpink', alpha=0.3)
    
    gray_patch = mpatches.Patch(color='gray', label='wv = Word vector')
    hotpink_patch = mpatches.Patch(color='hotpink', label='dv = Paragraph vector')
    ax.legend(handles=[gray_patch , hotpink_patch], loc='lower right')
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# [Plot] Vector Space Model - DOC2VEC with Attacut #################
    ################################################################################
    fig, ay = plt.subplots()
    ay.set_yticklabels([]) #Hide ticks
    ay.set_xticklabels([]) #Hide ticks
    ay.set_title('Vector Space Model - DOC2VEC with Attacut')
    for i, word in enumerate(awords):
        plt.annotate(word, xy=(aresult_X_all[i, 0], aresult_X_all[i, 1]))
        ay.plot(aresult_X_all[i, 0], aresult_X_all[i, 1], '^', color='gray', alpha=0.3)
    
    for j, doc in enumerate(adocs):
        plt.annotate(doc, xy=(aresult_Y_all[j, 0], aresult_Y_all[j, 1]))
        ay.plot(aresult_Y_all[j, 0], aresult_Y_all[j, 1], 'o', color='hotpink', alpha=0.3)  
    
    ay.plot(aresult_X_all[:, 0], aresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ay.plot(aresult_Y_all[:, 0], aresult_Y_all[:, 1], 'o', color='hotpink', alpha=0.3)
    
    gray_patch = mpatches.Patch(color='gray', label='wv = Word vector')
    hotpink_patch = mpatches.Patch(color='hotpink', label='dv = Paragraph vector')
    ay.legend(handles=[gray_patch , hotpink_patch], loc='lower right')
    plt.grid(True)
    plt.show()'''
    
    
    ###############################################################################################
    ############# [Plot] Result: Similar words - Target Word vector with Deepcut #################
    ###############################################################################################
    fig, ax1 = plt.subplots()
    ax1.set_yticklabels([]) #Hide ticks
    ax1.set_xticklabels([]) #Hide ticks
    #ax1.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ax1.set_title('Result: Similar words - Target Word vector with Deepcut')
    for i, word in enumerate(dwords_arr):
        ax1.plot(dresult_X_simword[i, 0], dresult_X_simword[i, 1], 'o', color='darkcyan')###
        plt.annotate(word, xy=(dresult_X_simword[i, 0], dresult_X_simword[i, 1]), horizontalalignment='right')
        #print(word)
        
    #ax1.plot(dresult_X_tword[0], dresult_X_tword[1], 'o', color='crimson')###
    #plt.annotate(word, xy=(dresult_X_tword[0], dresult_X_tword[1]))
    
    #gray_patch = mpatches.Patch(color='gray', label='Word vector')
    darkcyan_patch = mpatches.Patch(color='darkcyan', label='Similar Word vector')
    ax1.legend(handles=[darkcyan_patch],  loc='upper right')
    #ax1.legend(handles=[gray_patch , darkcyan_patch], loc='lower right')
    plt.grid(True)
    #plt.xlim(dresult_X_simword[:, 0].min()+0.00005, dresult_X_simword[:, 0].max()+0.00005)
    #plt.ylim(dresult_X_simword[:, 0].min()+0.00005, dresult_X_simword[:, 0].max()+0.00005)
    plt.show()
    
    
    ###############################################################################################
    ############# [Plot] Result: Similar words - Target Word vector with Attacut #################
    ###############################################################################################
    fig, ay1 = plt.subplots()
    ay1.set_yticklabels([]) #Hide ticks
    ay1.set_xticklabels([]) #Hide ticks
    #ay1.plot(aresult_X_all[:, 0], aresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ay1.set_title('Result: Similar words - Target Word vector with Attacut')
    for i, word in enumerate(awords_arr):
        ay1.plot(aresult_X_simword[i, 0], aresult_X_simword[i, 1], 'o', color='darkcyan')###
        plt.annotate(word, xy=(aresult_X_simword[i, 0], aresult_X_simword[i, 1]), horizontalalignment='right')
        #print(word)
        
    #ay1.plot(aresult_X_tword[0], aresult_X_tword[1], 'o', color='crimson')###
    #plt.annotate(word, xy=(aresult_X_tword[0], aresult_X_tword[1]))
    
    #gray_patch = mpatches.Patch(color='gray', label='Word vector')
    darkcyan_patch = mpatches.Patch(color='darkcyan', label='Similar Word vector')
    ay1.legend(handles=[darkcyan_patch], loc='upper right')
    #ay1.legend(handles=[gray_patch , gold_patch], loc='lower right')
    plt.grid(True)
    plt.show()
    
    
    
    
    
    ############# Vector Space Model - DOC2VEC with Attacut #################
    '''ax.set_title('Vector Space Model - DOC2VEC with Deepcut')
    for i, dword in enumerate(dwords):
        #plt.annotate(word, xy=(result_X[i, 0], result_X[i, 1]))
        ax.plot(dresult_X[i, 0], dresult_X[i, 1], '^', color='black', alpha=0.3)
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
    for j, doc in enumerate(docs):
        #plt.annotate(doc, xy=(result_Y[j, 0], result_Y[j, 1]))
        ax.plot(dresult_Y[j, 0], dresult_Y[j, 1], 'o', color='hotpink', alpha=0.3)    
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
    ax.plot(result_X[:, 0], result_X[:, 1], '^', color='black', alpha=0.3)
    ax.plot(result_Y[:, 0], result_Y[:, 1], 'o', color='hotpink', alpha=0.3)
    #ax.legend(('Word vector (WV)', 'Paragraph vector (PV)'),loc='lower right')
    black_patch = mpatches.Patch(color='black', label='wv = Word vector')
    hotpink_patch = mpatches.Patch(color='hotpink', label='dv = Paragraph vector')
    ax.legend(handles=[black_patch , hotpink_patch], loc='lower right')'''
    
    
    
    
    ################ 2D Cosine similarity - Feature ################
    '''ddoc0 = 'it_1241' 
    ddoc1 = 'it_314'
    dvec0 = model_loaded_deepcut.docvecs[ddoc0]
    dvec1 = model_loaded_deepcut.docvecs[ddoc1]
    dtasu1 = (dvec0+dvec1)
    #docs = list(model_loaded_deepcut.docvecs.most_similar([ddoc0]))
    docs1 = list(model_loaded_deepcut.docvecs.most_similar([ddoc0]))
    docs_target = ['it_1241']
    docs = docs1 + docs_target
    words_docs = list(model_loaded_deepcut.similar_by_vector(dtasu1, topn=10, restrict_vocab=None))
    print(words_docs)
    
    X = model_loaded_deepcut[model_loaded_deepcut.wv.vocab]
    Y = model_loaded_deepcut[model_loaded_deepcut.docvecs.doctags.keys()]
    #print(X)
    #print(Y)
    pca_X = PCA(n_components=2, random_state=0)
    result_X = pca_X.fit_transform(X)
    pca_Y = PCA(n_components=2, random_state=0)
    result_Y = pca_Y.fit_transform(Y)
    #print(result)
    #tsne = TSNE(n_components=2, random_state=0)
    #X_tsne = tsne.fit_transform(X)
    
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    fig, ax = plt.subplots()
    ax.set_title('DOC2VEC - PCA [Deepcut]')
    #ax.set_title('DOC2VEC - TSNE')
    ax.set_yticklabels([]) #Hide ticks
    ax.set_xticklabels([]) #Hide ticks
    for i, word in enumerate(words_docs):
        plt.annotate(word, xy=(result_X[i, 0], result_X[i, 1]))
        ax.plot(result_X[i, 0], result_X[i, 1], 'o')
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
        #ax.plot(X_tsne[i, 0], X_tsne[i, 1], 'o')
    for j, doc in enumerate(docs):
        plt.annotate(doc, xy=(result_Y[j, 0], result_Y[j, 1]))
        ax.plot(result_Y[j, 0], result_Y[j, 1], 'o')
    #ax.plot(result_X[200000000000000000000, 0], result_X[200000000000000000000, 1], 'o')
    #ax.plot(result_Y[200000000000000000000, 0], result_Y[200000000000000000000, 1], 'o')
    plt.show()'''
    
    ####################################
    
    ################ 2D Cosine similarity - Feature (test) ################
    '''ddoc0 = 'it_2413' #target
    ddoc1 = 'it_344'
    #ddoc2 = 'it_2196'
    dvec0 = model_loaded_deepcut.docvecs[ddoc0]
    dvec1 = model_loaded_deepcut.docvecs[ddoc1]
    #dvec2 = model_loaded_deepcut.docvecs[ddoc2]
    dtasu1 = (dvec0+dvec1)
    #docs = list(model_loaded_deepcut.docvecs.most_similar([ddoc0]))
    #docs_target = [ddoc0]
    #docsvec = ddocsim + docs_target
    
    ddocsim = list(model_loaded_deepcut.docvecs.most_similar([ddoc0]))
    docs = []
    for b, docsv in enumerate(ddocsim):
        most_similar_key, similarity = ddocsim[b]
        first_match = most_similar_key
        docs.append(first_match)
    docs.append(ddoc0)
    docs.append('Feature vector')
    print(docs)
    
    #words_docs = list(model_loaded_deepcut.similar_by_vector(dtasu1, topn=10, restrict_vocab=None))
    dsim = list(model_loaded_deepcut.similar_by_vector(dtasu1, topn=10, restrict_vocab=None))
    words_docs = []
    for a, words in enumerate(dsim):
        most_similar_key, similarity = dsim[a]
        first_match = most_similar_key
        words_docs.append(first_match)
    print(words_docs)
    
    
    X = model_loaded_deepcut[model_loaded_deepcut.wv.vocab]
    Y = model_loaded_deepcut[model_loaded_deepcut.docvecs.doctags.keys()]
    #print(X)
    #print(Y)
    pca_X = PCA(n_components=2, random_state=0)
    result_X = pca_X.fit_transform(X)
    pca_Y = PCA(n_components=2, random_state=0)
    result_Y = pca_Y.fit_transform(Y)
    #print(result)
    #tsne = TSNE(n_components=2, random_state=0)
    #X_tsne = tsne.fit_transform(X)
    
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    fig, ax = plt.subplots()
    ax.set_title('DOC2VEC - PCA [Deepcut]')
    #ax.set_title('DOC2VEC - TSNE')
    #ax.set_yticklabels([]) #Hide ticks
    #ax.set_xticklabels([]) #Hide ticks
    for i, word in enumerate(words_docs):
        plt.annotate(word, xy=(result_X[i, 0], result_X[i, 1]))
        ax.plot(result_X[i, 0], result_X[i, 1], 'o')
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
        #ax.plot(X_tsne[i, 0], X_tsne[i, 1], 'o')
    for j, doc in enumerate(docs):
        plt.annotate(doc, xy=(result_Y[j, 0], result_Y[j, 1]))
        ax.plot(result_Y[j, 0], result_Y[j, 1], 'o')
        plt.annotate(doc, xy=(result_Y[j, 0], result_Y[j, 1]))
        ax.plot(result_Y[j, 0], result_Y[j, 1], 'o')
    #ax.plot(result_X[900000000000000000000, 0], result_X[900000000000000000000, 1], 'o')
    #ax.plot(result_Y[900000000000000000000, 0], result_Y[900000000000000000000, 1], 'o')
    
    plt.show()'''
    
    ####################################
    
    
    
    ############ Attacut ##############
    ################ Vector Space Model ################
    '''words = list(model_loaded_attacut.wv.vocab)
    print(words)
    #print(model_loaded_attacut['ดิจิทัล'])
    
    X = model_loaded_attacut[model_loaded_attacut.wv.vocab]
    #print(X)
    
    pca = PCA(n_components=2, random_state=0)
    result = pca.fit_transform(X)
    #print(result)
    #tsne = TSNE(n_components=2, random_state=0)
    #X_tsne = tsne.fit_transform(X)
    
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    fig, ax = plt.subplots()
    #plt.font_manager.fontManager.addfont('thsarabunnew-webfont.ttf')
    #plt.rc('font', family='TH Sarabun New', size=11)
    ax.plot(result[:, 0], result[:, 1], 'o')
    #ax.plot(X_tsne[:, 0], X_tsne[:, 1], 'o')
    ax.set_title('DOC2VEC - PCA')
    ax.set_yticklabels([]) #Hide ticks
    ax.set_xticklabels([]) #Hide ticks
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
    plt.show()'''
    
    
    
    ################ Cosine similarity  Feature ################
    '''adoc0 = 'it_2349' 
    adoc1 = 'it_344'
    adoc2 = 'it_2196'
    avec0 = model_loaded_attacut.docvecs[adoc0]
    avec1 = model_loaded_attacut.docvecs[adoc1]
    avec2 = model_loaded_attacut.docvecs[adoc2]
    atasu1 = (avec0+avec1)
    #adocs = list(model_loaded_attacut.docvecs.most_similar([adoc0]))
    adocs1 = list(model_loaded_attacut.docvecs.most_similar([adoc0]))
    adocs_target = [adoc0]
    adocs = adocs1 + adocs_target
    awords_docs = list(model_loaded_attacut.similar_by_vector(atasu1, topn=10, restrict_vocab=None))
    print(awords_docs)
    
    aX = model_loaded_attacut[model_loaded_attacut.wv.vocab]
    aY = model_loaded_attacut[model_loaded_attacut.docvecs.doctags.keys()]
    #print(X)
    #print(Y)
    pca_aX = PCA(n_components=2, random_state=0)
    result_aX = pca_aX.fit_transform(aX)
    pca_aY = PCA(n_components=2, random_state=0)
    result_aY = pca_aY.fit_transform(aY)
    #print(result)
    #tsne = TSNE(n_components=2, random_state=0)
    #X_tsne = tsne.fit_transform(X)
    
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    fig, ay = plt.subplots()
    ay.set_title('DOC2VEC - PCA [Attacut]')
    #ax.set_title('DOC2VEC - TSNE')
    ay.set_yticklabels([]) #Hide ticks
    ay.set_xticklabels([]) #Hide ticks
    for i, word in enumerate(awords_docs):
        plt.annotate(word, xy=(result_aX[i, 0], result_aX[i, 1]))
        ay.plot(result_aX[i, 0], result_aX[i, 1], 'o')
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
        #ax.plot(X_tsne[i, 0], X_tsne[i, 1], 'o')
    for j, doc in enumerate(adocs):
        plt.annotate(doc, xy=(result_aY[j, 0], result_aY[j, 1]))
        ay.plot(result_aY[j, 0], result_aY[j, 1], 'o')
    ay.plot(result_aX[200000000000000000000, 0], result_aX[200000000000000000000, 1], 'o')
    ay.plot(result_aY[200000000000000000000, 0], result_aY[200000000000000000000, 1], 'o')
    plt.show()'''
    
    ####################################
    
    
    
    
    #############bokeh visaulization############### 
    '''output_notebook()
    #nrows = 12000
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    X_tsne_df = pd.DataFrame(data=X_tsne, index=doc_tags, columns=['x', 'y'])
    #print(df)
    #X_tsne_df["toxic_score"] = train["toxic_score"].values[:nrows]
    X_tsne_df.head()
    
    docs_top_tsne = tsne.fit_transform(X_tsne_df)
    p = figure(tools="pan,wheel_zoom,reset,save", toolbar_location="above", title="Doc2Vec t-SNE for first 10000 documents")

    colormap = np.array(["red"])

    source = ColumnDataSource(data=dict(x1=X_tsne_df["x"], x2=X_tsne_df["y"]))

    p.scatter(x="x1", y="x2", color='color', alpha=0.5, size=8, source=source)
   # hover = p.select(dict(type = HoverTool))
    #hover.tooltips = {"toxic_score":"@toxic_score"}

    show(p)'''
    
    
    '''labels = []
    tokens = []

    for word in model_loaded_deepcut.docvecs:
        tokens.append(model_loaded_deepcut.docvecs.doctags.keys())
        labels.append(word)
        
    print(list(labels))
    print(list(tokens))'''