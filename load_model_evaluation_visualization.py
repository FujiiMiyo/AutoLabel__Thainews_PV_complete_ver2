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
    
    
    
    #********--------- 0. [Load] Vector Space Model - DOC2VEC --------********# 
    dX_all = model_loaded_deepcut[model_loaded_deepcut.wv.vocab]
    dY_all = model_loaded_deepcut[model_loaded_deepcut.docvecs.doctags.keys()]
    aX_all = model_loaded_attacut[model_loaded_attacut.wv.vocab]
    aY_all = model_loaded_attacut[model_loaded_attacut.docvecs.doctags.keys()] 
    #********---------------------------------------------------------********#
    
    
    
    #********--------- 1. [Load] Target & Similar Word vectors-----------********#
    target_word = 'มัลแวร์' #<================================================================== * Target word * #มัลแวร์
    dtarget_word_vector = model_loaded_deepcut.wv[target_word]
    atarget_word_vector = model_loaded_attacut.wv[target_word]
    dsim_word = list(model_loaded_deepcut.wv.most_similar([target_word]))
    asim_word = list(model_loaded_attacut.wv.most_similar([target_word]))
    print("Similar words from Target Word vector [Deepcut] = ", dsim_word) 
    print("Similar words from Target Word vector [Attacut] = ", asim_word) 
    
    
    #---# 1.1 Add Similar Word vectors - Deepcut #---#
    dwords_arr = [target_word]
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
    
    
    #---# 1.2 Add Similar Word vectors - Attacut #---#
    awords_arr = [target_word]
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
    
    #********---------------------------------------------------------********#   
        
    
    
    #********--------- 2. [Load] Target Paragraph vectors / Feature vector-----------********#
    
    ######## 2.1 Target Paragraph vectors ######
    #---# 2.1.1 Find Similar Paragraph vectors #---#
    target_doc = 'it_1500' #<=================================================================== * Target paragraph * #it_1500
    dtarget_doc_vector = model_loaded_deepcut.docvecs[target_doc]
    atarget_doc_vector = model_loaded_attacut.docvecs[target_doc]
    ddoc_sim_word = list(model_loaded_deepcut.wv.similar_by_vector(dtarget_doc_vector, topn=10, restrict_vocab=None))
    adoc_sim_word = list(model_loaded_attacut.wv.similar_by_vector(atarget_doc_vector, topn=10, restrict_vocab=None))
    print("Similar words from Target Paragraph vector [Deepcut] = ", ddoc_sim_word)
    print("Similar words from Target Paragraph vector [Attacut] = ", adoc_sim_word)
    
    
    #---# 2.1.2 Add Similar Word vectors - Deepcut #---#
    ddoc_words_arr = [target_doc] #**
    for a, words in enumerate(ddoc_sim_word):
        most_similar_key, similarity = ddoc_sim_word[a]
        first_match = most_similar_key
        ddoc_words_arr.append(first_match)
    ddoc_sim_word_vector = []
    for b, sim_dword in enumerate(ddoc_words_arr):
        #print(sim_docword)
        if sim_dword != target_doc:
            sim_dword_vect = model_loaded_deepcut.wv[ddoc_words_arr[b]]
            #print(sim_dword_vect)
            ddoc_sim_word_vector.append(sim_dword_vect)
        else:
            ddoc_sim_word_vector.append(dtarget_doc_vector)
    #print(ddoc_sim_word_vector)   
    #-----------------------------------------------------#
    
    
    #---# 2.1.3 Add Similar Word vectors - Attacut #---#
    adoc_words_arr = [target_doc] #**
    for a, words in enumerate(adoc_sim_word):
        most_similar_key, similarity = adoc_sim_word[a]
        first_match = most_similar_key
        adoc_words_arr.append(first_match)
    adoc_sim_word_vector = []
    for b, sim_aword in enumerate(adoc_words_arr):
        #print(sim_docword)
        if sim_aword != target_doc:
            sim_aword_vect = model_loaded_attacut.wv[adoc_words_arr[b]]
            #print(sim_aword_vect)
            adoc_sim_word_vector.append(sim_aword_vect)
        else:
            adoc_sim_word_vector.append(atarget_doc_vector)
    #print(adoc_sim_word_vector)   
    #-----------------------------------------------------#
        
    
    ######## 2.2 Feature vector ######
    #---# 2.2.1 Find Similar Paragraph vectors #---#
    feat_vec = 'Feature vector'
    dsim_docs = list(model_loaded_deepcut.docvecs.most_similar([target_doc]))
    asim_docs = list(model_loaded_attacut.docvecs.most_similar([target_doc]))
    print("Similar paragraphs from Feature vector [Deepcut] = ", dsim_docs)
    print("Similar paragraphs from Feature vector [Attacut] = ", asim_docs)
    
    #---# 2.2.2-1 Similar Paragraph - Deepcut #---#
    dsim_docs_id = [] #**
    for g, ddoc_sim in enumerate(dsim_docs):
        most_similar_key, similarity = dsim_docs[g]
        first_match = most_similar_key
        dsim_docs_id.append(first_match)
    #print("Similar paragraph = ", dsim_docs_id)
    ddoc_sim_doc_vector = []
    for h, ddoc_sim_vec in enumerate(dsim_docs_id):
        sim_ddoc_vect = model_loaded_deepcut.docvecs[dsim_docs_id[h]]
        #print(sim_ddoc_vect)
        ddoc_sim_doc_vector.append(sim_ddoc_vect)
    #print("Similar Paragraph vectors = ", ddoc_sim_doc_vector)
    
    
    #---# 2.2.2-2 Similar Paragraph - Attacut #---#
    asim_docs_id = [] #**
    for g, adoc_sim in enumerate(asim_docs):
        most_similar_key, similarity = asim_docs[g]
        first_match = most_similar_key
        asim_docs_id.append(first_match)
    #print("Similar paragraph = ", asim_docs_id)
    adoc_sim_doc_vector = []
    for h, adoc_sim_vec in enumerate(asim_docs_id):
        sim_adoc_vect = model_loaded_attacut.docvecs[asim_docs_id[h]]
        #print(sim_adoc_vect)
        adoc_sim_doc_vector.append(sim_adoc_vect)
    #print("Similar Paragraph vectors = ", adoc_sim_doc_vector)
    
    
    #---# 2.2.3-1 Add Similar Word vectors - Deepcut #---#
    #dfeat_vec = dtarget_doc_vector + ddoc_sim_doc_vector[0] #<========================== * Feature vector *
    dfeat_vec = dtarget_doc_vector + ddoc_sim_doc_vector[0] + ddoc_sim_doc_vector[1] + ddoc_sim_doc_vector[2] + ddoc_sim_doc_vector[3] + ddoc_sim_doc_vector[4] + ddoc_sim_doc_vector[5] + ddoc_sim_doc_vector[6] + ddoc_sim_doc_vector[7] + ddoc_sim_doc_vector[8] + ddoc_sim_doc_vector[9]
    dfeat_vec_list = [dfeat_vec, dtarget_doc_vector] + ddoc_sim_doc_vector
    dfeat_docs_list = [feat_vec, target_doc] + dsim_docs_id
    #print("[Deepcut] Feature vector = ", dfeat_vec)
    #print(ddoc_sim_doc_vector[0])
    
    dfeat_sim_word = list(model_loaded_deepcut.wv.similar_by_vector(dfeat_vec, topn=10, restrict_vocab=None))
    #print("Similar words from Feature vector [Deepcut] = ", dfeat_sim_word)
    
    dfeat_words_arr = [feat_vec] #**
    for a, words in enumerate(dfeat_sim_word):
        most_similar_key, similarity = dfeat_sim_word[a]
        first_match = most_similar_key
        dfeat_words_arr.append(first_match)
    dfeat_sim_word_vector = []
    for b, sim_dword in enumerate(dfeat_words_arr):
        #print(sim_docword)
        if sim_dword != 'Feature vector':
            sim_dword_vect = model_loaded_deepcut.wv[dfeat_words_arr[b]]
            #print(sim_dword_vect)
            dfeat_sim_word_vector.append(sim_dword_vect)
        else:
            dfeat_sim_word_vector.append(dfeat_vec)
    #print(dfeat_sim_word_vector)
    
    
    #---# 2.2.3-2 Add Similar Word vectors - Attacut #---#
    #afeat_vec = atarget_doc_vector + adoc_sim_doc_vector[0]
    afeat_vec = atarget_doc_vector + adoc_sim_doc_vector[0] + adoc_sim_doc_vector[1] + adoc_sim_doc_vector[2] + adoc_sim_doc_vector[3] + adoc_sim_doc_vector[4] + adoc_sim_doc_vector[5] + adoc_sim_doc_vector[6] + adoc_sim_doc_vector[7] + adoc_sim_doc_vector[8] - adoc_sim_doc_vector[9]
    #afeat_vec = atarget_doc_vector + adoc_sim_doc_vector[0] + adoc_sim_doc_vector[1]
    afeat_vec_list = [afeat_vec, atarget_doc_vector] + adoc_sim_doc_vector
    afeat_docs_list = [feat_vec, target_doc] + asim_docs_id
    #print("[Attacut] Feature vector = ", afeat_vec)
    
    afeat_sim_word = list(model_loaded_attacut.wv.similar_by_vector(afeat_vec, topn=10, restrict_vocab=None))
    print("Similar words from Feature vector [Attacut] = ", afeat_sim_word)
    
    afeat_words_arr = [feat_vec] #**
    for a, words in enumerate(afeat_sim_word):
        most_similar_key, similarity = afeat_sim_word[a]
        first_match = most_similar_key
        afeat_words_arr.append(first_match)
    afeat_sim_word_vector = []
    for b, sim_aword in enumerate(afeat_words_arr):
        #print(sim_aocword)
        if sim_aword != feat_vec:
            sim_aword_vect = model_loaded_attacut.wv[afeat_words_arr[b]]
            #print(sim_aword_vect)
            afeat_sim_word_vector.append(sim_aword_vect)
        else:
            afeat_sim_word_vector.append(afeat_vec)
    #print(afeat_sim_word_vector)
     
    #********---------------------------------------------------------********#
    
    
    
    
    
    #########################################
    #******** Setting Visualization ********#
    #########################################
    
    #------------- PCA -----------------#
    pca = PCA(n_components=2, random_state=0)
    
    dresult_X_all = pca.fit_transform(dX_all)
    dresult_Y_all = pca.fit_transform(dY_all)
    dresult_X_simword = pca.fit_transform(dsim_word_vector)
    dresult_X_doc_simword = pca.fit_transform(ddoc_sim_word_vector)
    dresult_X_feat_simword = pca.fit_transform(dfeat_sim_word_vector)
    dresult_X_feat_simdocs = pca.fit_transform(dfeat_vec_list)
    
    aresult_X_all = pca.fit_transform(aX_all)
    aresult_Y_all = pca.fit_transform(aY_all)   
    aresult_X_simword = pca.fit_transform(asim_word_vector)
    aresult_X_doc_simword = pca.fit_transform(adoc_sim_word_vector)
    aresult_X_feat_simword = pca.fit_transform(afeat_sim_word_vector)
    aresult_X_feat_simdocs = pca.fit_transform(afeat_vec_list)
    
    
    #-------------- TSEN -----------------#
    '''tsne = TSNE(n_components=2, random_state=0)
    dresult_X_all = tsne.fit_transform(dX_all)
    dresult_Y_all = tsne.fit_transform(dY_all)
    aresult_X_all = tsne.fit_transform(aX_all)
    aresult_Y_all = tsne.fit_transform(aY_all)'''
    
    
    #------------- Setting plot -----------------#
    plt.rcParams['font.family'] = 'TH SarabunPSK'
    #plt.legend(('Paragraph vector (PV)', 'Word vector (WV)'),loc='lower right')
    '''d_x = np.random.randn(1000)
    d_y = np.random.randn(1000)'''
    wv_patch = mpatches.Patch(color='gray', label='wv = Word vector')
    dv_patch = mpatches.Patch(color='hotpink', label='dv = Paragraph vector')
    tfeat_patch = mpatches.Patch(color='salmon', label='Feature vector')
    
    tword_patch = mpatches.Patch(color='crimson', label='Target word vector')
    tdocs_patch = mpatches.Patch(color='crimson', label='Target paragraph vector')
    
    sword_patch = mpatches.Patch(color='goldenrod', label='Similar word vector')
    sdoc_patch = mpatches.Patch(color='hotpink', label='Similar paragraph vector')

    
    #********---------------------------------------------------------********#
    
    
    
    ################################################################################
    ############# 0.1 [Plot] Vector Space Model - DOC2VEC with Deepcut #############
    ################################################################################
    fig, ax = plt.subplots()
    #ax.set_yticklabels([]) #Hide ticks
    #ax.set_xticklabels([]) #Hide ticks
    ax.set_title('Vector Space Model - DOC2VEC with Deepcut')
    for i, word in enumerate(dwords):
        #plt.annotate(word, xy=(dresult_X_all[i, 0], dresult_X_all[i, 1]))
        ax.plot(dresult_X_all[i, 0], dresult_X_all[i, 1], '^', color='gray', alpha=0.3)
        
    for j, doc in enumerate(ddocs):
        #plt.annotate(doc, xy=(dresult_Y_all[j, 0], dresult_Y_all[j, 1]))
        ax.plot(dresult_Y_all[j, 0], dresult_Y_all[j, 1], 'o', color='hotpink', alpha=0.3)    
        
    ax.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ax.plot(dresult_Y_all[:, 0], dresult_Y_all[:, 1], 'o', color='hotpink', alpha=0.3)
    
    ax.legend(handles=[wv_patch , dv_patch], loc='lower right')
    plt.grid(True)
    plt.show()
    
    
    ################################################################################
    ############# 0.2 [Plot] Vector Space Model - DOC2VEC with Attacut #############
    ################################################################################
    fig, ay = plt.subplots()
    #ay.set_yticklabels([]) #Hide ticks
    #ay.set_xticklabels([]) #Hide ticks
    ay.set_title('Vector Space Model - DOC2VEC with Attacut')
    for i, word in enumerate(awords):
        #plt.annotate(word, xy=(aresult_X_all[i, 0], aresult_X_all[i, 1]))
        ay.plot(aresult_X_all[i, 0], aresult_X_all[i, 1], '^', color='gray', alpha=0.3)
    
    for j, doc in enumerate(adocs):
        #plt.annotate(doc, xy=(aresult_Y_all[j, 0], aresult_Y_all[j, 1]))
        ay.plot(aresult_Y_all[j, 0], aresult_Y_all[j, 1], 'o', color='hotpink', alpha=0.3)  
    
    ay.plot(aresult_X_all[:, 0], aresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ay.plot(aresult_Y_all[:, 0], aresult_Y_all[:, 1], 'o', color='hotpink', alpha=0.3)
    
    ay.legend(handles=[wv_patch , dv_patch], loc='lower right')
    plt.grid(True)
    plt.show()
    

    ###############################################################################################
    ############# 1.1 [Plot] Result: Similar words - Target Word vector with Deepcut ##############
    ###############################################################################################
    fig, ax1 = plt.subplots()
    #ax1.set_yticklabels([]) #Hide ticks
    #ax1.set_xticklabels([]) #Hide ticks
    #ax1.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ax1.set_title('Result: Similar words - Target Word vector with Deepcut')
    for i, word in enumerate(dwords_arr):
        if i != 0:
            ax1.plot(dresult_X_simword[i, 0], dresult_X_simword[i, 1], 'o', color='goldenrod')###
            plt.annotate(word, xy=(dresult_X_simword[i, 0], dresult_X_simword[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        else:
            ax1.plot(dresult_X_simword[0, 0], dresult_X_simword[0, 1], '^', color='crimson')###
            plt.annotate(word, xy=(dresult_X_simword[0, 0], dresult_X_simword[0, 1]), horizontalalignment='right')
            #print("Target word = ", word)
    
    ax1.legend(handles=[tword_patch , sdoc_patch], loc='upper right')
    plt.grid(True)
    plt.show()
    
    
    ###############################################################################################
    ############# 1.2 [Plot] Result: Similar words - Target Word vector with Attacut ##############
    ###############################################################################################
    fig, ay1 = plt.subplots()
    #ay1.set_yticklabels([]) #Hide ticks
    #ay1.set_xticklabels([]) #Hide ticks
    #ay1.plot(aresult_X_all[:, 0], aresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ay1.set_title('Result: Similar words - Target Word vector with Attacut')
    for i, word in enumerate(awords_arr):
        if i != 0:
            ay1.plot(aresult_X_simword[i, 0], aresult_X_simword[i, 1], 'o', color='goldenrod')###
            plt.annotate(word, xy=(aresult_X_simword[i, 0], aresult_X_simword[i, 1]), horizontalalignment='right')
            print("Similar words = ",word)
        else:
            ay1.plot(aresult_X_simword[0, 0], aresult_X_simword[0, 1], '^', color='crimson')###
            plt.annotate(word, xy=(aresult_X_simword[0, 0], aresult_X_simword[0, 1]), horizontalalignment='right')
            print("Target word = ",word)
    
    ay1.legend(handles=[tword_patch , sdoc_patch], loc='lower left')
    plt.grid(True)
    plt.show()
    
    
    ###############################################################################################
    ######### 2.1 [Plot] Result: Similar words - Target Paragraph vector with Deepcut #############
    ###############################################################################################
    fig, ax2 = plt.subplots()
    #ax2.set_yticklabels([]) #Hide ticks
    #ax2.set_xticklabels([]) #Hide ticks
    #ax2.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ax2.set_title('Result: Similar words - Target Paragraph vector with Deepcut')
    for i, word in enumerate(ddoc_words_arr):
        if i != 0:
            ax2.plot(dresult_X_doc_simword[i, 0], dresult_X_doc_simword[i, 1], 'o', color='goldenrod')###
            plt.annotate(word, xy=(dresult_X_doc_simword[i, 0], dresult_X_doc_simword[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        else:
            ax2.plot(dresult_X_doc_simword[0, 0], dresult_X_doc_simword[0, 1], '^', color='crimson')###
            plt.annotate(word, xy=(dresult_X_doc_simword[0, 0], dresult_X_doc_simword[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    
    ax2.legend(handles=[tdocs_patch , sword_patch], loc='upper right')
    plt.grid(True)
    plt.show()
    
    
    ###############################################################################################
    ######### 2.2 [Plot] Result: Similar words - Target Paragraph vector with Attapcut ############
    ###############################################################################################
    fig, ay2 = plt.subplots()
    #ay2.set_yticklabels([]) #Hide ticks
    #ay2.set_xticklabels([]) #Hide ticks
    #ay2.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ay2.set_title('Result: Similar words - Target Paragraph vector with Attacut')
    for i, word in enumerate(adoc_words_arr):
        if i != 0:
            ay2.plot(aresult_X_doc_simword[i, 0], aresult_X_doc_simword[i, 1], 'o', color='goldenrod')###
            plt.annotate(word, xy=(aresult_X_doc_simword[i, 0], aresult_X_doc_simword[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        else:
            ay2.plot(aresult_X_doc_simword[0, 0], aresult_X_doc_simword[0, 1], '^', color='crimson')###
            plt.annotate(word, xy=(aresult_X_doc_simword[0, 0], aresult_X_doc_simword[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    
    ay2.legend(handles=[tdocs_patch , sword_patch], loc='upper right')
    plt.grid(True)
    plt.show()
    

    ###############################################################################################
    ######### 3.1 [Plot] Result: Similar words - Feature vector with Deepcut #############
    ###############################################################################################
    '''fig, ax2 = plt.subplots()
    ax2.set_yticklabels([]) #Hide ticks
    ax2.set_xticklabels([]) #Hide ticks
    #ax2.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ax2.set_title('Result: Similar words - Feature vector with Deepcut')
    for i, word in enumerate(dfeat_words_arr):
        if i != 0:
            ax2.plot(dresult_X_feat_simword[i, 0], dresult_X_feat_simword[i, 1], 'o', color='gray')###
            plt.annotate(word, xy=(dresult_X_feat_simword[i, 0], dresult_X_feat_simword[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        else:
            ax2.plot(dresult_X_feat_simword[0, 0], dresult_X_feat_simword[0, 1], '*', color='crimson')###
            plt.annotate(word, xy=(dresult_X_feat_simword[0, 0], dresult_X_feat_simword[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    
    hotpink_patch = mpatches.Patch(color='hotpink', label='Target Paragraph vector')
    gray_patch = mpatches.Patch(color='gray', label='Similar Word vector')
    ax2.legend(handles=[gray_patch , hotpink_patch], loc='upper right') 
    plt.grid(True)
    plt.show()
    
    
    
    ###############################################################################################
    ######### 3.2 [Plot] Result: Similar words - Feature vector with Attacut ######################
    ###############################################################################################
    fig, ay2 = plt.subplots()
    #ay2.set_yticklabels([]) #Hide ticks
    #ay2.set_xticklabels([]) #Hide ticks
    #ay2.plot(aresult_X_all[:, 0], aresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ay2.set_title('Result: Similar words - Feature vector with Attacut')
    for i, word in enumerate(afeat_words_arr):
        if i != 0:
            ay2.plot(aresult_X_feat_simword[i, 0], aresult_X_feat_simword[i, 1], 'o', color='gray')###
            plt.annotate(word, xy=(aresult_X_feat_simword[i, 0], aresult_X_feat_simword[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        else:
            ay2.plot(aresult_X_feat_simword[0, 0], aresult_X_feat_simword[0, 1], '^', color='hotpink')###
            plt.annotate(word, xy=(aresult_X_feat_simword[0, 0], aresult_X_feat_simword[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    
    hotpink_patch = mpatches.Patch(color='hotpink', label='Target Paragraph vector')
    gray_patch = mpatches.Patch(color='gray', label='Similar Word vector')
    ay2.legend(handles=[hotpink_patch , gray_patch], loc='upper right')
    ay2.set_xticks([-10,-4,0,4,10])
    plt.grid(True)
    plt.show()'''
    
    
    ###############################################################################################
    ######### 3.3-1 [Plot] Result: Similar paragraphs - Feature vector with Deepcut #############
    ###############################################################################################
    '''fig, ax3 = plt.subplots()
    ax3.set_yticklabels([]) #Hide ticks
    ax3.set_xticklabels([]) #Hide ticks
    #ax3.plot(dresult_X_all[:, 0], dresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ax3.set_title('Result: Similar paragraphs - Feature vector with Deepcut')
    for i, docs in enumerate(dfeat_docs_list):
        if i != 0 and i !=1:
            ax3.plot(dresult_X_feat_simdocs[i, 0], dresult_X_feat_simdocs[i, 1], 'o', color='gold')###
            plt.annotate(docs, xy=(dresult_X_feat_simdocs[i, 0], dresult_X_feat_simdocs[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        elif(i == 0):
            ax3.plot(dresult_X_feat_simdocs[0, 0], dresult_X_feat_simdocs[0, 1], '^', color='hotpink')###
            plt.annotate(docs, xy=(dresult_X_feat_simdocs[0, 0], dresult_X_feat_simdocs[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
        elif(i == 1):
            ax3.plot(dresult_X_feat_simdocs[1, 0], dresult_X_feat_simdocs[1, 1], '^', color='crimson')###
            plt.annotate(docs, xy=(dresult_X_feat_simdocs[1, 0], dresult_X_feat_simdocs[1, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    
    crimson_patch = mpatches.Patch(color='crimson', label='Target Paragraph vector')
    gold_patch = mpatches.Patch(color='gold', label='Similar Word vector')
    ax3.legend(handles=[crimson_patch , gold_patch], loc='upper right')
    plt.grid(True)
    plt.show()
    
    
    ###############################################################################################
    ######### 3.3-2 [Plot] Result: Similar paragraphs - Feature vector with Attacut #############
    ###############################################################################################
    fig, ay3 = plt.subplots()
    ay3.set_yticklabels([]) #Hide ticks
    ay3.set_xticklabels([]) #Hide ticks
    ay3.plot(aresult_X_all[:, 0], aresult_X_all[:, 1], '^', color='gray', alpha=0.3)
    ay3.set_title('Result: Similar paragraphs - Feature vector with Attacut')
    for i, docs in enumerate(afeat_docs_list):
        if i != 0:
            ay3.plot(aresult_X_feat_simdocs[i, 0], aresult_X_feat_simdocs[i, 1], 'o', color='gold')###
            plt.annotate(docs, xy=(aresult_X_feat_simdocs[i, 0], aresult_X_feat_simdocs[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        elif(i == 0):
            ay3.plot(aresult_X_feat_simdocs[0, 0], aresult_X_feat_simdocs[0, 1], '^', color='crimson')###
            plt.annotate(docs, xy=(aresult_X_feat_simdocs[0, 0], aresult_X_feat_simdocs[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
        elif(i == 1):
            ay3.plot(aresult_X_feat_simdocs[1, 0], aresult_X_feat_simdocs[1, 1], '^', color='crimson')###
            plt.annotate(docs, xy=(aresult_X_feat_simdocs[1, 0], aresult_X_feat_simdocs[1, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    
    crimson_patch = mpatches.Patch(color='crimson', label='Target Paragraph vector')
    gold_patch = mpatches.Patch(color='gold', label='Similar Word vector')
    ay3.legend(handles=[crimson_patch , gold_patch], loc='upper right')
    plt.grid(True)
    plt.show()'''
    
    
    ###############################################################################################
    ######### 3.2 [Plot] Result: Similar words + Similar paragraph - Feature vector with Attacut ######################
    ###############################################################################################
    fig, ax4 = plt.subplots()
    #ay4.set_yticklabels([]) #Hide ticks
    #ay4.set_xticklabels([]) #Hide ticks
    #ay4.plot(dresult_X_all[22008, 0], dresult_X_all[22008, 1], '^', color='gray', alpha=0.3)
    ax4.set_title('Result: Similar words - Feature vector with Deepcut')
    for i, word in enumerate(dfeat_words_arr):
        if i != 0:
            ax4.plot(dresult_X_feat_simword[i, 0], dresult_X_feat_simword[i, 1], '^', color='goldenrod')###
            plt.annotate(word, xy=(dresult_X_feat_simword[i, 0], dresult_X_feat_simword[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        else:
            ax4.plot(dresult_X_feat_simword[0, 0], dresult_X_feat_simword[0, 1], '^', color='salmon')###
            plt.annotate(word, xy=(dresult_X_feat_simword[0, 0], dresult_X_feat_simword[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    for j, docs in enumerate(dfeat_docs_list):
        if j != 0 and j != 1:
            ax4.plot(dresult_X_feat_simdocs[j, 0], dresult_X_feat_simdocs[j, 1], 'o', color='hotpink')###
            plt.annotate(docs, xy=(dresult_X_feat_simdocs[j, 0], dresult_X_feat_simdocs[j, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        elif(j == 1):
            ax4.plot(dresult_X_feat_simdocs[1, 0], dresult_X_feat_simdocs[1, 1], '*', color='crimson')###
            plt.annotate(docs, xy=(dresult_X_feat_simdocs[1, 0], dresult_X_feat_simdocs[1, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
        '''elif(j == 0):
            ax4.plot(dresult_X_feat_simdocs[0, 0], dresult_X_feat_simdocs[0, 1], 'o', color='goldenrod')###
            plt.annotate(docs, xy=(dresult_X_feat_simdocs[0, 0], dresult_X_feat_simdocs[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)'''
            
    
    ax4.legend(handles=[tdocs_patch, tfeat_patch, sword_patch, sdoc_patch], loc='upper right')
    #ay4.set_xticks([-5,0,5])
    #ay4.set_yticks([-5,0,5])
    plt.grid(True)
    plt.show()
    
    
    ###############################################################################################
    ######### 3.4-2 [Plot] Result: Similar words + Similar paragraph - Feature vector with Attacut ######################
    ###############################################################################################
    fig, ay4 = plt.subplots()
    #ay4.set_yticklabels([]) #Hide ticks
    #ay4.set_xticklabels([]) #Hide ticks
    #ay4.plot(aresult_X_all[22008, 0], aresult_X_all[22008, 1], '^', color='gray', alpha=0.3)
    ay4.set_title('Result: Similar words - Feature vector with Attacut')
    for i, word in enumerate(afeat_words_arr):
        if i != 0:
            ay4.plot(aresult_X_feat_simword[i, 0], aresult_X_feat_simword[i, 1], '^', color='goldenrod')###
            plt.annotate(word, xy=(aresult_X_feat_simword[i, 0], aresult_X_feat_simword[i, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        else:
            ay4.plot(aresult_X_feat_simword[0, 0], aresult_X_feat_simword[0, 1], '^', color='salmon')###
            plt.annotate(word, xy=(aresult_X_feat_simword[0, 0], aresult_X_feat_simword[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
    for j, docs in enumerate(afeat_docs_list):
        if j != 0 and j != 1:
            ay4.plot(aresult_X_feat_simdocs[j, 0], aresult_X_feat_simdocs[j, 1], 'o', color='hotpink')###
            plt.annotate(docs, xy=(aresult_X_feat_simdocs[j, 0], aresult_X_feat_simdocs[j, 1]), horizontalalignment='right')
            #print("Similar words = ",word)
        elif(j == 1):
            ay4.plot(aresult_X_feat_simdocs[1, 0], aresult_X_feat_simdocs[1, 1], '*', color='crimson')###
            plt.annotate(docs, xy=(aresult_X_feat_simdocs[1, 0], aresult_X_feat_simdocs[1, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)
            
        '''elif(j == 0):
            ay4.plot(aresult_X_feat_simdocs[0, 0], aresult_X_feat_simdocs[0, 1], 'o', color='gold')###
            plt.annotate(docs, xy=(aresult_X_feat_simdocs[0, 0], aresult_X_feat_simdocs[0, 1]), horizontalalignment='right')
            #print("Target paragraph = ", word)'''
            
    ay4.legend(handles=[tdocs_patch, tfeat_patch, sword_patch, sdoc_patch], loc='upper right')
    #ay4.set_xticks([-5,0,5])
    #ay4.set_yticks([-5,0,5])
    plt.grid(True)
    plt.show()