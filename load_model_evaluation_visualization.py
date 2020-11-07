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

#http://csmoon-ml.com/index.php/2019/02/15/tutorial-doc2vec-and-t-sne/
#https://mlexplained.com/2018/09/14/paper-dissected-visualizing-data-using-t-sne-explained/
######https://leightley.com/visualizing-tweets-with-word2vec-and-t-sne-in-python/


if __name__ == '__main__':
    #model = models.doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)
    
    
    
    ####### Load model ########
    #model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_attacut_test_10') #test
    model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_deepcut_test1')
    #model_loaded_attacut = models.doc2vec.Doc2Vec.load('model_attacut_test1')
    print(model_loaded_deepcut)
    #print(model_loaded_attacut)
    
    #doc_tags = list(model_loaded_deepcut.docvecs.doctags.keys())
    #X = model_loaded_deepcut[doc_tags]
    #print(X)
    
    '''tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    X_tsne_df = pd.DataFrame(data=X_tsne, index=doc_tags, columns=['x', 'y'])
    #print(df)'''
    
    
    ################ Vector Space Model ################
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
    ax.set_yticklabels([]) #Hide ticks
    ax.set_xticklabels([]) #Hide ticks
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        #plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]))
    plt.show()'''
    
    
    
    ################ Cosine similarity  Feature ################
    ddoc0 = 'it_458' 
    ddoc1 = 'it_183'
    dvec0 = model_loaded_deepcut.docvecs[ddoc0]
    dvec1 = model_loaded_deepcut.docvecs[ddoc1]
    dtasu1 = (dvec0+dvec1)
    docs = list(model_loaded_deepcut.docvecs.most_similar([ddoc0]))
    words_docs = list(model_loaded_deepcut.similar_by_word(dvec0, topn=10, restrict_vocab=None))
    #print(words_docs)
    
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
    ax.set_title('DOC2VEC - PCA')
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
    ax.plot(result_X[:, 0], result_X[:, 1], 'o')
    ax.plot(result_Y[:, 0], result_Y[:, 1], 'o')
    plt.show()
    
    
    
    
    
    
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