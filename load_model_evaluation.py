# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:44:12 2020
"""

from gensim import models
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import max_error

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix



if __name__ == '__main__':
    
    #C:\Users\RIKO\AnacondaProjects\AutoLabel_by_PV_newswebsite_ver2\DB
    ####### Get data from CSV ########
    economic = pd.read_csv('/Users/USER/Anacondaprojects/AutoLabel__Thainews_PV_complete_ver2-main/AutoLabel__Thainews_PV_complete_ver2-main/news.sql (2)/economic_con.csv')
    education = pd.read_csv('/Users/USER/Anacondaprojects/AutoLabel__Thainews_PV_complete_ver2-main/AutoLabel__Thainews_PV_complete_ver2-main/news.sql (2)/education_con.csv')
    entertainment = pd.read_csv('/Users/USER/Anacondaprojects/AutoLabel__Thainews_PV_complete_ver2-main/AutoLabel__Thainews_PV_complete_ver2-main/news.sql (2)/entertainment_con.csv')
    foreign = pd.read_csv('/Users/USER/Anacondaprojects/AutoLabel__Thainews_PV_complete_ver2-main/AutoLabel__Thainews_PV_complete_ver2-main/news.sql (2)/foreign_con.csv')
    it = pd.read_csv('/Users/USER/Anacondaprojects/AutoLabel__Thainews_PV_complete_ver2-main/AutoLabel__Thainews_PV_complete_ver2-main/news.sql (2)/it_con.csv')
    sports = pd.read_csv('/Users/USER/Anacondaprojects/AutoLabel__Thainews_PV_complete_ver2-main/AutoLabel__Thainews_PV_complete_ver2-main/news.sql (2)/sports_con.csv')
    #all_pd = pd.concat([economic,education,entertainment,foreign,it,sports],axis=1)
    #all_pd = pd.merge([economic,education,entertainment,foreign,it,sports])
    all_pd = pd.concat([foreign,sports],axis=1)
    print(all_pd.columns)
    
    ####### Load model ########
    model = models.doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    model_loaded_deepcut = models.doc2vec.Doc2Vec.load('model_deepcut_test1_3.bin')
    model_loaded_attacut = models.doc2vec.Doc2Vec.load('model_attacut_test1_3.bin')
    
    ####### Create lists of labels and vectors ########
    all_label_vec_deepcut = []
    all_list_deepcut = []
    all_label_vec_attacut = []
    all_list_attacut = []
    
    '''cnt_eco = 0
    for i in all_pd['economic'].values:
        eco_vec_deepcut = model_loaded_deepcut.docvecs['economic_%s' %(cnt_eco)]
        eco_vec_attacut = model_loaded_attacut.docvecs['economic_%s' %(cnt_eco)]
        #print('%s' %(eco_vec_deepcut))
        #print('%s' %(eco_vec_attacut))
        all_list_deepcut.append(eco_vec_deepcut)
        all_label_vec_deepcut.append(0)
        all_list_attacut.append(eco_vec_attacut)
        all_label_vec_attacut.append(0)
        cnt_eco += 1'''
        
        
    '''cnt_edu = 0
    for i in all_pd['education'].values:
        edu_vec_deepcut = model_loaded_deepcut.docvecs['education_%s' %(cnt_edu)]
        edu_vec_attacut = model_loaded_attacut.docvecs['education_%s' %(cnt_edu)]
        #print('%s' %(edu_vec_deepcut))
        #print('%s' %(edu_vec_attacut))
        all_list_deepcut.append(edu_vec_deepcut)
        all_label_vec_deepcut.append(0)
        all_list_attacut.append(edu_vec_attacut)
        all_label_vec_attacut.append(0)
        cnt_edu += 1'''
        
    '''cnt_ent = 0
    for i in all_pd['entertainment'].values:
        ent_vec_deepcut = model_loaded_deepcut.docvecs['entertainment_%s' %(cnt_ent)]
        ent_vec_attacut = model_loaded_attacut.docvecs['entertainment_%s' %(cnt_ent)]
        #print('%s' %(ent_vec_deepcut))
        #print('%s' %(ent_vec_attacut))
        all_list_deepcut.append(ent_vec_deepcut)
        all_label_vec_deepcut.append(0)
        all_list_attacut.append(ent_vec_attacut)
        all_label_vec_attacut.append(0)
        cnt_ent += 1'''
        
    cnt_fore = 0
    for i in all_pd['foreign'].values:
        fore_vec_deepcut = model_loaded_deepcut.docvecs['foreign_%s' %(cnt_fore)]
        fore_vec_attacut = model_loaded_attacut.docvecs['foreign_%s' %(cnt_fore)]
        #print('%s' %(fore_vec_deepcut))
        #print('%s' %(fore_vec_attacut))
        all_list_deepcut.append(fore_vec_deepcut)
        all_label_vec_deepcut.append(0)
        all_list_attacut.append(fore_vec_attacut)
        all_label_vec_attacut.append(0)
        cnt_fore += 1
        
    '''cnt_it = 0
    for i in all_pd['it'].values:
        it_vec_deepcut = model_loaded_deepcut.docvecs['it_%s' %(cnt_it)]
        it_vec_attacut = model_loaded_attacut.docvecs['it_%s' %(cnt_it)]
        #print('%s' %(it_vec_deepcut))
        #print('%s' %(it_vec_attacut))
        all_list_deepcut.append(it_vec_deepcut)
        all_label_vec_deepcut.append(0)
        all_list_attacut.append(it_vec_attacut)
        all_label_vec_attacut.append(0)
        cnt_it += 1'''
        
    cnt_spo = 0
    for i in all_pd['sports'].values:
        spo_vec_deepcut = model_loaded_deepcut.docvecs['sports_%s' %(cnt_spo)]
        spo_vec_attacut = model_loaded_attacut.docvecs['sports_%s' %(cnt_spo)]
        #print('%s' %(spo_vec_deepcut))
        #print('%s' %(spo_vec_attacut))
        all_list_deepcut.append(spo_vec_deepcut)
        all_label_vec_deepcut.append(1)
        all_list_attacut.append(spo_vec_attacut)
        all_label_vec_attacut.append(1)
        cnt_spo += 1
        

    #print(all_list_deepcut)
    #print(all_label_vec_deepcut)
    #print(all_list_attacut)
    #print(all_label_vec_attacut)
    
    
    ####### Evaluation: Precision, Recall, F-score ########
    data_train_deepcut = all_list_deepcut
    label_train_deepcut = all_label_vec_deepcut
    data_train_attacut = all_list_attacut
    label_train_attacut = all_label_vec_attacut
    estimator = SVC()
    #estimator = SVC(kernel='linear', C=0.01)
    #target = ['economic','education','entertainment','foreign','it','sports']
    #target1 = ['eco','edu','ent','fore','it','spo']
    target = ['foreign','sports']
    target1 = ['fore','spo']
    
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    
    
    ######### Deepcut ##########
    data_train_s_deepcut, data_test_s_deepcut, label_train_s_deepcut, label_test_s_deepcut = train_test_split(data_train_deepcut, label_train_deepcut, test_size=0.2)
    estimator.fit(data_train_s_deepcut, label_train_s_deepcut)
    actual_deepcut, predicted_deepcut = label_test_s_deepcut, estimator.predict(data_test_s_deepcut)
    y_true_deepcut, y_pred_deepcut = label_test_s_deepcut, estimator.predict(data_test_s_deepcut)
    print("Deepcut")
    print (metrics.classification_report(actual_deepcut, predicted_deepcut, target_names=target))
    print(accuracy_score(actual_deepcut, predicted_deepcut))
    print(multilabel_confusion_matrix(actual_deepcut, predicted_deepcut))
    print(max_error(actual_deepcut, predicted_deepcut))
    #plot_confusion_matrix(estimator, data_test_s_deepcut, label_test_s_deepcut)  # doctest: +SKIP
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(estimator, data_test_s_deepcut, label_test_s_deepcut, display_labels=target1, cmap=plt.cm.YlOrRd, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()  # doctest: +SKIP
    #-------------
    print("")
    #-------------
    
    ######### Attacut ##########
    data_train_s_attacut, data_test_s_attacut, label_train_s_attacut, label_test_s_attacut = train_test_split(data_train_attacut, label_train_attacut, test_size=0.2)
    estimator.fit(data_train_s_attacut, label_train_s_attacut)
    actual_attacut, predicted_attacut = label_test_s_attacut, estimator.predict(data_test_s_attacut)
    y_true_attacut, y_pred_attacut = label_test_s_attacut, estimator.predict(data_test_s_attacut)
    print("Attacut")
    print (metrics.classification_report(actual_attacut, predicted_attacut, target_names=target))
    print(accuracy_score(actual_attacut, predicted_attacut))
    print(multilabel_confusion_matrix(actual_attacut, predicted_attacut))
    print(max_error(actual_attacut, predicted_attacut))
    #plot_confusion_matrix(estimator, data_test_s_attacut, label_test_s_attacut)  # doctest: +SKIP
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(estimator, data_test_s_attacut, label_test_s_attacut, display_labels=target1, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        print(title)
        print(disp.confusion_matrix)
    plt.show()  # doctest: +SKIP
    
    
    #https://dsdi.msu.ac.th/?article=data-science&fn=train_test_split
    #https://fzr72725.github.io/2018/01/14/genism-guide.html
    #https://sysadmin.psu.ac.th/2019/01/07/python-03-train-validation-test-and-accuracy-assessment-with-confusion-matrix/
