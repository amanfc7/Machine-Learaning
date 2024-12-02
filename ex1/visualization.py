#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
# from pprint import pprint
# import sys

# from ds_load_util import load_dataset

import wine_class_mlp
import second_ds_class_mlp
import congress_class_mlp
import reviews_class_mlp

import wine_class_gpc
import sick_class_gpc
# import reviews_class_gpc
# import reviews_class_sparse_gpc
import reviews_class_sparse_gpc_OLD as reviews_class_sparse_gpc
# import reviews_class_gpc as reviews_class_sparse_gpc
import congress_class_gpc

import wine_class_bbdt
import second_ds_class_bbdt
import congress_class_bbdt
import reviews_class_bbdt



def main():
    best_wine_clfs = []
    for clf in [wine_class_mlp, 
                      wine_class_bbdt, 
                      wine_class_gpc]:
        t0= time.time()
        best_wine_clfs.append(clf.train_model(skip_eval=True))
        print(time.time() - t0)
    plot_evaluation_values('Wine', best_wine_clfs)
    
    
    best_sick_clfs = []
    for clf in [second_ds_class_mlp, 
                      second_ds_class_bbdt, 
                      sick_class_gpc]:
        t0= time.time()
        best_sick_clfs.append(clf.train_model(skip_eval=True))
        print(time.time() - t0)
    plot_evaluation_values('Sick', best_sick_clfs) #average = 'binary' might be better, but issue with encoding
    

    best_congress_clfs = []
    for clf in [congress_class_mlp, 
                          congress_class_bbdt, 
                          congress_class_gpc]:
        t0= time.time()
        best_congress_clfs.append(clf.train_model(skip_eval=True))
        print(time.time() - t0)
    plot_evaluation_values('Congress', best_congress_clfs)  #average = 'binary' might be better, but issue with encoding
    

    best_reviews_clfs = []
    for clf in [
            reviews_class_mlp, 
            reviews_class_bbdt, 
            reviews_class_sparse_gpc
                          ]:
        t0= time.time()
        best_reviews_clfs.append(clf.train_model(skip_eval=True))
        print(time.time() - t0)
    plot_evaluation_values('Reviews', best_reviews_clfs)
    
def plot_evaluation_values(data_set_name, clfs, average='macro'):
    round_scores_to = 3
    accuracy_scores = [clf.score(X_test, y_test) for (clf, X_test, y_test) in clfs]
    precision_scores = [precision_score(y_test, clf.predict(X_test), average=average) for (clf, X_test, y_test) in clfs]
    recall_scores = [recall_score(y_test, clf.predict(X_test), average=average) for (clf, X_test, y_test) in clfs]
    f1_scores = [f1_score(y_test, clf.predict(X_test), average=average)  for (clf, X_test, y_test) in clfs]
    # print(accuracy_scores)
    
    fig, axs = plt.subplots(2, 2)
    clfs = ['MLP', 'DT', 'GPC']
    # bar_labels = ['MLP', 'DT', 'GPC']
    bar_colors = ['r', 'g', 'b']
    
    #accuracy
    axs[0,0].bar(clfs, accuracy_scores, color=bar_colors)
    addlabels(axs[0,0], clfs, [round(score, round_scores_to) for score in accuracy_scores])
    axs[0,0].set_title("Accuracy for %s" % data_set_name)
    
    #precision
    axs[0,1].bar(clfs, precision_scores, color=bar_colors)
    addlabels(axs[0,1], clfs, [round(score, round_scores_to) for score in precision_scores])
    axs[0,1].set_title("Precision for %s" % data_set_name)
    
    #recall
    axs[1,0].bar(clfs, recall_scores, color=bar_colors)
    addlabels(axs[1,0], clfs, [round(score, round_scores_to) for score in recall_scores])
    axs[1,0].set_title("Recall for %s" % data_set_name)
    
    #F1
    axs[1,1].bar(clfs, accuracy_scores, color=bar_colors)
    addlabels(axs[1,1], clfs, [round(score, round_scores_to) for score in f1_scores])
    axs[1,1].set_title("F1 for %s" % data_set_name)
    
    plt.show()
    
    
# function to add value labels
def addlabels(a_plot, x,y):
    for i in range(len(x)):
        a_plot.text(i, y[i], y[i], ha = 'center')




if __name__ == '__main__':
    main()