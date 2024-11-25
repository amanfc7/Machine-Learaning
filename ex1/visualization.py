#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
import reviews_class_sparse_gpc
import congress_class_gpc




def main():
    # best_wine_class_mlp_clf = wine_class_mlp.train_model()
    # # best_second_class_mlp_clf = second_ds_class_mlp.train_model()
    # # best_congress_class_mlp_clf = congress_class_mlp.train_model()
    # # best_reviews_class_mlp_clf = reviews_class_mlp.train_model()

    # # best_wine_class_dt_clf = None # TODO
    # best_wine_class_dt_clf = best_wine_class_mlp_clf #DUMMY
    # # best_second_class_dt_clf = None
    # # best_congress_class_dt_clf = None
    # # best_reviews_class_dt_clf = None

    # best_wine_class_gpc_clf = wine_class_gpc.train_model()
    # # best_second_class_gpc_clf = None
    # # best_congress_class_gpc_clf = None
    # # best_reviews_class_gpc_clf = None
    

    best_wine_clfs = [wine_class_mlp.train_model(skip_eval=True), 
                      wine_class_mlp.train_model(skip_eval=True),     #TODO: add decision tree clf
                      wine_class_gpc.train_model()]
    plot_evaluation_values('Wine - Unfinished', best_wine_clfs)
    
    #TODO: add other clfs 
    best_sick_clfs = [second_ds_class_mlp.train_model(skip_eval=True), 
                      second_ds_class_mlp.train_model(skip_eval=True), #TODO: add decision tree clf
                      sick_class_gpc.train_model()]
    plot_evaluation_values('Sick - Unfinished', best_sick_clfs)
    

    best_congress_clfs = [congress_class_mlp.train_model(skip_eval=True), 
                          congress_class_mlp.train_model(skip_eval=True),  #TODO: add decision tree clf
                          congress_class_gpc.train_model()]
    plot_evaluation_values('Congress - Unfinished', best_congress_clfs)
    

    best_reviews_clfs = [reviews_class_mlp.train_model(skip_eval=True), 
                         reviews_class_mlp.train_model(skip_eval=True),     #TODO: add decision tree clf
                         reviews_class_sparse_gpc.train_model()]
    plot_evaluation_values('Reviews - Unfinished', best_reviews_clfs)
    
def plot_evaluation_values(data_set_name, clfs):
    round_scores_to = 3
    # average = 'weighted'
    # average = 'micro'
    average = 'macro' #here we see the most difference between the values
    accuracy_scores = [clf.score(X_test, y_test) for (clf, X_test, y_test) in clfs]
    try:
        precision_scores = [precision_score(y_test, clf.predict(X_test)) for (clf, X_test, y_test) in clfs]
    except ValueError:
        precision_scores = [precision_score(y_test, clf.predict(X_test), average=average) for (clf, X_test, y_test) in clfs] #maybe need other average?
    try:
        recall_scores = [recall_score(y_test, clf.predict(X_test)) for (clf, X_test, y_test) in clfs]
    except ValueError:
        recall_scores = [recall_score(y_test, clf.predict(X_test), average=average) for (clf, X_test, y_test) in clfs]
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
    
    #TODO placeholder, add other scores
    axs[1,1].bar(clfs, accuracy_scores, color=bar_colors)
    addlabels(axs[1,1], clfs, [round(score, round_scores_to) for score in accuracy_scores])
    axs[1,1].set_title("TODO for %s" % data_set_name)
    
    plt.show()
    
    
# function to add value labels
def addlabels(a_plot, x,y):
    for i in range(len(x)):
        a_plot.text(i, y[i], y[i], ha = 'center')




if __name__ == '__main__':
    main()