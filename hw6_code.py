# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:40:11 2021

@author: ruchuan2
"""

import pandas as pd
import numpy as np

####################################################### Processing the data ############################################

data = pd.read_csv("C:/Users/ruchuan2/Box/IE 517 Machine Learning in FIN Lab/HW6/ccdefault.csv", header='infer')


################################ Part 1: split the training and test sets ##############################


from sklearn.model_selection import train_test_split


# Split the dataset into a training and a testing set
### with loop

X, y = data.iloc[:,0:24].values, data.iloc[:,24]


from sklearn import tree
from sklearn import metrics

in_sample = []
out_sample = []

for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    tree_model = tree.DecisionTreeClassifier(criterion='gini', 
                                    max_depth=5)
    tree_model.fit(X_train, y_train)
    y_train_pred = tree_model.predict(X_train)
    y_test_pred = tree_model.predict(X_test)
    in_sample.append(metrics.accuracy_score(y_train, y_train_pred))  ## in sample
    out_sample.append(metrics.accuracy_score(y_test, y_test_pred))  ## output sample
    
in_sample_mean = np.mean(in_sample)
in_sample_std_dev = np.std(in_sample)  

out_sample_mean = np.mean(out_sample)
out_sample_std_dev = np.std(out_sample)  
################################### Part 2: Cross validation ######################################
from sklearn.model_selection import cross_val_score

in_sample_cv_scores = cross_val_score(estimator=tree_model,
                            X=X_train,
                            y=y_train,
                            cv=10,
                            n_jobs=1)

in_sample_cv_mean = np.mean(in_sample_cv_scores)
in_sample_cv_std_dev = np.std(in_sample_cv_scores) 
     
out_sample_cv_scores = cross_val_score(estimator=tree_model,
                            X=X_test,
                            y=y_test,
                            cv=10,
                            n_jobs=1)

out_sample_cv_mean = np.mean(out_sample_cv_scores)
out_sample_cv_std_dev = np.std(out_sample_cv_scores) 


