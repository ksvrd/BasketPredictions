# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:39:29 2019

@author: Steff
"""
import os
cwd = os.getcwd()

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
import random

random.seed( 30 )

df = pd.read_csv(r'C:\\Users\\THIS PC\\Desktop\\data\\combined.csv', encoding = "ISO-8859-1")

#selecting training set to split .8, .2: train & test sets
df_train = df.loc[df['eval_set'] == 'train']
df_prior = df.loc[df['eval_set'] == 'prior']

#splitting train data into train and test
df_train_rand = pd.DataFrame(np.random.randn(419313, 2))
num = np.random.rand(len(df_train_rand)) < 0.8

train = df_train[num] #train data
train = pd.concat([train, df_prior], axis = 0) #combining train data with prior data
test = df_train[~num] #test data

predictors = ["add_to_cart_order", "order_dow", "order_hour_of_day", "days_since_prior_order", "count(user_id)"]

#train predictors & target
train_x = train[predictors]
train_y = train[["reordered"]]

#test predictors & target
test_y = test[["reordered"]]
test_x = test[predictors]

#Multivariate Logistic Regression

logmodel = linear_model.LogisticRegression()

logmodel.fit(train_x, train_y)

coef = logmodel.coef_

print(list(zip(coef[0], predictors))) #coeffienct, predictors combination

#predicting new data & extracting target
predictions = logmodel.predict_proba(test_x)
predictions_target = predictions[:,1]

# Calculated the AUC value
auc = roc_auc_score(test_y, predictions_target)
print(round(auc,2))

test['predictions'] = predictions_target #add predictions to test set

reorder = test.loc[test['predictions'] >= .5] #high probability predictions
next_order = reorder[["user_id", "order_id", "product_id", "product_name"]]

next_order.to_csv(r'C:\\Users\\THIS PC\\Desktop\\logregression.csv',index=False)