# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:00:42 2019

@author: THIS PC
"""
import os
cwd = os.getcwd()

import pandas as pd
import numpy as np
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

#K - means clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2) #generating cluster centers
model = kmeans.fit(train_x)
prediction = kmeans.predict(test_x)

test['predictions'] = prediction #add predictions to test set

reorder = test.loc[test['predictions'] == 1] #predictions == 1

next_order = reorder[["user_id", "order_id", "product_id", "product_name"]]

next_order.to_csv(r'C:\\Users\\THIS PC\\Desktop\\kmeans.csv',index=False)