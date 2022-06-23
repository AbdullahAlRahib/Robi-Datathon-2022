# -*- coding: utf-8 -*-

# **Dataset Import**
"""

import numpy as np
import pandas as pd

dataset=pd.read_csv("train.csv")
dataset1=pd.read_csv("test.csv")

len(dataset. columns)

dataset.shape

dataset.head()

#Info of dataset
dataset.info()

"""# **Handle Missing Value**"""

df=dataset

for i in df.columns:
    if df[i].isnull().sum() > 0:
        print(i)
        print('the total null values are:', df[i].isnull().sum())
        print('the datatype is', df[i].dtypes)
        print()

dataset = dataset.drop(["s53","s54","s55","s56","s57","s59"],axis=1)
dataset1 = dataset1.drop(["s53","s54","s55","s56","s57","s59"],axis=1)

dataset

for i in df.columns:
    if df[i].dtypes == 'object':
        print(i)
        print()
        print('the values are:') 
        print(df[i].value_counts())
        print()
        print()

x = dataset.drop(["id","label"],axis=1)
y_train = dataset["label"]
x_test = dataset1.drop(["id"],axis=1)

x

y_train

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
x['gender']= label_encoder.fit_transform(x['gender']) 
x['s11']= label_encoder.fit_transform(x['s11'])
x['s12']= label_encoder.fit_transform(x['s12'])
x['s58']= label_encoder.fit_transform(x['s58'])

x_test['gender']= label_encoder.fit_transform(x_test['gender']) 
x_test['s11']= label_encoder.fit_transform(x_test['s11'])
x_test['s12']= label_encoder.fit_transform(x_test['s12'])
x_test['s58']= label_encoder.fit_transform(x_test['s58'])

x.head(5)

pd.set_option('display.max_columns', None)
x



x_train = pd.get_dummies(x,drop_first=True)
x_test = pd.get_dummies(x_test,drop_first=True)

x_train.head(2)

x_test.head(2)



x_train

x_train.head(2)

x_test.head()

"""Logistic Regression Model"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)

pred1=model_lr.predict(x_test)

"""Decision Tree Model"""

from sklearn import tree
DT_model= tree.DecisionTreeClassifier()
DT_model.fit(x_train, y_train)

pred2=DT_model.predict(x_test)

"""Random Forest Model"""

from sklearn.ensemble import RandomForestClassifier
RF_model=RandomForestClassifier(n_estimators=100,random_state=1)
RF_model.fit(x_train, y_train)

pred3=RF_model.predict(x_test)

"""xgboost Model"""

import xgboost as xgb
XGB_model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
XGB_model.fit(x_train, y_train)

pred4=XGB_model.predict(x_test)

pred4

submission=dataset1[["id"]]
submission

submission["label"] = pred4

submission

submission.to_csv("Submission_Future Vision_1f35va.csv", index=None)