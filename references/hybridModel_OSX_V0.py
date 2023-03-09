#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:55:47 2023

@author: brunobmp
"""

# -*- coding: utf-8 -*-
"""
supervised baseline using LightGBM
"""
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import h5py
import os

## Custom, utilitary functions
import Utils

NUM_AL_ROUNDS=100
NUM_TO_ADD=100
FULL_DATA = True
SAVE = False


root = Path("data")
filename = root/"train_dccweek2023.h5"
testFilename = root/'test_dccweek2023.h5'

x, traces_ids = Utils.load_data(filename)
x_test, traces_ids_test = Utils.load_data(testFilename)

## get Labels
labels = Utils.getLabelsDf(root/"train_dccweek2023-labels.csv", toText=False).reset_index()
tracesDict = dict(zip(traces_ids, range(len(traces_ids))))
labels["idx"] = labels.exam_id.apply(lambda x: tracesDict[x])

if FULL_DATA:
    print("Using FULL DATA!")
    train, test = train_test_split(labels, test_size=0.3, random_state=42)
    initialTrainData = train.sort_values(by=["idx"])
    initialTestData = test.sort_values(by=["idx"])
else:
    print("Using a small sample of the data!")
    initialTrainData = labels.sample(n=1_000).sort_values(by=["idx"])
    initialTestData = labels[~labels.index.isin(initialTrainData.index)].sample(n=1_00).sort_values(by=["idx"])


print("Sampling training data...")
trainIdx = initialTrainData.idx.tolist()
testIdx = initialTestData.idx.tolist()

X_train = np.stack([xi.T for xi in Utils.getDataFromSamples(trainIdx, x)])
X_test = np.stack([xi.T for xi in  Utils.getDataFromSamples(testIdx, x)])
y_train = initialTrainData.classe.values
y_test = initialTestData.classe.values

model = tf.keras.models.load_model("model_3.h5")
model.summary()

embedding = tf.keras.models.Model(model.inputs, model.layers[-2].output)


X_train_embedded =  embedding.predict(np.transpose(X_train, (0, 2, 1)))
X_test_embedded =  embedding.predict(np.transpose(X_test, (0, 2, 1)))


print("training base classifier...")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
clf = lgb.LGBMClassifier(n_jobs=-1, class_weight=None, n_estimators=500).fit(X_train_embedded, y_train)
y_pred = clf.predict(X_test_embedded)
y_pred_nn = model.predict(np.transpose(X_test, (0, 2, 1)))
macroF1 = f1_score(y_test, y_pred_nn.argmax(axis=1), average="macro")
microF1 = f1_score(y_test, y_pred_nn.argmax(axis=1), average="micro")
print("(NN) MACRO:",macroF1, "MICRO:",microF1)

macroF1 = f1_score(y_test, y_pred, average="macro")
microF1 = f1_score(y_test, y_pred, average="micro")
print("(LGBM) MACRO:",macroF1, "MICRO:",microF1)

print("Re-training main classifier...")
X = np.vstack([X_train_embedded, X_test_embedded])
y = np.hstack([y_train, y_test])
finalOpt = lgb.LGBMClassifier(n_jobs=-1, class_weight=None, n_estimators=500)
finalOpt = finalOpt.fit(X, y)

ss = pd.read_csv("data/sample-submission.csv")
X_submit_pre = np.stack([xj for xj in tqdm(x_test[:])])
#X_submit_pre = np.transpose(X_submit_pre, (0, 2, 1))
X_test_submit = embedding.predict(X_submit_pre)
y_pred_submission = finalOpt.predict(X_test_submit)
df_submit = ss.copy()
df_submit.classe = y_pred_submission
df_submit.exam_id = traces_ids_test
df_submit.to_csv("submission_OSX_lgbm.csv", sep=",", index=False)