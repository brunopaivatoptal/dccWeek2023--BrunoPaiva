# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 07:23:26 2023

@author: angel
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import h5py

root = Path("data")
filename = root/"train_dccweek2023.h5"

f = h5py.File(filename, 'r')
# Get ids
traces_ids = np.array(f['exam_id'])
x = f['tracings']
## get Labels
classes = {0 :"Nenhum diagnóstico",
           1 :"1dAVb: bloqueio atrioventricular de primeiro grau",
           2 :"RBBB: bloqueio de ramo direito",
           3 :"LBBB: bloqueio de ramo esquerdo",
           4 :"SB: bradicardia sinusal",
           5 :"AF: fibrilação atrial",
           6 :"ST: taquicardia sinusal"}
labels = pd.read_csv(root/"train_dccweek2023-labels.csv").set_index('exam_id').classe.replace(classes)

labels.value_counts(normalize=True)

x_batch = x[0:512]
pickle.dump(x_batch, open(root/'sampleData.pickle', "wb"))

## Explore a single data point
idx = 0
xi = x_batch[idx]
xi.shape
sid = traces_ids[idx]
label = labels.loc[sid]

colnames = ["DI","DII","DIII","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6",]
df_xi = pd.DataFrame(xi, columns=colnames)
plt.figure(figsize=(8,12))
for i,c in enumerate(colnames):
    plt.subplot(6,2,i+1)
    df_xi[c].plot.line()
    plt.title(c)
plt.tight_layout()

x_full = f['tracings'][:]

(x_full.min(), x_full.max())

EPS = 1e-6
x_batchPre = x_batch + EPS
x_batch_pctChange = np.nan_to_num(x_batchPre[1:] / x_batchPre[:-1])
(x_batch.min(), x_batch.max())

idx = 0
xi = x_batch_pctChange[idx]
xi.shape
sid = traces_ids[idx]
label = labels.loc[sid]

colnames = ["DI","DII","DIII","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6",]
df_xi = pd.DataFrame(xi, columns=colnames)
plt.figure(figsize=(8,12))
for i,c in enumerate(colnames):
    plt.subplot(6,2,i+1)
    df_xi[c].plot.line()
    plt.title(c)
plt.tight_layout()


