# -*- coding: utf-8 -*-
"""
supervised baseline using LightGBM
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