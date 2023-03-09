# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:22:36 2023

@author: angel
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pathlib import Path
import lightgbm as lgb
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import h5py
import os

## Custom, utilitary functions
import Utils

FFT_BATCH_SIZE=128
NUM_AL_ROUNDS=100
NUM_TO_ADD=100
FULL_DATA = True
SAVE=False

root = Path("data")
filename = root/"train_dccweek2023.h5"
testFilename = root/'test_dccweek2023.h5'

x, traces_ids = Utils.load_data(filename)
x_test, traces_ids_test = Utils.load_data(testFilename)

## get Labels
labels = Utils.getLabelsDf(root/"train_dccweek2023-labels.csv")
labels.value_counts()


normalIdx = Utils.getSampleIndices(labels[labels == "Nenhum diagnóstico"].sample(n=1).index, traces_ids)
normal = Utils.getDataFromSamples(normalIdx, x)[0]
Utils.plotEcg(normal, label="Nenhum diagnóstico")


ddx = "ST: taquicardia sinusal"
normalIdx = Utils.getSampleIndices(labels[labels == ddx].sample(n=1).index, traces_ids)
normal = Utils.getDataFromSamples(normalIdx, x)[0]
Utils.plotEcg(normal, label=ddx)

ddx = "SB: bradicardia sinusal"
normalIdx = Utils.getSampleIndices(labels[labels == ddx].sample(n=1).index, traces_ids)
normal = Utils.getDataFromSamples(normalIdx, x)[0]
Utils.plotEcg(normal, label=ddx)

