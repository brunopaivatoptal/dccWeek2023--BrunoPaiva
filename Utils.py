# -*- coding: utf-8 -*-
"""
A collection of utilitary functions to maximize code reuse.
"""
import matplotlib.pyplot as plt
try:
    import seaborn as sns; sns.set()
except:
    pass
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py

def getSampleIndices(sampledIds, traces_ids):
    """
        sampledIds -> collection of sample exam_id's
    """
    indices = pd.Series(traces_ids)
    indices = indices[indices.isin(sampledIds)]
    return indices.index.tolist()


def getDataFromSamples(sampleIndices : list, tracings : h5py._hl.dataset.Dataset):
    return tracings[sampleIndices]


def load_data(filepath):
    f = h5py.File(filepath, 'r')
    # Get ids
    traces_ids = np.array(f['exam_id'])
    x = f['tracings']
    return x, traces_ids


def getLabelsDf(filepath, toText=True):
    if toText:
        classes = {0 :"Nenhum diagnóstico",
                   1 :"1dAVb: bloqueio atrioventricular de primeiro grau",
                   2 :"RBBB: bloqueio de ramo direito",
                   3 :"LBBB: bloqueio de ramo esquerdo",
                   4 :"SB: bradicardia sinusal",
                   5 :"AF: fibrilação atrial",
                   6 :"ST: taquicardia sinusal"}
        labels = pd.read_csv(filepath).set_index('exam_id').classe.replace(classes)
    else:
        labels = pd.read_csv(filepath).set_index('exam_id').classe
    return labels

def applyFFT(data: np.array, maxFreq=500):
    dataFFT = np.stack([np.stack([np.abs(np.fft.fft(yi))[:maxFreq] for yi in xi.T]).T for xi in tqdm(data)])
    return dataFFT

def getBatchFromLabels(y_batch, traces_ids, x):
    return getDataFromSamples(getSampleIndices(y_batch.index, traces_ids), x)

def featurizeEcg(xi):
    """
        Obtain statistical features from a single ECG.
    """
    means = np.mean(xi, axis=0)
    maxs = np.max(xi, axis=0)
    mins = np.min(xi, axis=0)
    fft = np.fft.fft(xi.mean(axis=1))
    return [np.hstack([maxs-mins,
                       maxs-means,
                       means-mins,
                       fft])]

def frequencyTokenizeEcg(xi, nTokens=1024):
    """
        Obtain most important frequencies as tokens.
    """
    fft = np.fft.fft(xi.mean(axis=1).clip(-1,1))
    return np.argsort(fft)[-nTokens:]
    
def plotEcg(xi, label="normal"):
    plt.figure(figsize=(8,12))
    plt.suptitle(label)
    for i, di in enumerate(xi.T):
        plt.subplot(6,2,i+1)
        plt.plot(xi.T[i])
    plt.tight_layout()
    
def filterSignal(xi):
    rollingMean = pd.Series(xi).rolling(window=16).mean()
    toDrop = rollingMean.shape[0] - rollingMean.dropna().shape[0]
    stabXi = (xi - rollingMean)[toDrop+1:].values
    stabXiNorm = ((stabXi - stabXi.min()) / (stabXi.max() - stabXi.min())) - 0.5
    stabXiNorm = np.nan_to_num(stabXiNorm)
    return stabXiNorm

def plot(xi):
    for xii in xi.T:
        plt.plot(xii)

def preprocess(xi, augment=True):
    """
        perform data augmentation on xi. 
    """
    #xi = np.nan_to_num((xi - xi.mean(axis=0)) / xi.std(axis=0))
    if augment:
        xiT = np.roll(xi, shift=np.random.randint(0, xi.shape[0])//2)
        xiS = np.random.uniform(0.4, 0.999) * xiT
        xiN = xiS + np.random.uniform(size=xi.shape) * 0.01
        return xiN
    else:
        return xi