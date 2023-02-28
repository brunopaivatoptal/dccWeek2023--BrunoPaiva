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

x_batch = x[0:1024]
pickle.dump(x_batch, open(root/'sampleData.pickle', "wb"))
