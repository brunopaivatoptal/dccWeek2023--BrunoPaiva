# -*- coding: utf-8 -*-
"""
Código para gerar a submissão apresentada no kaggle. 

"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
## Custom, utilitary functions
import Utils
import os
import tensorflow as tf
from customEcgGenerator import DataGenerator

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
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

print("GPU",tf.config.list_physical_devices('GPU'))

## PARAMETROS DA PRIMEIRA PARTE
BATCH_SIZE=128
NUM_AL_ROUNDS=100
NUM_TO_ADD=100
FULL_DATA = False
MODEL_FN = "models/model_3.h5"
MODEL_FN_iNT = "models/model__interrupted_3.h5"

## PARAMETROS DA SEGUNDA PARTE
AUGMENTATION_FACTOR=10
TUNING=False
XGB=True
SAVE = False
USE_XGB_GPU = True

## PARAMETROS COMUNS
root = Path("../data")
filename = root/"train_dccweek2023.h5"
testFilename = root/'test_dccweek2023.h5'

h5file, traces_ids = Utils.load_data(filename)
x_test, traces_ids_test = Utils.load_data(testFilename)

## get Labels
labels = Utils.getLabelsDf(root/"train_dccweek2023-labels.csv", toText=False).reset_index()#.sample(n=6000)
tracesDict = dict(zip(traces_ids, range(len(traces_ids))))
labels["idx"] = labels.exam_id.apply(lambda x: tracesDict[x])


train, test = train_test_split(labels, test_size=0.01, random_state=42)
train = train.sort_values(by=["idx"])
test = test.sort_values(by=["idx"])
testSample = test.idx.tolist()
trainSample = train.idx.tolist()



print("Sampling test data...")
X_test = np.stack([Utils.preprocess(xi) for xi in  tqdm(Utils.getDataFromSamples(testSample, h5file))])
y_test = test.classe.values

def block(x, filters=64):
    xT = tf.keras.layers.Conv1D(filters, 3, padding="same", activation="gelu")(x)
    xT2= tf.keras.layers.Conv1D(filters, 3, padding="same", activation="gelu")(xT)
    xf = xT2 + xT
    xf = tf.keras.layers.BatchNormalization()(xf)
    xf = tf.keras.layers.Dropout(0.1)(xf)
    return xf

def blockFFT(x, filters=64):
    xT = tf.keras.layers.Conv1D(filters, 3, padding="same", activation="gelu")(x)
    fft = tf.cast(tf.signal.fft2d(tf.cast(xT, tf.complex64)), tf.float32)
    xT2= tf.keras.layers.Conv1D(filters, 3, padding="same", activation="gelu")(fft)
    xf = xT2 + xT
    xf = tf.keras.layers.BatchNormalization()(xf)
    xf = tf.keras.layers.Dropout(0.1)(xf)
    return xf

if os.path.exists(MODEL_FN):
    print("LOADING PRETRAINED MODEL!")
    clf = tf.keras.models.load_model(MODEL_FN)
else:
    print("USING NEW MODEL !")
    NUM_BLOCKS=5
    VOCABULARY_SZ=1024
    print("Building neural net...")
    inp = tf.keras.layers.Input(shape=(4096,12))
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="gelu")(inp)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="gelu")(x)
    sm = tf.keras.layers.Conv1D(VOCABULARY_SZ, 8, strides=4, padding="same", activation="softmax")(x)
    x = tf.math.argmax(sm, axis=-1)
    x = tf.keras.layers.Embedding(VOCABULARY_SZ, VOCABULARY_SZ)(x)
    ## In here, we add a skip-connection to propagate gradients, seeing as the 
    ## softmax operation from the previous node has no gradient.
    x = tf.keras.layers.BatchNormalization()(x) + sm
    n_filters = VOCABULARY_SZ
    for n in range(NUM_BLOCKS):
        n_filters = 64 * (n + 1)
        x = block(x, filters=n_filters)
        x = blockFFT(x,  filters=n_filters)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="elu")(x)
    x = tf.keras.layers.Dense(7, activation="softmax")(x)
    clf = tf.keras.models.Model(inp, x)
print(clf.summary())

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(labels.classe),
                                     y=labels.classe.values)
cw=dict(zip(range(7), class_weights))

clf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy", metrics=["acc"])

try:
    """
        Aqui, fazemos um pré-treino de 1.000 épocas com data augmentation. 
        Não estamos preocupados com over-fitting, já que nosso data augmentation
        gera praticamente uma variação semi-infinita de amostras.
    """
    print("Training...")
    print("Sampling training data...")
    TrainGenerator = DataGenerator(h5file, train)
    clf.fit(TrainGenerator, 
            validation_data=(X_test, y_test),
            epochs=1000,
            shuffle=True,
            class_weight=cw
            )
    clf.save(MODEL_FN)
except Exception as e:
    print(e)
    print("Skipping training...")
    print("saving model...")
    clf.save(MODEL_FN_iNT)

y_pred_test = clf.predict(X_test).argmax(axis=1)
macroF1 = f1_score(test.classe, y_pred_test, average="macro")
microF1 = f1_score(test.classe, y_pred_test, average="micro")
print("MACRO:",macroF1, "MICRO:",microF1)

### SEGUNDA PARTE -- USAR A REDE NEURAL PRÉ-TREINADA COMO EMBEDDING.
## this is my Base ResNet model.

x, traces_ids = Utils.load_data(filename)
x_test, traces_ids_test = Utils.load_data(testFilename)

## get Labels
labels = Utils.getLabelsDf(root/"train_dccweek2023-labels.csv", toText=False).reset_index()
tracesDict = dict(zip(traces_ids, range(len(traces_ids))))
labels["idx"] = labels.exam_id.apply(lambda x: tracesDict[x])

params = { 'max_depth': [3,6,10, 20, 31, 64],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.5, 0.7],
           'max_leaves':[0, 16, 32, 64, 128]}

## TURN OFF GPU FOR THE SECOND PART>
with tf.device('/cpu:0'):
    model = tf.keras.models.load_model("models/model_3.h5")
    model.summary()

embedding = tf.keras.models.Model(model.inputs, model.layers[-2].output)

def getAugmentedDataset(X_train, embedding):
    return embedding.predict(np.transpose(np.stack([Utils.preprocess(xi, augment=True) for xi in X_train]), (0, 2, 1)))
if FULL_DATA:
    print("Using FULL DATA!")
    train, test = train_test_split(labels, test_size=0.3, random_state=42)
    initialTrainData = train.sort_values(by=["idx"])
    initialTestData = test.sort_values(by=["idx"])
else:
    print("Using a small sample of the data!")
    initialTrainData = labels.sample(n=5_000).sort_values(by=["idx"])
    initialTestData = labels[~labels.index.isin(initialTrainData.index)].sample(n=1_000).sort_values(by=["idx"])

trainIdx = initialTrainData.idx.tolist()
testIdx = initialTestData.idx.tolist()
X_train = np.stack([xi.T for xi in Utils.getDataFromSamples(trainIdx, x)])
X_test = np.stack([xi.T for xi in  Utils.getDataFromSamples(testIdx, x)])
y_train = initialTrainData.classe.values
y_test = initialTestData.classe.values

with tf.device('/cpu:0'):
    X_train_embedded =  [embedding.predict(np.transpose(X_train, (0, 2, 1)))] +\
                        [getAugmentedDataset(X_train, embedding) for augStep in tqdm(range(AUGMENTATION_FACTOR))]
    X_train_embedded = np.vstack(X_train_embedded)
    
with tf.device('/cpu:0'):
    X_test_embedded =  embedding.predict(np.transpose(X_test, (0, 2, 1)))

y_train_aug = np.hstack([y_train for augStep in tqdm(range(AUGMENTATION_FACTOR + 1))])

dfTrain = pd.DataFrame({"X_train_aug":[xi for xi in X_train_embedded],
                        "y_train_aug":y_train_aug})
dfTrain.to_parquet("data/DfAugmentedTrain.gzip")

print("training base classifier...")
if TUNING:
    print(f"Using RandomizedSearchCv and XGB={XGB}")
    base = {True:xgb.XGBClassifier(n_jobs=-1, n_estimators=500),
            False:lgb.LGBMClassifier(n_jobs=-1, n_estimators=500)}
    print("Using",base[XGB])
    opt = RandomizedSearchCV(base[XGB], params)
    clf = opt.fit(X_train_embedded, y_train_aug)
else:
    print(f"Using direct training and XGB={XGB}")
    if XGB:
        if USE_XGB_GPU:
            clf = xgb.XGBClassifier(n_jobs=-1, n_estimators=500, tree_method="gpu_hist").fit(X_train_embedded, y_train_aug)
        else:
            clf = xgb.XGBClassifier(n_jobs=-1, n_estimators=500).fit(X_train_embedded, y_train_aug)
    else:
        clf = lgb.LGBMClassifier(n_jobs=-1, n_estimators=500).fit(X_train_embedded, y_train_aug)

y_pred = clf.predict(X_test_embedded)
macroF1 = f1_score(y_test, y_pred, average="macro")
microF1 = f1_score(y_test, y_pred, average="micro")
print("(XGBM) MACRO:",macroF1, "MICRO:",microF1)

print("Re-training main classifier...")
X = np.vstack([X_train_embedded, X_test_embedded])
y = np.hstack([y_train_aug, y_test])
#finalOpt = lgb.LGBMClassifier(n_jobs=-1, class_weight=None, n_estimators=500)

#finalOpt=clf
if TUNING:
    base = {True:xgb.XGBClassifier(n_jobs=-1, n_estimators=500),
            False:lgb.LGBMClassifier(n_jobs=-1, n_estimators=500)}
    finalOpt = RandomizedSearchCV(base[XGB], params)
else:
    if XGB:
        if USE_XGB_GPU:
            finalOpt = xgb.XGBClassifier(n_jobs=-1, n_estimators=500, tree_method="gpu_hist")
        else:
            finalOpt = xgb.XGBClassifier(n_jobs=-1, n_estimators=500)
    else:
        finalOpt = lgb.LGBMClassifier(n_jobs=-1, n_estimators=500)
finalOpt = finalOpt.fit(X, y)

ss = pd.read_csv("data/sample-submission.csv")
X_submit_pre = np.stack([xj for xj in tqdm(x_test[:])])
X_test_submit = embedding.predict(X_submit_pre)
y_pred_submission = finalOpt.predict(X_test_submit)
df_submit = ss.copy()
df_submit.classe = y_pred_submission
df_submit.exam_id = traces_ids_test
df_submit.to_csv("submissions/submission.csv", sep=",", index=False)

if SAVE:
    pickle.dump(clf, open(f"models/boosting_intermediate_Tuning={TUNING}.pickle","wb"))
    pickle.dump(finalOpt, open(f"models/boosting_Tuning={TUNING}.pickle","wb"))
