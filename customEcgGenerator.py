import tensorflow.keras as keras
import numpy as np

import Utils

class DataGenerator(keras.utils.Sequence):
    """
        'Generates data for Keras'
        self = DataGenerator(h5file, labels, traces_ids)
    """
    def __init__(self, h5file, trainLabels,
                 batch_size=128, dim=(4096,12), n_channels=1,
                 n_classes=7, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = trainLabels.sort_values(by=["idx"])
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.h5file = h5file
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.labels.iloc[index*self.batch_size:(index+1)*self.batch_size].sort_values(by=["idx"])

        # Find list of IDs
        y = indexes.classe.values
        X = np.stack([Utils.preprocess(xi, augment=np.random.uniform() > 0.5)
                      for xi in  Utils.getDataFromSamples(indexes.idx.tolist(), self.h5file)])
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.labels = self.labels.sample(frac=1)
