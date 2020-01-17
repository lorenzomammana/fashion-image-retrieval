import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

class FashionSequence(keras.utils.Sequence):

    def __init__(self, data, img_size, img_dir, n_classes, batch_size=32, shuffle=True):

        self.data = data
        self.img_size = img_size
        self.img_dir = img_dir
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x_index = 10
        self.y_index = 11

        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):

        indices = self.indexes[index * self.batch_size:(index + 1 ) * self.batch_size]

        x = np.ndarray(shape=(self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]))
        y = np.ndarray(shape=(self.batch_size, self.n_classes))

        for idx, i in enumerate(indices):
            
            img = load_img(self.img_dir / self.data.iloc[i, self.x_index], target_size=self.img_size)
            img = img_to_array(img)
            img /= 255.

            x[idx] = img
            y[idx] = self.data.iloc[i, self.y_index]

        return x, [x, y]

    def on_epoch_end(self):

        self.indexes = np.arange(self.data.shape[0])

        if self.shuffle == True:
            np.random.shuffle(self.indexes)
