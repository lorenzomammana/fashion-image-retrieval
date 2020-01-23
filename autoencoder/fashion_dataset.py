import pandas as pd
import numpy as np
from autoencoder.files import ROOT
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

class FashionDataset():

    def __init__(self, img_dir):
        # https://www.kaggle.com/paramaggarwal/fashion-product-images-small
        self.label = 'articleType'
        self.x = 'id'
        self.x_col = 'x'
        self.y_col = 'y'
        self.label_filters = [
            'Handbags', 'Belts', 'Casual Shoes', 
            'Sports Shoes', 'Heels', 'Sandals', 
            'Formal Shoes', 'Flats', 'Dresses', 
            'Sunglasses', 'Jeans', 'Trousers', 
            'Tshirts', 'Shirts', 'Tops', 'Shorts'
        ]
        self.n_classes = len(self.label_filters)
        self.data = pd.read_csv(ROOT / 'styles.csv', usecols=range(10))
        self.data[self.x_col] = self.data[self.x].apply(lambda v: '{}.jpg'.format(v))

        drop_idx = []
        for i in range(self.data.shape[0]):

            if not (img_dir / self.data.iloc[i, 10]).exists():
                drop_idx.append(i)

        self.data = self.data.drop(drop_idx)

        self.data = self.data[(self.data[self.label] == self.label_filters[0]) | 
            (self.data[self.label] == self.label_filters[1]) |
            (self.data[self.label] == self.label_filters[2]) |
            (self.data[self.label] == self.label_filters[3]) |
            (self.data[self.label] == self.label_filters[4]) |
            (self.data[self.label] == self.label_filters[5]) |
            (self.data[self.label] == self.label_filters[6]) |
            (self.data[self.label] == self.label_filters[7]) |
            (self.data[self.label] == self.label_filters[8]) |
            (self.data[self.label] == self.label_filters[9]) |
            (self.data[self.label] == self.label_filters[10]) |
            (self.data[self.label] == self.label_filters[11]) |
            (self.data[self.label] == self.label_filters[12]) |
            (self.data[self.label] == self.label_filters[13]) |
            (self.data[self.label] == self.label_filters[14]) |
            (self.data[self.label] == self.label_filters[15])
        ]

        encoder = LabelBinarizer()
        y = encoder.fit_transform(self.data[self.label])
        
        self.data[self.y_col] = y.tolist()

        y_integers = np.argmax(y, axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        self.class_weights = dict(enumerate(class_weights))

