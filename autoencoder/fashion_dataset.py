import pandas as pd
from files import ROOT
from sklearn.preprocessing import LabelBinarizer

class FashionDataset():

    def __init__(self):
        # https://www.kaggle.com/paramaggarwal/fashion-product-images-small
        self.label = 'articleType'
        self.n_classes = 143
        self.x = 'id'
        self.x_col = 'x'
        self.y_col = 'y'
        self.data = pd.read_csv(ROOT / 'styles.csv', usecols=range(10))

        encoder = LabelBinarizer()
        y = encoder.fit_transform(self.data[self.label])
        
        self.data[self.x_col] = self.data[self.x].apply(lambda v: '{}.jpg'.format(v))
        self.data[self.y_col] = y.tolist()
