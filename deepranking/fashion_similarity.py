import fashion_utils
import files
import fashion_dataset
from fashion_ranking_model import FashionRankingModel
from sklearn.cluster import MiniBatchKMeans
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from skimage import transform
import numpy as np
import os
import shutil
import joblib

class FashionSimilarity():

    def __init__(self):

        self.model = FashionRankingModel().compile(weights=files.deepranking_weights_path)
        self.kmeans = joblib.load(files.small_images_classes_kmeans)
        self.centroid_classes = pd.read_csv(files.small_images_classes_centroids)

    def get_similar_images(self, img_path, n):

        img = load_img(img_path)
        img = img_to_array(img).astype('float64')
        img = transform.resize(img, (224, 224))
        img *= 1. / 255
        img = np.expand_dims(img, axis=0)
            
        ev = self.model.predict([img, img, img])[0]
        centroid = self.kmeans.predict(np.array([ev]))[0]

        predicted_class = self.centroid_classes[self.centroid_classes['centroid'] == centroid].iloc[0, 1]

        # TODO retrieve `n` similar images using embedding values saved

        return predicted_class

