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
from keras.applications.vgg16 import preprocess_input

class FashionSimilarity():

    def __init__(self):

        self.model = FashionRankingModel().compile(weights=files.deepranking_weights_path)
        self.kmeans = joblib.load(files.small_images_classes_kmeans)
        self.centroid_classes = pd.read_csv(files.small_images_classes_centroids)
        self.embeddings = pd.read_csv(files.small_images_classes_features)

    def get_similar_images(self, img_path, n):

        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        ev = self.model.predict([img, img, img])[0]
        centroid = self.kmeans.predict(np.array([ev]))[0]

        predicted_class = self.centroid_classes[self.centroid_classes['centroid'] == centroid].iloc[0, 1]
        cluster_images = self.embeddings[self.embeddings['cluster'] == centroid]

        similarity_scores = []
        for i in range(cluster_images.shape[0]):
            
            ev_id = cluster_images.iloc[i, 0]
            ev_class = cluster_images.iloc[i, 2]
            ev_path = files.small_images_classes_embeddings / ev_class / '{}.txt'.format(ev_id)

            current_ev = np.loadtxt(ev_path)
            distance = np.linalg.norm(ev - current_ev)

            similarity_scores.append([ev_id, ev_class, distance])

        similarity_scores = pd.DataFrame(similarity_scores, columns=['id', 'class', 'score'])
        similarity_scores = similarity_scores.sort_values(by='score', ascending=True)

        max_index = np.min([similarity_scores.shape[0], n])
        similarity_scores = similarity_scores.iloc[0:max_index]

        return predicted_class, similarity_scores
