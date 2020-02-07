import deepranking.files as files
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import joblib
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from deepranking.fashion_utils import loss_tensor, triplet_loss_adapted_from_tf
from keras_applications.resnext import preprocess_input
import keras


class FashionSimilarity:

    def __init__(self):
        self.model = load_model((files.output_directory / 'onlinemining_loss.h5').absolute().as_posix(),
                                custom_objects={'triplet_loss_adapted_from_tf': triplet_loss_adapted_from_tf})
        self.kmeans = joblib.load(files.small_images_classes_kmeans)
        self.centroid_classes = pd.read_csv(files.small_images_classes_centroids)
        self.embeddings = pd.read_csv(files.small_images_classes_features)
        # self.centroids_distance = np.ndarray((self.kmeans.cluster_centers_.shape[0], self.kmeans.cluster_centers_.shape[0]))
        # self.nearest_centroids = np.ndarray((self.centroids_distance.shape[0],))

        # Compute centroids distance matrix
        # self.__centroids_distance_matrix__()

    def get_similar_images(self, img_path, n):
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        img = np.expand_dims(img, axis=0)

        ev, c = self.model.predict([img, img, img, np.zeros(1)])
        label = np.argmax(c)
        perc = np.max(c)
        ev = ev[:, 1:]
        centroid = self.kmeans.predict(ev)[0]

        predicted_class = self.centroid_classes[self.centroid_classes['centroid'] == centroid].iloc[0, 1]
        
        data_dir = files.small_images_classes_embeddings

        cluster_data = joblib.load(data_dir / '{}.joblib'.format(centroid))
        cluster_data_ids = joblib.load(data_dir / '{}_ids.joblib'.format(centroid))
        cluster_data_class = joblib.load(data_dir / '{}_class.joblib'.format(centroid))

        similarity_scores = []
        for i in range(cluster_data.shape[0]):

            distance = np.linalg.norm(ev - cluster_data[i])
            similarity_scores.append([cluster_data_ids[i], cluster_data_class[i], distance])

        # cluster_images = self.embeddings[self.embeddings['cluster'] == centroid]

        # similarity_scores = []
        # for i in range(cluster_images.shape[0]):
        #     ev_id = cluster_images.iloc[i, 0]
        #     ev_class = cluster_images.iloc[i, 2]
        #     ev_path = files.small_images_classes_embeddings / ev_class / '{}.txt'.format(ev_id)

        #     current_ev = np.loadtxt(ev_path)
        #     distance = np.linalg.norm(ev - current_ev)

        #     similarity_scores.append([ev_id, ev_class, distance])

        similarity_scores = pd.DataFrame(similarity_scores, columns=['id', 'class', 'score'])
        similarity_scores = similarity_scores.sort_values(by='score', ascending=True)

        max_index = np.min([similarity_scores.shape[0], n])
        similarity_scores = similarity_scores.iloc[0:max_index]

        return predicted_class, similarity_scores, label, perc

    def __centroids_distance_matrix__(self):

        for i in range(self.centroids_distance.shape[0]):
            for j in range(self.centroids_distance.shape[1]):
                
                if i == j:
                    self.centroids_distance[i, j] = np.Inf
                else:
                    self.centroids_distance[i, j] = np.linalg.norm(self.kmeans.cluster_centers_[i] - self.kmeans.cluster_centers_[j])

        self.nearest_centroids = np.argmin(self.centroids_distance, axis=1)
