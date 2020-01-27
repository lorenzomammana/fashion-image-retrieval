import files
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import joblib
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from fashion_utils import loss_tensor, triplet_loss_adapted_from_tf
from keras_applications.resnext import preprocess_input
import keras


class FashionSimilarity:

    def __init__(self):
        self.model = load_model((files.output_directory / 'onlinemining.h5').absolute().as_posix(),
                                custom_objects={'triplet_loss_adapted_from_tf': triplet_loss_adapted_from_tf})
        self.kmeans = joblib.load(files.small_images_classes_kmeans)
        self.centroid_classes = pd.read_csv(files.small_images_classes_centroids)
        self.embeddings = pd.read_csv(files.small_images_classes_features)

    def get_similar_images(self, img_path, n):
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        img = np.expand_dims(img, axis=0)

        ev, label = self.model.predict([img, img, img, np.zeros(1)])
        ev = ev[:, 1:]
        centroid = self.kmeans.predict(ev)[0]

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
