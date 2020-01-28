import joblib
import files
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sklearn.metrics.cluster as cscores

class ClusteringStrategy():

    def __init__(self, k, batch_size):

        self.k = k
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=self.batch_size)
        self.labels_true = []
        self.labels_pred = []

    def init_centroids(self, c_dict):

        centroids = []
        for _, v in c_dict.items():
            data = np.array(v)
            data = np.mean(data, axis=0)
            centroids.append(data)

        centroids = np.array(centroids)

        self.kmeans = MiniBatchKMeans(n_clusters=self.k, init=centroids, batch_size=self.batch_size)

    def fit_batch(self, x):

        self.kmeans.partial_fit(x)

    def compute_cluster_classes(self, classes_names):

        # Assign each cluster to a specific class
        centroids_class_frequency = []
        for i in range(len(classes_names)):
            class_frequency = {}

            for c in classes_names:
                class_frequency[c] = 0

            centroids_class_frequency.append(class_frequency)

        img_cluster_class = []
        for c in classes_names:

            embedding_dir = files.small_images_classes_embeddings / c
            embedding_files = [v for v in embedding_dir.iterdir() if v.is_file()]

            for f in embedding_files:
                embedding = np.loadtxt(f)
                prediction = self.kmeans.predict(np.array([embedding]))[0]

                self.labels_pred.append(prediction)
                self.labels_true.append(c)

                centroids_class_frequency[prediction][c] += 1
                img_cluster_class.append([f.stem, prediction, c])

        img_cluster_class = pd.DataFrame(img_cluster_class, columns=['id', 'cluster', 'class'])
        img_cluster_class.to_csv(files.small_images_classes_features, index=False)

        centroids_classes = []
        for i in range(len(centroids_class_frequency)):

            max_frequency = (None, -1)

            for k, v in centroids_class_frequency[i].items():

                if v > max_frequency[1]:
                    max_frequency = (k, v)

            centroids_classes.append([i, max_frequency[0]])
            
            for j in range(len(self.labels_true)):

                if self.labels_true[j] == max_frequency[0]:
                    self.labels_true[j] = i

        centroids_classes = pd.DataFrame(centroids_classes, columns=['centroid', 'class'])
        centroids_classes.to_csv(files.small_images_classes_centroids, index=False)

    def compute_scores(self):
        
        confusion_matrix = cscores.contingency_matrix(self.labels_true, self.labels_pred)
        purity_score = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
        homogeneity_score, completeness_score, v_measure_score = cscores.homogeneity_completeness_v_measure(self.labels_true, self.labels_pred)

        scores = [
            ['purity_score', purity_score],
            ['adjusted_rand_score', cscores.adjusted_rand_score(self.labels_true, self.labels_pred)],
            ['completeness_score', completeness_score],
            ['fowlkes_mallows_score', cscores.fowlkes_mallows_score(self.labels_true, self.labels_pred)],
            ['homogeneity_score', homogeneity_score],
            ['mutual_info_score', cscores.mutual_info_score(self.labels_true, self.labels_pred)],
            ['normalized_mutual_info_score', cscores.normalized_mutual_info_score(self.labels_true, self.labels_pred)],
            ['v_measure_score', v_measure_score]
        ]

        scores = pd.DataFrame(scores, columns=['name', 'score'])
        scores.to_csv(files.small_images_classes_kmeans_scores, index=False)

    def load(self):

        path = files.small_images_classes_kmeans
        self.kmeans = joblib.load(path)

    def save(self):

        path = files.small_images_classes_kmeans
        joblib.dump(self.kmeans, path)
