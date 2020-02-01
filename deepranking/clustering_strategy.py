import joblib
import files
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sklearn.metrics.cluster as cscores
from sklearn import metrics

class ClusteringStrategy():

    def __init__(self, k, batch_size):

        self.k = k
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=self.batch_size)
        self.cluster_labels = []
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

    def compute_scores(self, x):
        
        self.cluster_labels = np.ndarray((x.shape[0],))

        for i in range(0, x.shape[0], self.batch_size):
            predictions = self.kmeans.predict(x[i:(i + self.batch_size)])
            self.cluster_labels[i:(i + self.batch_size)] = predictions
    
        if (i + self.batch_size) > x.shape[0]:
            predictions = self.kmeans.predict(x[i:x.shape[0]])
            self.cluster_labels[i:x.shape[0]] = predictions

        confusion_matrix = cscores.contingency_matrix(self.labels_true, self.labels_pred)
        purity_score = np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
        homogeneity_score, completeness_score, v_measure_score = cscores.homogeneity_completeness_v_measure(self.labels_true, self.labels_pred)

        scores = [
            #['calinski_harabasz_score', 'internal', cscores.calinski_harabasz_score(x, self.cluster_labels)],
            ['davies_bouldin_score', 'internal', metrics.davies_bouldin_score(x, self.cluster_labels)],
            ['silhouette_score', 'internal', metrics.silhouette_score(x, self.cluster_labels)],
            #['silhouette_samples', 'internal', cscores.silhouette_samples(x, self.cluster_labels)],
            ['purity_score', 'external', purity_score],
            ['adjusted_rand_score', 'external', cscores.adjusted_rand_score(self.labels_true, self.labels_pred)],
            ['completeness_score', 'external', completeness_score],
            ['fowlkes_mallows_score', 'external', cscores.fowlkes_mallows_score(self.labels_true, self.labels_pred)],
            ['homogeneity_score', 'external', homogeneity_score],
            ['adjusted_mutual_info_score', 'external', cscores.adjusted_mutual_info_score(self.labels_true, self.labels_pred)],
            ['mutual_info_score', 'external', cscores.mutual_info_score(self.labels_true, self.labels_pred)],
            ['normalized_mutual_info_score', 'external', cscores.normalized_mutual_info_score(self.labels_true, self.labels_pred)],
            ['v_measure_score', 'external', v_measure_score]
        ]

        scores = pd.DataFrame(scores, columns=['name', 'type', 'score'])
        scores.to_csv(files.small_images_classes_kmeans_scores, index=False)

    def load(self):

        path = files.small_images_classes_kmeans
        self.kmeans = joblib.load(path)

    def save(self):

        path = files.small_images_classes_kmeans
        joblib.dump(self.kmeans, path)
