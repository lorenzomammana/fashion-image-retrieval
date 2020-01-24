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
import pprint

if __name__ == '__main__':
    
    classes_dirs = [d for d in files.small_images_classes_directory.iterdir() if d.is_dir()]
    classes_names = [d.parts[-1] for d in classes_dirs]
    n_classes = len(classes_names)
    kmeans = joblib.load(files.small_images_classes_kmeans)

    # Assign each cluster to a specific class
    centroids_class_frequency = []
    for i in range(len(classes_names)):
        class_frequency = {}

        for c in classes_names:
            class_frequency[c] = 0

        centroids_class_frequency.append(class_frequency)

    for c in classes_names:

        embedding_dir = files.small_images_classes_embeddings / c
        embedding_files = [v for v in embedding_dir.iterdir() if v.is_file()]

        for f in embedding_files:

            embedding = np.loadtxt(f)
            prediction = kmeans.predict(np.array([embedding]))[0]

            centroids_class_frequency[prediction][c] += 1

    centroids_classes = []
    for i in range(len(centroids_class_frequency)):

        max_frequency = (None, -1)

        for k, v in centroids_class_frequency[i].items():
            
            if v > max_frequency[1]:
                max_frequency = (k, v)

        centroids_classes.append([i, max_frequency[0]])

    centroids_classes = pd.DataFrame(centroids_classes, columns=['centroid', 'class'])
    centroids_classes.to_csv(files.small_images_classes_centroids, index=False)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(centroids_class_frequency)
    exit(0)
