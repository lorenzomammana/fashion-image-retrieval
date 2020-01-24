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
from tqdm import tqdm

if __name__ == '__main__':
    model = FashionRankingModel().compile(weights=files.deepranking_weights_path)

    classes_dirs = [d for d in files.small_images_classes_directory.iterdir() if d.is_dir()]
    classes_names = [d.parts[-1] for d in classes_dirs]

    images = []
    for p, c in zip(classes_dirs, classes_names):
        images = images + [i.as_posix().split('/')[-2:] for i in p.iterdir() if i.is_file()]
    n_classes = len(classes_names)
    embedding_values = []

    batch_size = 128
    kmeans = MiniBatchKMeans(n_clusters=n_classes, random_state=0, batch_size=batch_size)

    # For each class
    images = np.random.permutation(images)

    for label, i in tqdm(images):

        # Compute embedding value and save to txt file
        img = load_img(files.small_images_classes_directory / label / i, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        ev = model.predict([img, img, img])[0]
        embedding_values.append(ev)
        np.savetxt(files.small_images_classes_embeddings / label / i.replace('jpg', 'txt'), ev)

        # Update kmeans
        if len(embedding_values) % batch_size == 0:
            kmeans.partial_fit(np.array(embedding_values))
            embedding_values = []

    # Update kmeans
    if len(embedding_values) > 0:
        kmeans.partial_fit(np.array(embedding_values))
        embedding_values = []

    # Save kmeans object to file
    joblib.dump(kmeans, files.small_images_classes_kmeans)

    # kmeans = joblib.load(files.small_images_classes_kmeans)

    # Assign each cluster to a specific class
    centroids_class_frequency = []
    for i in range(len(classes_names)):
        class_frequency = {}

        for c in classes_names:
            class_frequency[c] = 0

        centroids_class_frequency.append(class_frequency)

    file = open(files.small_images_classes_embeddings / 'features.csv', 'w')

    for c in classes_names:

        embedding_dir = files.small_images_classes_embeddings / c
        embedding_files = [v for v in embedding_dir.iterdir() if v.is_file()]

        for f in embedding_files:
            embedding = np.loadtxt(f)
            prediction = kmeans.predict(np.array([embedding]))[0]

            file.write('/'.join(f.as_posix().split('/')[-2:]) + ',' + str(prediction) + ',' + c + '\n')
            centroids_class_frequency[prediction][c] += 1

    file.close()

    centroids_classes = []
    for i in range(len(centroids_class_frequency)):

        max_frequency = (None, -1)

        for k, v in centroids_class_frequency[i].items():

            if v > max_frequency[1]:
                max_frequency = (k, v)

        centroids_classes.append([i, max_frequency[0]])

    centroids_classes = pd.DataFrame(centroids_classes, columns=['centroid', 'class'])
    centroids_classes.to_csv(files.small_images_classes_centroids, index=False)
