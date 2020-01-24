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

if __name__ == '__main__':
    
    model = FashionRankingModel().compile(weights=files.deepranking_weights_path)

    classes_dirs = [d for d in files.small_images_classes_directory.iterdir() if d.is_dir()]
    classes_names = [d.parts[-1] for d in classes_dirs]
    n_classes = len(classes_names)
    embedding_values = []

    batch_size = 128
    kmeans = MiniBatchKMeans(n_clusters=n_classes, random_state=0, batch_size=batch_size)

    # For each class
    for p, c in zip(classes_dirs, classes_names):

        images = [i for i in p.iterdir() if i.is_file()]
        images_names = [i.stem for i in images]
        
        embedding_dir = files.small_images_classes_embeddings / c

        shutil.rmtree(embedding_dir, ignore_errors=True)
        os.mkdir(embedding_dir)

        # Compute embedding value and save to txt file
        for i, n in zip(images, images_names):

            img = load_img(i)
            img = img_to_array(img).astype('float64')
            img = transform.resize(img, (224, 224))
            img *= 1. / 255
            img = np.expand_dims(img, axis=0)
            
            ev = model.predict([img, img, img])[0]
            embedding_values.append(ev)
            np.savetxt(embedding_dir / '{}.txt'.format(n), ev)

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
