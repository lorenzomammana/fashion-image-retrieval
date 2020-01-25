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
from keras.models import load_model
import tensorflow as tf
from keras import backend as K


def loss_tensor(y_true, y_pred, batch_size=8):
    total_loss = tf.convert_to_tensor(0, dtype=tf.float32)
    g = tf.constant(1.0, shape=[1], dtype=tf.float32)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    for i in range(0, batch_size, 3):
        try:
            q_embedding = y_pred[i]
            p_embedding = y_pred[i + 1]
            n_embedding = y_pred[i + 2]
            D_q_p = K.sqrt(K.sum((q_embedding - p_embedding) ** 2))
            D_q_n = K.sqrt(K.sum((q_embedding - n_embedding) ** 2))
            loss = tf.maximum(g + D_q_p - D_q_n, zero)
            total_loss = total_loss + loss
        except:
            continue
    total_loss = total_loss / batch_size
    return total_loss


if __name__ == '__main__':
    model = load_model(files.deepranking_weights_path.absolute().as_posix(),
                       custom_objects={'loss_tensor': loss_tensor})

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
            kmeans.partial_fit(np.array(embedding_values).squeeze())
            embedding_values = []

    # Update kmeans
    if len(embedding_values) > 0:
        kmeans.partial_fit(np.array(embedding_values).squeeze())
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

    img_cluster_class = []
    for c in classes_names:

        embedding_dir = files.small_images_classes_embeddings / c
        embedding_files = [v for v in embedding_dir.iterdir() if v.is_file()]

        for f in embedding_files:
            embedding = np.loadtxt(f)
            prediction = kmeans.predict(np.array([embedding]))[0]

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

    centroids_classes = pd.DataFrame(centroids_classes, columns=['centroid', 'class'])
    centroids_classes.to_csv(files.small_images_classes_centroids, index=False)
