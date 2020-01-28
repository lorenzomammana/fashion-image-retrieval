import files
import pandas as pd
import numpy as np
import joblib
import keras
from clustering_strategy import ClusteringStrategy
from tqdm import tqdm
from keras.models import load_model
from fashion_utils import loss_tensor
from keras.preprocessing.image import load_img, img_to_array
from keras_applications.resnext import preprocess_input

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
    strategy = ClusteringStrategy(n_classes, batch_size)

    # For each class
    images = np.random.permutation(images)

    print('Computing embedding...')
    
    sample_per_class_needed = 10
    centroids_initialization = {}
    for c in classes_names:
        centroids_initialization[c] = []
    
    embedding_files = []

    for label, i in tqdm(images):

        # Compute embedding value and save to txt file
        img = load_img(files.small_images_classes_directory / label / i, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        img = np.expand_dims(img, axis=0)

        ev = model.predict([img, img, img])[0]
        embedding_values.append(ev)
        embedding_path = files.small_images_classes_embeddings / label / i.replace('jpg', 'txt')
        embedding_files.append(embedding_path)
        np.savetxt(embedding_path, ev)

        if len(centroids_initialization[label]) < sample_per_class_needed:
            centroids_initialization[label].append(ev.squeeze())

    print('Computing clustering...')

    # Init clustering centroids
    strategy.init_centroids(centroids_initialization)

    embedding_values = []
    for f in tqdm(embedding_files):

        ev = np.loadtxt(f)
        embedding_values.append(ev)

        # Update clustering
        if len(embedding_values) % batch_size == 0:
            strategy.fit_batch(np.array(embedding_values).squeeze())
            embedding_values = []
    
    # Update clustering
    if len(embedding_values) > 0:
        strategy.fit_batch(np.array(embedding_values).squeeze())
        embedding_values = []

    # Save clustering to file
    strategy.save()

    print('Computing clusters fashion class...')

    strategy.compute_cluster_classes(classes_names)

    print('Computing clustering scores...')

    strategy.compute_scores()
