import keras
import joblib
import files
import numpy as np
import pandas as pd
from clustering_strategy import ClusteringStrategy
from keras.preprocessing.image import load_img, img_to_array
from keras_applications.resnext import preprocess_input
from tqdm import tqdm
from keras.models import load_model
from fashion_utils import triplet_loss_adapted_from_tf

if __name__ == '__main__':
    model = load_model((files.output_directory / 'onlinemining.h5').absolute().as_posix(),
                       custom_objects={'triplet_loss_adapted_from_tf': triplet_loss_adapted_from_tf})

    classes_dirs = [d for d in files.small_images_classes_directory.iterdir() if d.is_dir()]
    classes_names = [d.parts[-1] for d in classes_dirs]

    images = []
    for p, c in zip(classes_dirs, classes_names):
        images = images + [i.as_posix().split('/')[-2:] for i in p.iterdir() if i.is_file()]
    n_classes = len(classes_names)
    embedding_values = []

    batch_size = 128
    strategy = ClusteringStrategy(n_classes)

    # For each class
    images = np.random.permutation(images)

    print('Computing clustering...')

    for label, i in tqdm(images):

        # Compute embedding value and save to txt file
        img = load_img(files.small_images_classes_directory / label / i, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        img = np.expand_dims(img, axis=0)

        # Passo una label 0, non serve a nulla nella forward
        ev = model.predict([img, img, img, np.zeros(1)])[0]

        # La ignoro nell'output
        ev = ev[:, 1:]
        embedding_values.append(ev)
        np.savetxt(files.small_images_classes_embeddings / label / i.replace('jpg', 'txt'), ev)

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
