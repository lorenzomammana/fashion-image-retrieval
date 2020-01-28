import os
import shutil

import fashion_utils
import files
from pathlib import Path
from fashion_similarity_online import FashionSimilarity
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array


def plot_output_image(clothes):
    plt.clf()
    fig = plt.figure(figsize=(5, 2))
    columns = 5
    rows = 2

    for i in range(1, columns * rows + 1):
        filename = similar_images['class'].iloc[i - 1] + '/' + str(similar_images['id'].iloc[i - 1]) + '.jpg'
        img = load_img((files.small_images_classes_directory / filename).absolute().as_posix())
        img = img_to_array(img) / 255
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)

    plt.savefig((files.ROOT / 'similarity-output' / ('out_' + clothes)).absolute().as_posix())


if __name__ == '__main__':

    # TODO get from cmd or something else
    similarity = FashionSimilarity()

    for fname in os.listdir(files.test_images):
        query_img = files.test_images / fname
        n = 10

        img_class, similar_images = similarity.get_similar_images(query_img, n)

        print(similar_images)
        plot_output_image(fname)
