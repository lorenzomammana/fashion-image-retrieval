import os
import shutil

import fashion_utils
import files
from pathlib import Path
from fashion_similarity_online import FashionSimilarity
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array


def plot_output_image(label, clothes):
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

    plt.savefig((files.ROOT / 'similarity-output' / label / ('out_' + clothes)).absolute().as_posix())
    plt.close()


if __name__ == '__main__':

    # TODO get from cmd or something else
    similarity = FashionSimilarity()

    idx_to_class = {
        0: 'Belts',
        1: 'Casual Shoes',
        2: 'Dresses',
        3: 'Flats',
        4: 'Formal Shoes',
        5: 'Handbags',
        6: 'Heels',
        7: 'Jeans',
        8: 'Sandals',
        9: 'Shirts',
        10: 'Shorts',
        11: 'Sports Shoes',
        12: 'Sunglasses',
        13: 'Tops',
        14: 'Trousers',
        15: 'Tshirts'
    }

    for label in os.listdir(files.test_images):
        for i, fname in enumerate(os.listdir(files.test_images / label)):

            if i > 5:
                break

            query_img = files.test_images / label / fname
            n = 10

            img_class, similar_images, c = similarity.get_similar_images(query_img, n)

            print(fname)
            print("Predicted class: " + idx_to_class.get(c))
            print(similar_images)
            plot_output_image(label, fname)
