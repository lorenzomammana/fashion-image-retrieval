import os
import shutil

import fashion_utils
import files
from pathlib import Path
from fashion_similarity_online_aquiladb import FashionSimilarity
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import gridspec

def plot_output_image(query, pred, perc, label, clothes):
    gs00 = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs00[0, 1])
    fig = plt.figure(figsize=(16, 6))

    fig.suptitle('Classification: ' + pred + ' with confidence ' + str(np.round(perc, 3)) +
                 '    Clustering: ' + "/".join(np.array(similar_images['class'].mode())),
                 fontsize=16)
    left_ax = fig.add_subplot(gs00[0, 0])
    plt.axis('off')
    input = load_img(query)
    input = img_to_array(input) / 255
    plt.imshow(input)
    for i in range(2):
        for j in range(5):
            filename = similar_images['class'].iloc[5 * i + j] + '/' + \
                       str(similar_images['id'].iloc[5 * i + j]) + '.jpg'
            img = load_img((files.small_images_classes_directory / filename).absolute().as_posix())
            img = img_to_array(img) / 255
            ax = plt.subplot(gs01[i, j])
            ax.axis('off')
            ax.imshow(img)

    plt.savefig((files.ROOT / 'similarity-output' / label / ('out_' + clothes)).absolute().as_posix())
    plt.clf()
    plt.close(fig)


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

            similar_images, pred, perc = similarity.get_similar_images(query_img, n)

            # print(fname)
            # print("Predicted class: {}".format(idx_to_class.get(pred)))
            # print(similar_images)
            plot_output_image(query_img, idx_to_class.get(pred), perc, label, fname)
