import files
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras_preprocessing.image import load_img, img_to_array

def plot_output_image(filenames, classes):
    gs = gridspec.GridSpec(2, 8)
    fig = plt.figure(figsize=(16, 6))
    plt.axis('off')

    for i in range(2):
        for j in range(8):
            filename = filenames[8 * i + j]
            img = load_img(filename)
            img = img_to_array(img) / 255
            ax = plt.subplot(gs[i, j])
            ax.axis('off')
            ax.imshow(img)
            ax.text(0.5, -0.1, classes[8 * i + j], size=12, ha='center', transform=ax.transAxes)

    plt.savefig(files.small_images_classes_directory / 'one_image_per_class.pdf')
    plt.clf()
    plt.close(fig)

if __name__ == '__main__':
    
    data = pd.read_csv(files.small_images_classes_features)
    classes = data['class'].unique()
    filenames = []

    for c in classes:

        class_data = data[data['class'] == c].iloc[0]
        path = files.small_images_classes_directory / c / '{}.jpg'.format(class_data['id'])
        filenames.append(path)

    plot_output_image(filenames, classes)
