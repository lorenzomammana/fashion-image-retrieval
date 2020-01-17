import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fashion_model import FashionModel
from fashion_dataset import FashionDataset
from fashion_sequence import FashionSequence
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

if __name__ == '__main__':
    
    img_shape = (224, 168, 3)
    batch_size = 128
    n_workers = 8
    epochs = 50
    shuffle = True

    dataset = FashionDataset(files.small_images_directory)
    
    # TODO: fix 1-sample class
    # stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    # train_idx, validation_idx = next(stratified_split.split(dataset.data[dataset.x].values, dataset.data[dataset.label].values))
    
    # train_sequence = FashionSequence(dataset.data.iloc[train_idx], img_shape, files.small_images_directory, dataset.n_classes, batch_size=batch_size, shuffle=shuffle)
    # validation_sequence = FashionSequence(dataset.data.iloc[validation_idx], img_shape, files.small_images_directory, dataset.n_classes, batch_size=batch_size, shuffle=shuffle)

    train_sequence = FashionSequence(dataset.data, img_shape, files.small_images_directory, dataset.n_classes, batch_size=batch_size, shuffle=shuffle)

    model = FashionModel(img_shape, dataset.n_classes)
    model.build()
    model.compile()

    result = model.autoencoder.fit_generator(
        train_sequence,
        # validation_data=validation_sequence,
        epochs=50, workers=n_workers, use_multiprocessing=True, shuffle=False)

    plt.clf()
    plt.plot(result.history['output_decoder_loss'], label='autoencoder')
    plt.plot(result.history['output_classifier_loss'], label='classifier')
    plt.title('Training loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(files.output_directory / 'train_loss.pdf')

    plt.clf()
    plt.plot(result.history['output_decoder_acc'], label='autoencoder')
    plt.plot(result.history['output_classifier_acc'], label='classifier')
    plt.title('Training accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(files.output_directory / 'train_accuracy.pdf')
