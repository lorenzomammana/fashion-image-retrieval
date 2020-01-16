import pandas as pd
import numpy as np
from fashion_model import FashionModel
from fashion_dataset import FashionDataset
from keras.preprocessing.image import ImageDataGenerator
import files

if __name__ == '__main__':
    
    img_shape = (224, 224, 3)

    # TODO: fix generator in order to handle both original image and label as y col
    dataset = FashionDataset()
    train_datagen = ImageDataGenerator(
        rescale=1./255)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=dataset.data,
        directory=files.small_images_directory,
        x_col=dataset.x_col,
        y_col=dataset.y_col,
        data_format='channels_last',
        batch_size=32,
        class_mode='multi_output')

    model = FashionModel(img_shape, dataset.n_classes)
    model.build()
    model.compile()
