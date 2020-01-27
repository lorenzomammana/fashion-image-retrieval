from __future__ import absolute_import
from __future__ import print_function

from classification_models import Classifiers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
from keras.models import Model
from keras import backend as K

import files
from OnlineDataGenerator import ImageDataGenerator
from fashion_ranking_model import FashionRankingModel
from keras_applications.resnext import preprocess_input
from fashion_utils import triplet_loss_adapted_from_tf
import tensorflow as tf
import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def preprocessing(x):
    return preprocess_input(x, backend=keras.backend,
                            layers=keras.layers,
                            models=keras.models,
                            utils=keras.utils)


if __name__ == '__main__':
    base_network = FashionRankingModel()
    input_image1 = Input(shape=(224, 224, 3), name='input_image1')
    input_image2 = Input(shape=(224, 224, 3), name='input_image2')
    input_image3 = Input(shape=(224, 224, 3), name='input_image3')
    input_labels = Input(shape=(1,), name='input_label')

    embeddings, classification = base_network.deeprank(
        [input_image1, input_image2, input_image3])  # output of network -> embeddings
    labels_plus_embeddings = concatenate([input_labels, embeddings])

    model = Model(inputs=[input_image1, input_image2, input_image3, input_labels],
                  outputs=[labels_plus_embeddings, classification])

    model.compile(loss=[triplet_loss_adapted_from_tf, 'categorical_crossentropy'],
                  optimizer='adam',
                  metrics=['acc'])

    batch_size = 128
    dg = ImageDataGenerator(
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        zoom_range=[0.5, 1.5],
        preprocessing_function=preprocessing
    )

    train_generator = dg.flow_from_directory(files.small_images_classes_directory,
                                             batch_size=batch_size,
                                             target_size=(224, 224),
                                             shuffle=True,
                                             class_mode='categorical'
                                             )

    mcp_save_loss = ModelCheckpoint((files.output_directory / 'deepranking_loss.h5').absolute().as_posix(),
                                    save_best_only=True,
                                    save_weights_only=False,
                                    monitor='concatenate_3_loss', mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='concatenate_3_loss', factor=0.1, patience=5, verbose=1, mode='min',
                                  min_delta=0.001, cooldown=0, min_lr=0)

    train_steps_per_epoch = int(train_generator.n / batch_size)
    train_epocs = 20
    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=train_epocs,
                        verbose=1,
                        callbacks=[mcp_save_loss, reduce_lr],
                        use_multiprocessing=False,
                        workers=16,
                        max_queue_size=32
                        )

    model_path = files.output_directory / "onlinemining.h5"
    model.save(model_path.absolute().as_posix())
