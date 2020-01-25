# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from fashion_ranking_model import FashionRankingModel
import tensorflow as tf
from keras import backend as K
import os
from classification_models import Classifiers
from keras_applications.resnext import preprocess_input
import keras
import sys

sys.path.append("..")
from deepranking.ImageDataGeneratorCustom import ImageDataGeneratorCustom
import autoencoder.files as files

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def preprocessing(x):
    return preprocess_input(x, backend=keras.backend,
                            layers=keras.layers,
                            models=keras.models,
                            utils=keras.utils)


class DataGenerator(object):
    def __init__(self, params, target_size=(224, 224)):
        self.params = params
        self.target_size = target_size
        self.idg = ImageDataGeneratorCustom(**params)

    def get_train_generator(self, batch_size):
        return self.idg.flow_from_directory('/home/ubuntu/fashion-dataset/small_classes/',
                                            batch_size=batch_size,
                                            target_size=self.target_size, shuffle=True,
                                            triplet_path='output/triplets.txt'
                                            )

    def get_test_generator(self, batch_size):
        return self.idg.flow_from_directory(files.small_images_classes_directory,
                                            batch_size=batch_size,
                                            target_size=self.target_size, shuffle=False,
                                            triplet_path='output/triplets.txt'
                                            )


dg = DataGenerator({
    "horizontal_flip": True,
    "preprocessing_function": preprocessing
}, target_size=(224, 224))

batch_size = 8
train_generator = dg.get_train_generator(batch_size)


def _loss_tensor(y_true, y_pred):
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


# deep_rank_model.load_weights('deepranking.h5')
deep_rank_model = FashionRankingModel().compile()

mcp_save_loss = ModelCheckpoint((files.output_directory / 'deepranking_loss.h5').absolute().as_posix(),
                                save_best_only=True,
                                save_weights_only=False,
                                monitor='lambda_4_loss', mode='min')

reduce_lr = ReduceLROnPlateau(monitor='lambda_4_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                              min_delta=0.0001, cooldown=0, min_lr=0)

train_steps_per_epoch = int(train_generator.n / batch_size)
train_epocs = 5
deep_rank_model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              epochs=train_epocs,
                              verbose=1,
                              callbacks=[mcp_save_loss, reduce_lr],
                              use_multiprocessing=False,
                              workers=16,
                              max_queue_size=32
                              )

model_path = files.output_directory / "deepranking.h5"
deep_rank_model.save(model_path.absolute().as_posix())
# f = open('deepranking.json','w')
# f.write(deep_rank_model.to_json())
# f.close()
