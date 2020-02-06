from __future__ import absolute_import
from __future__ import print_function

from classification_models import Classifiers
from keras.layers import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf


class FashionRankingModel:

    def __init__(self):

        self.convnet = None
        self.deeprank = None
        self.optimizer = 'adam'
        self.loss = None
        self.batch_size = 8

        self.__build_convnet__()
        self.__build_deeprank__()
        self.__create_loss_function__()

    def compile(self, weights=None):

        if weights is not None:
            self.deeprank.load_weights(weights, by_name=True)

        self.deeprank.compile(loss=self.loss, optimizer=self.optimizer, metrics=['acc'])

        return self.deeprank

    def __build_convnet__(self):
        resnet, _ = Classifiers.get('resnet18')
        base_model = resnet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Lambda(lambda x_: K.l2_normalize(x_, axis=1))(x)
        convnet_model = Model(inputs=base_model.input, outputs=x)

        self.convnet = convnet_model

    def __build_deeprank__(self):
        convnet_model = self.convnet
        first_input = Input(shape=(224, 224, 3))
        first_conv = Conv2D(96, kernel_size=(8, 8), strides=(16, 16), padding='same')(first_input)
        first_max = MaxPool2D(pool_size=(3, 3), strides=(4, 4), padding='same')(first_conv)
        first_max = Flatten()(first_max)
        first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

        second_input = Input(shape=(224, 224, 3))
        second_conv = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(second_input)
        second_max = MaxPool2D(pool_size=(7, 7), strides=(2, 2), padding='same')(second_conv)
        second_max = Flatten()(second_max)
        second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)

        merge_one = concatenate([first_max, second_max])

        merge_two = concatenate([merge_one, convnet_model.output])
        emb = Dense(4096)(merge_two)
        dense_classification = Dense(256, activation='relu')(convnet_model.output)
        drop_classification = Dropout(0.6)(dense_classification)
        classification_layer = Dense(16, activation='softmax')(drop_classification)
        l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

        final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=[l2_norm_final,
                                                                                              classification_layer])

        self.deeprank = final_model

    def __create_loss_function__(self):
        batch_size = self.batch_size

        def loss_tensor(y_true, y_pred):
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

        self.loss = [loss_tensor, 'categorical_crossentropy']
