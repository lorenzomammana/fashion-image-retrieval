from __future__ import absolute_import
from __future__ import print_function
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
import tensorflow as tf

class FashionRankingModel():

    def __init__(self):

        self.convnet = None
        self.deeprank = None
        self.optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
        self.loss = None
        self.batch_size = 8
        self.batch_size *= 3

        self.__build_convnet__()
        self.__build_deeprank__()
        self.__create_loss_function__()

    def compile(self, weights=None):

        if weights is not None:
            self.deeprank.load_weights(weights)

        self.deeprank.compile(loss=self.loss, optimizer=self.optimizer)

        return self.deeprank

    def __build_convnet__(self):

        vgg_model = VGG16(weights='imagenet', include_top=False)
        x = vgg_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.6)(x)
        x = Lambda(lambda x_: K.l2_normalize(x, axis=1))(x)
        convnet_model = Model(inputs=vgg_model.input, outputs=x)
        
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
        l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

        final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

        self.deeprank = final_model

    def __create_loss_function__(self):

        _EPSILON = K.epsilon()
        batch_size = self.batch_size

        def loss_tensor(y_true, y_pred):

            y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
            loss = tf.convert_to_tensor(0, dtype=tf.float32)
            g = tf.constant(1.0, shape=[1], dtype=tf.float32)
            
            for i in range(0, batch_size, 3):
                try:
                    q_embedding = y_pred[i + 0]
                    p_embedding = y_pred[i + 1]
                    n_embedding = y_pred[i + 2]
                    D_q_p = K.sqrt(K.sum((q_embedding - p_embedding) ** 2))
                    D_q_n = K.sqrt(K.sum((q_embedding - n_embedding) ** 2))
                    loss = (loss + g + D_q_p - D_q_n)
                except:
                    continue
            
            loss = loss / (batch_size / 3)
            zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
            
            return tf.maximum(loss, zero)

        self.loss = loss_tensor
