import tensorflow as tf
from keras import backend as K
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def loss_tensor(y_true, y_pred, batch_size=8):
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
