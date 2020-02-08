from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.storage import RedisStorage
import deepranking.files as files
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from deepranking.fashion_utils import triplet_loss_adapted_from_tf
from keras_applications.resnext import preprocess_input
import keras
from redis import Redis

redis_object = Redis(host='localhost', port=6379, db=0)
redis_storage = RedisStorage(redis_object)

config = redis_storage.load_hash_configuration('MyHash')

lshash = RandomBinaryProjections(None, None)
lshash.apply_config(config)
engine = Engine(4096, lshashes=[lshash], storage=redis_storage)

redis_object.close()


class FashionSimilarity:
    def __init__(self):
        self.model = load_model((files.output_directory / 'onlinemining_loss.h5').absolute().as_posix(),
                                custom_objects={'triplet_loss_adapted_from_tf': triplet_loss_adapted_from_tf})

    def get_similar_images(self, img_path, n):
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        img = np.expand_dims(img, axis=0)

        ev, c = self.model.predict([img, img, img, np.zeros(1)])
        label = np.argmax(c)
        perc = np.max(c)
        ev = ev[:, 1:]

        N = engine.neighbours(ev.squeeze())

        similarity = []

        for r in N:
            final = r[1].absolute().as_posix().split('/')
            similarity.append([final[-1].replace('.txt', ''), final[-2], r[2]])

        similarity = pd.DataFrame(similarity, columns=['id', 'class', 'score'])

        return similarity, label, perc
