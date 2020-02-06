import files
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import joblib
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from fashion_utils import loss_tensor, triplet_loss_adapted_from_tf, DB_HOST, DB_PORT
from keras_applications.resnext import preprocess_input
import keras
from aquiladb import AquilaClient
import json

class FashionSimilarity:

    def __init__(self):
        self.model = load_model((files.output_directory / 'onlinemining_loss.h5').absolute().as_posix(),
                                custom_objects={'triplet_loss_adapted_from_tf': triplet_loss_adapted_from_tf})

        self.db = AquilaClient(DB_HOST, DB_PORT)

    def get_similar_images(self, img_path, n):
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        img = np.expand_dims(img, axis=0)

        ev, c = self.model.predict([img, img, img, np.zeros(1)])
        label = np.argmax(c)
        perc = np.max(c)
        ev = ev[:, 1:]

        ev_as_list = np.reshape(ev, (-1,)).tolist()
        search_ev = self.db.convertMatrix(ev_as_list)
        result = self.db.getNearest(search_ev, n)
        result = json.loads(result.documents)

        similarity = []

        for r in result:
            similarity.append([r['doc']['image'], r['doc']['label']])

        similarity = pd.DataFrame(similarity, columns=['id', 'class'])

        return similarity, label, perc
