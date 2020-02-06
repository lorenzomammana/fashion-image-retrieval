import time

from nearpy.storage import RedisStorage

import files
import pandas as pd
import numpy as np
from tqdm import tqdm
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance
from redis import Redis
from fashion_utils import DB_HOST


redis_object = Redis(host=DB_HOST, port=6379, db=0)
redis_storage = RedisStorage(redis_object)

if __name__ == '__main__':
    batch_size = 128
    data = pd.read_csv(files.small_images_classes_features)

    config = redis_storage.load_hash_configuration('MyHash')

    dimension = 4096

    if config is None:
        # Config is not existing, create hash from scratch, with 10 projections
        lshash = RandomBinaryProjections('MyHash', 10)
    else:
        # Config is existing, create hash with None parameters
        lshash = RandomBinaryProjections(None, None)
        # Apply configuration loaded from redis
        lshash.apply_config(config)

    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[lshash], storage=redis_storage, distance=EuclideanDistance)

    embedding_values = []

    for i in tqdm(range(data.shape[0])):
        embedding_dir = files.small_images_classes_embeddings / data.iloc[i, 2]

        embedding = np.loadtxt(embedding_dir / '{}.txt'.format(data.iloc[i, 0]))
        engine.store_vector(embedding, embedding_dir / '{}.txt'.format(data.iloc[i, 0]))

    redis_storage.store_hash_configuration(lshash)
