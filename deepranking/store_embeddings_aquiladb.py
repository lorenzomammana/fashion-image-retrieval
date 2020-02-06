import time
import files
import pandas as pd
import numpy as np
from tqdm import tqdm
from aquiladb import AquilaClient
from fashion_utils import DB_HOST, DB_PORT

if __name__ == '__main__':
    
    db = AquilaClient(DB_HOST, DB_PORT)
    batch_size = 128
    data = pd.read_csv(files.small_images_classes_features)

    embedding_values = []

    for i in tqdm(range(data.shape[0])):

        embedding_dir = files.small_images_classes_embeddings / data.iloc[i, 2]

        embedding = np.loadtxt(embedding_dir / '{}.txt'.format(data.iloc[i, 0]))
        embedding = db.convertDocument(embedding.tolist(), {'image': int(data.iloc[i, 0]), 'cluster': int(data.iloc[i, 1]), 'label': str(data.iloc[i, 2])})
        embedding_values.append(embedding)

        if len(embedding_values) % batch_size == 0:
            db.addDocuments(embedding_values)
            embedding_values = []
            time.sleep(1)

    if len(embedding_values) > 0:
        db.addDocuments(embedding_values)
        embedding_values = []
        time.sleep(1)
