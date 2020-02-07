import joblib
import files
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    data_dir = files.small_images_classes_embeddings
    embeddings = pd.read_csv(files.small_images_classes_features)
    clusters_count = embeddings['cluster'].unique().shape[0]

    for i in tqdm(range(clusters_count)):

        cluster_data = []
        cluster_data_ids = []
        cluster_data_class = []

        for j in range(embeddings.shape[0]):

            ev_cluster = embeddings.iloc[j, 1]

            if ev_cluster == i:
                ev_id = embeddings.iloc[j, 0]
                ev_class = embeddings.iloc[j, 2]
                ev_path = files.small_images_classes_embeddings / ev_class / '{}.txt'.format(ev_id)
                
                cluster_data.append(np.loadtxt(ev_path))
                cluster_data_ids.append(ev_id)
                cluster_data_class.append(ev_class)
        
        cluster_data = np.array(cluster_data)
        cluster_data_ids = np.array(cluster_data_ids)
        cluster_data_class = np.array(cluster_data_class)

        joblib.dump(cluster_data, data_dir / '{}.joblib'.format(i))
        joblib.dump(cluster_data_ids, data_dir / '{}_ids.joblib'.format(i))
        joblib.dump(cluster_data_class, data_dir / '{}_class.joblib'.format(i))

        cluster_data = None
        cluster_data_ids = None
        cluster_data_class = None
