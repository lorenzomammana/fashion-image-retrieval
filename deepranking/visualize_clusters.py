import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from clustering_strategy import ClusteringStrategy

if __name__ == '__main__':

    classes_dirs = [d for d in files.small_images_classes_directory.iterdir() if d.is_dir()]
    classes_names = [d.parts[-1] for d in classes_dirs]
    classes_colors = np.random.rand(len(classes_names), 3)

    batch_size = 128
    embedding_values = []
    embedding_values_classes = []

    print('Loading data...')
    for c in classes_names:

        embedding_dir = files.small_images_classes_embeddings / c
        embedding_files = [v for v in embedding_dir.iterdir() if v.is_file()]

        for f in embedding_files:

            embedding = np.loadtxt(f)
            embedding_values.append(embedding)
            embedding_values_classes.append(c)

    embedding_values = np.array(embedding_values)
    print('Dataset computation...')
    reduction = PCA(n_components=2)
    reduction_3d = PCA(n_components=3)
    x_embedded = reduction.fit_transform(embedding_values)
    x_embedded_3d = reduction_3d.fit_transform(embedding_values)

    save_embeddings = []

    for name in classes_names:

        for i in range(x_embedded_3d.shape[0]):
            
            if embedding_values_classes[i] == name: 
                save_embeddings.append(x_embedded_3d[i].tolist() + [name])

    columns = ['x_{}'.format(i) for i in range(3)] + ['class']
    save_embeddings = pd.DataFrame(save_embeddings, columns=columns)
    save_embeddings.to_csv(files.reduced_embeddings_path, index=False)

    # Plot single file clusters
    for name, color in zip(classes_names, classes_colors):

        plot_vals = []

        for i in range(x_embedded.shape[0]):

            if embedding_values_classes[i] == name:
                plot_vals.append([x_embedded[i, 0], x_embedded[i, 1]])

        plt.clf()
        plot_vals = np.array(plot_vals)
        sc = plt.scatter(plot_vals[:, 0], plot_vals[:, 1], s=1, color=color)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.legend([sc], [name], scatterpoints=1, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=1)
        plt.tight_layout()
        plt.savefig(files.clusters_visualization_path / '{}.pdf'.format(name))

    # Plot all clusters in one file
    all_scatter_clusters = []
    plt.clf()
    for name, color in zip(classes_names, classes_colors):

        plot_vals = []

        for i in range(x_embedded.shape[0]):

            if embedding_values_classes[i] == name:
                plot_vals.append([x_embedded[i, 0], x_embedded[i, 1]])

        plot_vals = np.array(plot_vals)
        sc = plt.scatter(plot_vals[:, 0], plot_vals[:, 1], s=1, color=color)
        all_scatter_clusters.append(sc)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend(all_scatter_clusters, classes_names, scatterpoints=1, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.tight_layout()
    plt.savefig(files.clusters_visualization_path / 'all_clusters.pdf')

    strategy = ClusteringStrategy(len(classes_names), batch_size)
    strategy.load()
    kmeans = strategy.kmeans

    print('Centroid computation...')
    x_embedded = reduction.transform(kmeans.cluster_centers_)

    x_1 = x_embedded[:, 0]
    x_2 = x_embedded[:, 1]

    plt.clf()
    scatter_centroids = []
    for i in range(x_embedded.shape[0]):

        sc = plt.scatter(x_1[i], x_2[i], marker='*', s=5)
        scatter_centroids.append(sc)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend(scatter_centroids, np.arange(x_embedded.shape[0]), scatterpoints=1, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.tight_layout()
    plt.savefig(files.clusters_visualization_path / 'centroids.pdf')
