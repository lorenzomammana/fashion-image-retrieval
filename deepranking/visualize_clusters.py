import files
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joblib

if __name__ == '__main__':
    
    METH0D = 'PCA' # 'TSNE' 'PCA'
    n_components = 2

    classes_dirs = [d for d in files.small_images_classes_directory.iterdir() if d.is_dir()]
    classes_names = [d.parts[-1] for d in classes_dirs]
    classes_colors = np.random.rand(len(classes_names), 3)

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
    print('{} dataset computation...'.format(METH0D))
    if METH0D == 'TSNE':
        x_embedded = TSNE(n_components=n_components).fit_transform(embedding_values)
    
    if METH0D == 'PCA':
        x_embedded = PCA(n_components=n_components).fit_transform(embedding_values)

    # Plot single file clusters
    for name, color in zip(classes_names, classes_colors):

        plot_vals = []

        for i in range(x_embedded.shape[0]):

            if embedding_values_classes[i] == name:
                plot_vals.append([x_embedded[i, 0], x_embedded[i, 1]])

        plt.clf()
        plot_vals = np.array(plot_vals)
        sc = plt.scatter(plot_vals[:, 0], plot_vals[:, 1], s=1, color=color)
        plt.legend([sc], [name], scatterpoints=1)
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
    
    plt.legend(all_scatter_clusters, classes_names, scatterpoints=1)
    plt.tight_layout()
    plt.savefig(files.clusters_visualization_path / 'all_clusters.pdf')

    kmeans = joblib.load(files.small_images_classes_kmeans)
    embedding_values = kmeans.cluster_centers_
    print('{} centroid computation...'.format(METH0D))
    if METH0D == 'TSNE':
        x_embedded = TSNE(n_components=n_components).fit_transform(embedding_values)
    
    if METH0D == 'PCA':
        x_embedded = PCA(n_components=n_components).fit_transform(embedding_values)

    x_1 = x_embedded[:, 0]
    x_2 = x_embedded[:, 1]

    plt.clf()
    scatter_centroids = []
    for i in range(x_embedded.shape[0]):

        sc = plt.scatter(x_1[i], x_2[i], marker='*', s=5)
        scatter_centroids.append(sc)

    plt.legend(scatter_centroids, np.arange(x_embedded.shape[0]), scatterpoints=1)
    plt.tight_layout()
    plt.savefig(files.clusters_visualization_path / 'centroids.pdf')
