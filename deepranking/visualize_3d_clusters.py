import sys
import pandas as pd
import numpy as np
import vispy.scene
from pathlib import Path
from vispy.scene import visuals

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('ERROR: specify data path')
        exit(1)

    path = Path(sys.argv[1])
    data = pd.read_csv(path)

    classes = list(set(data['class'].values))
    clusters = list(set(data['cluster'].values))
    
    colors_classes = {}
    for c in classes:
        colors_classes[c] = np.random.rand(3)

    color_clusters = {}
    for c in clusters:
        color_clusters[str(c)] = np.random.rand(3)

    points = []
    colors = []
    for i in range(data.shape[0]):
        p = np.array([data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2]])
        #c = color_clusters[str(data.iloc[i, 3])] # Cluster
        c = colors_classes[data.iloc[i, 4]] # Class
        points.append(p)
        colors.append(c)

    points = np.array(points)
    colors = np.array(colors)

    canvas = vispy.scene.SceneCanvas(bgcolor='white', keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    scatter = visuals.Markers()
    scatter.set_data(points, face_color=colors, edge_color=None, size=5)
   
    view.add(scatter)
    view.camera = 'turntable'

    if sys.flags.interactive != 1:
        vispy.app.run()
