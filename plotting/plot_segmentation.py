import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import proj3d
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH_NPY = os.path.join(ROOT_DIR, 'data', 'npy-files')
DATA_PATH_EVAL = os.path.join(ROOT_DIR, 'data', 'eval-npy')

data_test = np.load(DATA_PATH_NPY + "/data_test.npy")
label_test = np.load(DATA_PATH_NPY + "/label_test.npy")
predictions_LANCHEB_GCN = np.load(DATA_PATH_EVAL + "/LANCHEBGCN_predictions.npy")

def visualize_data(points,labels):
    df = pd.DataFrame(
        data={
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "label": labels - min(labels),
        }
    )

    colors = ['#fecd50', '#b482c2', '#339900', '#cc3300', '00ffef', '494f5f']
    fig = plt.figure(figsize=(22,22))
    ax = plt.axes(projection="3d")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.scatter(df.x, df.y, df.z, color=np.array(colors)[df.label])
    ax.view_init(45,0,90)
    plt.axis('off')
    plt.show()

def visualize_data_no_color(points):
    df = pd.DataFrame(
        data={
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
        }
    )
    fig = plt.figure(figsize=(22,22))
    ax = plt.axes(projection="3d")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.scatter(df.x, df.y, df.z)
    ax.view_init(0,0,0)
    plt.axis('off')
    plt.show()

i = 0   # Change i to your desire

visualize_data_no_color(data_test[i])                   # Show point cloud with no labels
visualize_data(data_test[i],label_test[i])              # Show point cloud with true labels
visualize_data(data_test[i],                            # Show point cloud with predicted labels
               predictions_LANCHEB_GCN.astype(int)[i])
