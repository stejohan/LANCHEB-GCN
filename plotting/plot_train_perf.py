import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH_EVAL = os.path.join(ROOT_DIR, 'data', 'eval-npy')
LANCHEBGCN_FILE = os.path.join(DATA_PATH_EVAL, "LANCHEBGCN_training_perf.npy")

font = {'size'   : 22}
matplotlib.rc('font', **font)

lanchebgcn_perf = np.load(LANCHEBGCN_FILE)
def visualize_data(data1):
    ef = pd.DataFrame(
        data={
            "epochs": data1[:,0],
            "accuracy": data1[:,1],
            "loss": data1[:,2],
        }
    )

    fig,ax1 = plt.subplots()
    ax1.set_xlabel("Nr. of epochs")
    ax1.set_ylabel("Validation Accuracy")
    line1, = ax1.plot(ef.epochs, ef.accuracy, color = "blue", label = "Val. Acc. (LANCHEB-GCN)")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss")
    line2, = ax2.plot(ef.epochs, ef.loss, color = "red", label = "Loss (LANCHEB-GCN)")
    plt.legend(handles=[line1,line2],loc = "upper left")
    plt.show()

visualize_data(lanchengcn_perf)
