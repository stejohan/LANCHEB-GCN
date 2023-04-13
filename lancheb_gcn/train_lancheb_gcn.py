import seg_model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time,json
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# Pathing to get to npy files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH_NPY = os.path.join(ROOT_DIR, 'data', 'npy-files')
DATA_PATH_EVAL = os.path.join(ROOT_DIR, 'data', 'eval-npy')

def genData(cls,limit=None):
    assert type(cls) is str

    # Semantic labels for all 16 objects in the dataset
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                   'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15],
                   'Knife': [22, 23]}

    # Input the data (train, val, test, depending on input 'cls')
    data = np.load(DATA_PATH_NPY + "/data_%s.npy" % cls)
    label = np.load(DATA_PATH_NPY + "/label_%s.npy" % cls)

    data = data[:limit]
    label = label[:limit]

    seg = {}
    name = {}
    i = 0
    for k,v in sorted(seg_classes.items()):
        for value in v:
            seg[value] = i
            name[value] = k
        i += 1
    cnt = data.shape[0]
    cat = np.zeros((cnt))
    for i in range(cnt):
        cat[i] = seg[label[i][0]]
    return data,label,cat

def train():
    train_data, train_label, train_cat = genData('train')   # Get training data
    val_data, val_label, val_cat = genData('val')           # Get validation data
    test_data, test_label, test_cat = genData('test')       # Get test data

    params = dict()
    params['dir_name'] = 'model'
    params['num_epochs'] = 3            # Set nr. of epochs for training, default = 3
    params['batch_size'] = 26           # Set batch size, defaul = 26
    params['eval_frequency'] = 30       # Validate every 30 shapes

    # Building blocks.
    params['filter'] = 'lancheb'
    params['brelu'] = 'b1relu'
    params['pool'] = 'apool1'

    # Architecture.
    params['F'] = [128, 512, 1024, 512, 128, 50]  # Number of graph convolutional filters.
    params['K'] = [6, 5, 3, 1, 1, 1]  # Polynomial orders.
    params['M'] = [384, 16, 1]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 1e-9
    params['dropout'] = 1
    params['learning_rate'] = 1e-3
    params['decay_rate'] = 0.95
    params['momentum'] = 0
    params['decay_steps'] = train_data.shape[0] / params['batch_size']

    # Import LANCHEB-GCN for point cloud segmentation training
    model = seg_model.lancheb_gcn(2048, **params)
    accuracy, loss, t_step, epochs = model.fit(train_data, train_cat, train_label, val_data, val_cat, val_label,
                                       is_continue=False)

    # Output validation accuracy and loss per evaluation to npy file
    training_perf = np.transpose(np.array([epochs,accuracy,loss]))
    with open(DATA_PATH_EVAL + "/LANCHEBGCN_training_perf.npy", "wb") as f:
        np.save(f, training_perf)

def test():
    # Import test data
    test_data, test_label, test_cat = genData('test')

    params = dict()
    params['dir_name'] = 'model'
    params['num_epochs'] = 3        # Set nr. of epochs for testing, default = 3
    params['batch_size'] = 26       # Set batch size, default = 26
    params['eval_frequency'] = 30   # Validate every 30 shapes

    # Building blocks.
    params['filter'] = 'lancheb'
    params['brelu'] = 'b1relu'
    params['pool'] = 'apool1'

    # Architecture.
    params['F'] = [128, 512, 1024, 512, 128, 50]  # Number of graph convolutional filters.
    params['K'] = [6, 5, 3, 1, 1, 1]  # Polynomial orders.
    params['M'] = [384, 16, 1]  # Output dimensionality of fully connected layers. For classification only

    # Optimization.
    params['regularization'] = 1e-9
    params['dropout'] = 1
    params['learning_rate'] = 1e-3
    params['decay_rate'] = 0.95
    params['momentum'] = 0
    params['decay_steps'] = test_data.shape[0] / params['batch_size']

    # Import LANCHEB-GCN for point cloud segmentation testing
    model = seg_model.rgcnn(2048, **params)
    string, accuracy, f1, loss, predictions = model.evaluate(test_data,test_cat,test_label)

    # Save the prediciton labels for visualizations
    with open(DATA_PATH_EVAL + "/LANCHEBGCN_predictions.npy", "wb") as f:
        np.save(f, predictions)

if __name__=="__main__":
    train()
    test()
