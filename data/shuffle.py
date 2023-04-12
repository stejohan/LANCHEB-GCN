import argparse
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
# import tf_util
import normalise_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
FLAGS = parser.parse_args()
NUM_POINT = FLAGS.num_point

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')

TRAIN_DATASET = normalise_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='train')
VAL_DATASET = normalise_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='val')
TEST_DATASET = normalise_dataset.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test')

train_length = len(TRAIN_DATASET)
val_length = len(VAL_DATASET)
test_length = len(TEST_DATASET)

data_train = np.zeros((train_length, NUM_POINT, 6))
label_train = np.zeros((train_length, NUM_POINT), dtype=np.int32)
idx_train = np.arange(0, len(TRAIN_DATASET))
np.random.shuffle(idx_train)

data_val = np.zeros((val_length, NUM_POINT, 6))
label_val = np.zeros((val_length, NUM_POINT), dtype=np.int32)
idx_val = np.arange(0, len(VAL_DATASET))
np.random.shuffle(idx_val)

data_test = np.zeros((test_length, NUM_POINT, 6))
label_test = np.zeros((test_length, NUM_POINT), dtype=np.int32)
idx_test = np.arange(0, len(TEST_DATASET))
np.random.shuffle(idx_test)

for i in range(train_length):
    ps_train, normal_train, seg_train = TRAIN_DATASET[idx_train[i]]
    data_train[i,:,0:3] = ps_train
    data_train[i,:,3:6] = normal_train
    label_train[i,:] = seg_train

for j in range(val_length):
    ps_val, normal_val, seg_val = VAL_DATASET[idx_val[j]]
    data_val[j,:,0:3] = ps_val
    data_val[j,:,3:6] = normal_val
    label_val[j,:] = seg_val

for k in range(test_length):
    ps_test, normal_test, seg_test = TEST_DATASET[idx_test[k]]
    data_test[k,:,0:3] = ps_test
    data_test[k,:,3:6] = normal_test
    label_test[k,:] = seg_test

with open("npy-files/data_train.npy", "wb") as f:
    np.save(f, data_train)
with open("npy-files/data_val.npy", "wb") as f:
    np.save(f,data_val)
with open("npy-files/data_test.npy", "wb") as f:
    np.save(f,data_test)
with open("npy-files/label_train.npy", "wb") as f:
    np.save(f,label_train)
with open("npy-files/label_val.npy", "wb") as f:
    np.save(f,label_val)
with open("npy-files/label_test.npy", "wb") as f:
    np.save(f,label_test)

print(train_length, val_length, test_length)
