# LANCHEB-GCN
This is the GitHub repository of my 3rd year project: The LANCHEB-GCN: A Hybrid Lanczos-Chebyshev GCN for Point Cloud Segmentation 

The majority of the code and the overall architecture was adapted from Gusi et al.'s [RGCNN: Regularized Graph Convolutional Network for Point Cloud Segmentation](https://arxiv.org/abs/1806.02952), where they implemented Chebyshev polynomials to perform point cloud segmentation. This GCN builds on top of that, in that it finds the eigenvalues via the Lanczos method, which is then used for the spectral filtering in the convolutional layers.

Here is an overview of the organisation of the repository:
- The ```/data/``` directory, where the files involving data handling and processing are held. Both of these files are based on scripts written by Qi et al. for their paper: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413). Furthermore, the ShapeNet-Part dataset should be installed in this directory
- The ```/lancheb_gcn/``` directory, which contains the files of the LANCHEB-GCN
- The ```/plotting/``` directory, which contains the python scripts for visualizing the output point clouds and the validation accuracy of the GCN during training.

## Prerequisites

It is assumed that ```conda``` is already installed in your system. Furthermore, it is recommended that a conda environment has already been created. If not, then write the following code in your shell:
```
conda create --name env
conda activate env
```

It is also recommended to do the following:
```
conda config --remove channels defaults
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda config --set channel_priority flexible
```

Finally, for better performance, it is recommended that CUDA and the necessary drivers have already been installed. 

## Installation
In your shell, copy and paste the following code:
```
conda install tensorflow-gpu pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia pandas scipy
```

These modules may also have to be installed:
```
conda install -c conda-forge matplotlib
```
```
pip install -U scikit-learn
```
Finally, the ShapeNet-Part dataset is the point cloud dataset that the GCN segments. It can downloaded by clicking [here (674 MB)](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Move the uncompressed data folder into the ```data``` directory. 

## Usage
First, for organisational purposes, make two new directories in the ```/data/``` directory named: ```npy-files``` and ```eval-npy```. Once done, go into the ```/data/``` directory and run:
```
python shuffle.py
```
If working, the command should be processing all of the ```.txt``` files from the ShapeNet dataset in NumPy files while simultaneously shuffling them. 

Once complete, move back to the main directory and run the following command:
```
python train_lancheb_gcn.py
```
If everything was installed correctly, then LANCHEB-GCN should be working. 

## (Optional) Display the results
If the test results of the segmentation is required, then go into the ```plotting``` directory and run the following:
```
python plot_segmentation.py
```
This should display (in order) the raw point cloud itself, the 'true' segmented point cloud, and the predicted segmented point cloud from LANCHEB-GCN

If the validation accuracy and loss results during training is needed, then, in the ```plotting``` directory, run the following:
```
python plot_train_perf.py
```
This should output a 2D plot, which shows both the validation accuracy and loss during training.
