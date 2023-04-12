# LANCHEB-GCN
This is the GitHub repository of my 3rd year project: The LANCHEB-GCN: A Hybrid Lanczos-Chebyshev GCN for Point Cloud Segmentation 

The majority of the code and the overall architecture was adapted from Gusi et al.'s [RGCNN: Regularized Graph Convolutional Network for Point Cloud Segmentation](https://arxiv.org/abs/1806.02952), where they implemented Chebyshev polynomials to perform point cloud segmentation. This GCN builds on top of that, in that it finds the eigenvalues via the Lanczos method, which is then used for the spectral filtering in the convolutional layers.

Here is an overview of the organisation of the repository:
- The ```/data/``` directory, where the files involving data handling and processing are held. Both of these files are based on scripts written by Qi et al. for their paper: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413). Furthermore, the ShapeNet-Part dataset should be installed in this directory
- The ```/lancheb_gcn/``` directory, which contains the files of the LANCHEB-GCN
- The ```/plotting/``` directory, which contains the python scripts for visualizing the output point clouds and the validation accuracy of the GCN during training.

## 
