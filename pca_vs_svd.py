import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from utils import get_Kaggle_MNIST

def pca_vs_svd():
    pca = PCA()
    svd = TruncatedSVD()

    X, y, _, _ = get_Kaggle_MNIST()
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    np.place(std, std == 0, 1)
    X = (X - mean) / std

    Z_pca = pca.fit_transform(X)
    Z_svd = pca.fit_transform(X)

    plt.subplot(1,2,1)
    plt.scatter(Z_pca[:,0], Z_pca[:,1], c=y)
    plt.subplot(1,2,2)
    plt.scatter(Z_svd[:,0], Z_svd[:,1], c=y)
    plt.show()

if __name__ == '__main__':
    pca_vs_svd()