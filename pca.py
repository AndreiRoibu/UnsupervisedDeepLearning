import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import get_Kaggle_MNIST

def basic_pca():
    X_train, y_train, _, _ = get_Kaggle_MNIST()
    
    pca = PCA()
    reduced_data = pca.fit_transform(X_train)
    plt.scatter(reduced_data[:,0], reduced_data[:,1], s=100, c=y_train, alpha=0.3)
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.show()

    cumulative_variance = []
    last = 0
    for variance in pca.explained_variance_ratio_:
        cumulative_variance.append(last+variance)
        last = cumulative_variance[-1]
    plt.plot(cumulative_variance)
    plt.show()

def implemented_pca():
    X_train, y_train, _, _ = get_Kaggle_MNIST()
    
    # perform data decomposition
    covariance_X = np.cov(X_train.T)
    lambdas, Q = np.linalg.eigh(covariance_X)
    index = np.argsort(-lambdas) 
    lambdas = lambdas[index] # sort in descending order
    lambdas = np.maximum(lambdas, 0)
    Q = Q[:, index] # We make sure that Q is also sorted in the right order

    Z = X_train.dot(Q)
    plt.scatter(Z[:,0], Z[:,1], s=100, c=y_train, alpha=0.3)
    plt.show()

    plt.plot(lambdas)
    plt.title("component variances")
    plt.show()

    plt.plot(np.cumsum(lambdas))
    plt.title("cumulative variance")
    plt.show()

if __name__ == '__main__':
    # basic_pca()
    implemented_pca()