import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Plotting function
def plot_data(X, Y, title):
    plt.figure(figsize=(8, 6))
    for i in range(10):
        # print(X[Y[:,0] == i])
        subset = X[Y == i]
        plt.scatter(subset[:, 0], subset[:, 1], label=str(i))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")

# Loading data
with open('mnist_small.pkl', 'rb') as file:
    data = pickle.load(file)
X = data['X']
Y = data['Y']
Y = Y.reshape((Y.shape[0]))
print(X.shape)
print(Y.shape)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca.shape)
print("PCA done")

# t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)
print("t-SNE done")

plot_data(X_pca, Y, 'PCA of MNIST Data')
plot_data(X_tsne, Y, 't-SNE of MNIST Data')