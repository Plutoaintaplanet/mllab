import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
X=data.data
y=data.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0],X_pca[:,1],c=y,cmap='viridis')
plt.xlabel("First PC")
plt.ylabel("Second PC")
plt.title("PCA of Iris Datasets")
plt.colorbar(scatter, label='Classes')
plt.savefig("pca.png")

print("Explained variance ratio: ",pca.explained_variance_ratio_)
