import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from scipy.cluster.hierarchy import dendrogram, linkage

X,y = make_blobs(n_samples=20,centers=3,random_state=42)

agnes = AgglomerativeClustering(n_clusters=3)
agnes_labels = agnes.fit_predict(X)

diana = BisectingKMeans(n_clusters=3)
diana_labels = diana.fit_predict(X)

#Dendrogram

plt.figure(figsize=(8,5))
linkage_matrix = linkage(X,method='ward')

dendrogram(linkage_matrix)

plt.title("Dendrogram")
plt.xlabel("Data points")
plt.ylabel("Distances")
plt.savefig("dendrogram.png")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=agnes_labels)
plt.title("Agnes Graph")

plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=diana_labels)
plt.title("Diana Graph")
plt.savefig("agnes_diana.png")
