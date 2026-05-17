from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X,y= make_blobs(n_samples=100,centers=3,random_state=0)

wcss=[]
for k in range(1,7):
  model = KMeans(n_clusters=k, random_state=0, n_init=10)
  model.fit(X)
  wcss.append(model.inertia_)

plt.plot(range(1,7),wcss,marker='o')
plt.title("Elbow Method")
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.savefig("elbow.png")

for k in [2,3,4]:
  model = KMeans(n_clusters=k,random_state=0,n_init=10)
  labels = model.fit_predict(X)

  print("\n K= ",k)
  print("WCSS: ",model.inertia_)

  plt.scatter(X[:,0],X[:,1],c=labels)
  plt.scatter(
              model.cluster_centers_[:,0],
              model.cluster_centers_[:,1],
              marker='X',
              color='red',
              s=200)

  plt.title(f"KMeans clustering:{k} ")
  plt.savefig(f"kmeans_{k}.png")
