import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

X,y = make_blobs(n_samples=100,centers=3,random_state=42)

db1=DBSCAN(eps=0.3,min_samples=5)
db2=DBSCAN(eps=0.8,min_samples=5)
db3=DBSCAN(eps=1.5,min_samples=3)

label1 = db1.fit_predict(X)
label2= db2.fit_predict(X)
label3 = db3.fit_predict(X)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.scatter(X[:,0],X[:,1],c=label1)
plt.title("DBSCAN1")

plt.subplot(1,3,2)
plt.scatter(X[:,0],X[:,1],c=label2)
plt.title("DBSCAN2")

plt.subplot(1,3,3)
plt.scatter(X[:,0],X[:,1],c=label3)
plt.title("DBSCAN3")
plt.savefig("dbscan.png")
