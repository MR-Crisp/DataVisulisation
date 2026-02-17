import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state = 23)

# fig =  plt.figure(0)
# plt.grid(True)
# plt.scatter(X[:,0], X[:,1], c=y, s=50, edgecolor='k', cmap='viridis')
# plt.title('Generated Data')
# plt.show()

k=3

clusters = {}
np.random.seed(23)
for i in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1)
    points = []
    cluster = {
        'center': center,
        'points': []
    }
    clusters[i] = cluster
print(clusters)

plt.scatter(X[:,0], X[:,1], c=y, s=50, edgecolor='k', cmap='viridis')
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = 'x',c = 'red')
plt.show()

def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
            
            clusters[i]['points'] = []
    return clusters