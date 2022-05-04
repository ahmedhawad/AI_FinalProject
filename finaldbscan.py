import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN as  DBSCAN2
from sklearn.datasets import make_moons
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import completeness_score

import math

def dist(X,Y):
	#euclidean distance
	return np.sqrt(sum([(x-y)*(x-y) for x,y in zip(X,Y)]))
class DBSCAN():
	def __init__(self, eps=1, minSamples=10):
		self.eps = eps
		self.minSamples = minSamples

	def expand(self, sample, neighbors):
		"""
		method  expands cluster until border of density
		"""
		cluster = set([sample])
		for neighbor in neighbors:
			if  not neighbor  in self.visited:
				self.visited.append(neighbor)
				self.neighbors[neighbor] = self.get_neigs(neighbor)
				if len(self.neighbors[neighbor]) >= self.minSamples:
					expanded_cluster = self.expand(neighbor, self.neighbors[neighbor])
					cluster = cluster.union(expanded_cluster)
				else:
					cluster.add(neighbor)
		return cluster

	def get_neigs(self, sample):
		"""
		return array of neighbors of sample
		including sample
		it means all elements of X with distance less than eps
		"""

		neighbors = [i   for i,s in enumerate(self.X) if (dist(self.X[sample], s) <self.eps) ]
		return np.array(neighbors)

	def get_labels(self):
		"""
		assign label of 
		each samples of each cluster
		"""
		labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
		for i, cluster in self.clusters.items():
			for sample in cluster:
				labels[sample] = i+1
		return labels

	def predict(self, X):
		"""
		based a dataset returns labels to X
		"""
		self.X = X
		self.clusters = {}
		self.visited = []
		self.neighbors = {}
		n_samples = np.shape(self.X)[0]
		for sample in range(n_samples):
			if  not (sample in self.visited):
				self.neighbors[sample] = self.get_neigs(sample)
				self.visited.append(sample)
				if len(self.neighbors[sample]) >= self.minSamples:
					
					new_cluster = self.expand(sample, self.neighbors[sample])
					self.clusters[len(self.clusters)]=new_cluster
				else:
                                        if self.clusters.get(-1)==None:
                                                self.clusters[-1]=set([sample])
                                        else:
                                                self.clusters[-1].add(sample)

		cluster_labels = self.get_labels()
		return cluster_labels
''


def euclidean_distance( point1, point2):
    """
    calcualates euclidean distance of 2 points. 
    expectation is in form point1 = (x,y)
    """

    sums = 0
    for x1,x2 in zip(point1, point2):
        sums+= ((x1-x2)**2)

    return math.sqrt(sums)

def findEpsilon(dataset, minPts):
    kNearestDist = []
    #List of Lists, whose index corresponds to the point in the dataset. ie: index 2 is the list of distances from all points to point 2
    distances = []

    # print(dataset)
    for p1 in dataset:
        distP1 = []
        for p2 in dataset:
            distP1.append(euclidean_distance(dataset[p1], dataset[p2]))
        distances.append(distP1)

    for i in range(len(distances)):
        sortedDist = sorted(distances[i])
        kNearestDist.append(list(sortedDist)[minPts])
    kNearestDist.sort()
    plt.plot(kNearestDist)
    plt.show()

    maxCurve = []
    for i in range(1, len(kNearestDist)-1):
        secDerivative = kNearestDist[i+1] + kNearestDist[i-1] - 2 * kNearestDist[i]
        maxCurve.append(secDerivative)
    maxValue = max(maxCurve)
    maxIndex = maxCurve.index(maxValue) + 1
    epsilon = kNearestDist[maxIndex]

    kn = KneeLocator(range(0, 200), kNearestDist, curve='convex', direction='increasing', interp_method='interp1d')
    epsilon2 = kNearestDist[kn.knee]

    print(epsilon)
    print(epsilon2)

    return epsilon2


df=pd.read_csv('Use_Data.csv')
df['Gender']=df['Gender'].map({'Male':1,'Female':0})
df=df.drop(['CustomerID'],axis=1)
scaler =RobustScaler()
X=df.to_numpy()

#X=scaler.fit_transform(X)
clf=DBSCAN(0.5,1)
ans=clf.predict(scaler.fit_transform(X))
clustering = DBSCAN2(eps=0.5, min_samples=1).fit(scaler.fit_transform(X))


F="vhHdDx.s+*"
C="bgrcmykww"
cl=[]  
plt.rcParams["figure.figsize"] = (5,5)
# color and shape of the chart

X2,y=make_moons(n_samples=400, shuffle=True, noise=0.1)

for f in F:
        for c in C:
                cl.append(c+f)
print(cl)
plt.grid()
clf=DBSCAN(0.18,1)
ans2=clf.predict(X2)
clustering2 = DBSCAN2(eps=0.18, min_samples=1).fit(X2)

for i in range(len(ans2)):
        plt.plot(X2[i][0],X2[i][1],cl[ans2[i]], markersize=7)

plt.show()
print(ans2)
print(clustering2.labels_)


print("homogeneity score:  ", homogeneity_score(clustering2.labels_,ans2))
print("rand score:         ", rand_score(clustering2.labels_,ans2))
print("completeness score: ", completeness_score(clustering2.labels_,ans2))
