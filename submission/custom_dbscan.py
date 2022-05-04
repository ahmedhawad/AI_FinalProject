import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN as  DBSCAN2
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import completeness_score


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
				labels[sample] = i
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



# df=pd.read_csv('Use_Data.csv')
# df['Gender']=df['Gender'].map({'Male':1,'Female':0})
# df=df.drop(['CustomerID'],axis=1)
# scaler =RobustScaler()
# X=df.to_numpy()

#X=scaler.fit_transform(X)
# clf=DBSCAN(0.5,1)
# ans=clf.predict(scaler.fit_transform(X))
# clustering = DBSCAN2(eps=0.5, min_samples=1).fit(scaler.fit_transform(X))



# F="vhHdDx.s+*"
# C="bgrcmykww"
# cl=[]  
# plt.rcParams["figure.figsize"] = (5,5)
# color and shape of the chart


# for f in F:
#         for c in C:
#                 cl.append(c+f)
# print(cl)
# plt.grid()







X,y = make_moons(n_samples=1000, shuffle=False, noise=0.07)

#########
epsilon = 0.159
min_points = 3

#########
custom_dbscan=DBSCAN(epsilon,min_points).predict(X)
sk_dbscan = DBSCAN2(eps=epsilon, min_samples=min_points).fit(X)

print(X)



plt.figure(figsize = (10, 5))
plt.subplot(121)
plt.scatter(x= X[:,0], y= X[:,1], c = custom_dbscan)
plt.title('Our own DBSCAN result')
plt.subplot(122)
plt.title('Sklearn DBSCAN result')
plt.scatter(x= X[:,0], y= X[:,1], c = sk_dbscan.labels_)
plt.show()



print('Using our own DBSCAN: ', custom_dbscan)
print('Sklearn DBSCAN: ', sk_dbscan.labels_)

print("homogeneity score:  ", homogeneity_score(sk_dbscan.labels_,custom_dbscan))
print("rand score:         ", rand_score(sk_dbscan.labels_,custom_dbscan))
print("completeness score: ", completeness_score(sk_dbscan.labels_,custom_dbscan))
