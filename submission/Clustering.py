import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import DBSCAN as  skDBSCAN
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



######################################################      Cluster & Graphing        ###################################################### 


##    Constansts     ##
epsilon = 0.2
min_points = 3 



###################### Make Moons Clustering and Graphing ######################

### Make Moons Data ###
X_moons,y = make_moons(n_samples = 500, shuffle=False, noise=0.07)

### Make Moons DBSCANS ###
custom_dbscan_moons=DBSCAN(epsilon,min_points).predict(X_moons)
sk_dbscan_moons = skDBSCAN(eps=epsilon, min_samples=min_points).fit(X_moons)

### Make Moons Extra Algorithms ###
kmeans_moon_model = sklearn.cluster.KMeans(n_clusters = 2)
kmeans_moons = kmeans_moon_model.fit(X_moons)
affinity_moons = sklearn.cluster.AffinityPropagation(damping = 0.9, max_iter = 250, random_state = None).fit(X_moons)
mean_moons = sklearn.cluster.MeanShift().fit(X_moons)

### Make Moons Graphs ###
plt.figure("Group 21 Clustering Graphs",figsize = (15, 10), dpi =80)

plt.subplot(351)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = custom_dbscan_moons)
plt.title('Custom DBSCAN Moon', fontsize = 10)

plt.subplot(352)
plt.title('Sklearn DBSCAN Moon', fontsize = 10)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = sk_dbscan_moons.labels_)

plt.subplot(353)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = kmeans_moons.labels_)
plt.title('Sklearn K-Means Moon (K = 2)', fontsize = 10)

plt.subplot(354)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = affinity_moons.labels_)
plt.title('Affinity Propagation Moon', fontsize = 10)

plt.subplot(355)
plt.scatter(x= X_moons[:,0], y= X_moons[:,1], c = mean_moons.labels_)
plt.title('Mean Shift Moon')
plt.tight_layout()



###################### Make Circles Clustering and Graphing ######################

### Make Circles Data ###
X_circles,y = make_circles(n_samples = 500, shuffle=False, noise=0.02, factor = 0.5)

### Make Circles DBSCANS ###
custom_dbscan_circles=DBSCAN(epsilon,min_points).predict(X_circles)
sk_dbscan_circles = skDBSCAN(eps=epsilon, min_samples=min_points).fit(X_circles)

### Make Circles Extra Algorithms ###
kmeans_circles_model = sklearn.cluster.KMeans(n_clusters = 2)
kmeans_circles = kmeans_moon_model.fit(X_circles)
affinity_circles = sklearn.cluster.AffinityPropagation(damping = 0.9, max_iter = 250, random_state = None).fit(X_circles)
mean_circles = sklearn.cluster.MeanShift().fit(X_circles)


### Make Circles Graphs ###
plt.subplot(356)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = custom_dbscan_circles)
plt.title('Custom DBSCAN Circles', fontsize = 10)

plt.subplot(357)
plt.title('Sklearn DBSCAN Circles', fontsize = 10)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = sk_dbscan_circles.labels_)

plt.subplot(358)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = kmeans_circles.labels_)
plt.title('Sklearn K-Means Circles (K = 2)', fontsize = 10)

plt.subplot(359)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = affinity_circles.labels_)
plt.title('Affinity Propagation Circles', fontsize = 10)

plt.subplot(3,5,10)
plt.scatter(x= X_circles[:,0], y= X_circles[:,1], c = mean_circles.labels_)
plt.title('Mean Shift Circles', fontsize = 10)

plt.tight_layout()


###################### Credit Card User Data Clustering and Graphing ######################

##    Constansts     ##
epsilon = 15
min_points = 8

### Credit Card User Data ###
filepath = "submission/CreditCard_User_Data.csv"
X_user_data = np.array(pd.read_csv(filepath))

### Credit Card User DBSCANS ###
custom_dbscan_user_data=DBSCAN(epsilon,min_points).predict(X_user_data)
sk_dbscan_user_data = skDBSCAN(eps=epsilon, min_samples=min_points).fit(X_user_data)

### Credit Card User Extra Algorithms ###
kmeans_ud_model = sklearn.cluster.KMeans(n_clusters = 2)
kmeans_uds = kmeans_moon_model.fit(X_user_data)
affinity_uds = sklearn.cluster.AffinityPropagation(damping = 0.9, max_iter = 250, random_state = None).fit(X_user_data)
mean_uds = sklearn.cluster.MeanShift().fit(X_user_data)

### Credit Card User Graphs ###
plt.subplot(3,5,11)
plt.scatter(X_user_data[:,0], X_user_data[:,1],X_user_data[:,2],  c = custom_dbscan_user_data)
plt.title('Custom DBSCAN User Data', fontsize = 10)

plt.subplot(3,5,12)
plt.title('Sklearn DBSCAN User Data', fontsize = 10)
plt.scatter(X_user_data[:,0], X_user_data[:,1],X_user_data[:,2], c = sk_dbscan_user_data.labels_)

plt.subplot(3,5,13)
plt.scatter(X_user_data[:,0], X_user_data[:,1],X_user_data[:,2], c = kmeans_uds.labels_)
plt.title('Sklearn K-Means User Data (K = 2)', fontsize = 10)

plt.subplot(3,5,14)
plt.scatter(X_user_data[:,0], X_user_data[:,1],X_user_data[:,2], c = affinity_uds.labels_)
plt.title('Affinity Propagation User Data', fontsize = 10)

plt.subplot(3,5,15)
plt.scatter(X_user_data[:,0], X_user_data[:,1],X_user_data[:,2], c = mean_uds.labels_)
plt.title('Mean Shift User Data', fontsize = 10)

plt.tight_layout()





#This shows all 15 graphs at once
plt.show()





####################################    Printing Accuracy for DBSCANs     #################################### 

print("\n")
print("moons dataset")
print("homogeneity score:  ", homogeneity_score(sk_dbscan_moons.labels_,custom_dbscan_moons))
print("rand score:         ", rand_score(sk_dbscan_moons.labels_,custom_dbscan_moons))
print("completeness score: ", completeness_score(sk_dbscan_moons.labels_,custom_dbscan_moons))
print("\n")

print("circles dataset")
print("homogeneity score:  ", homogeneity_score(sk_dbscan_circles.labels_,custom_dbscan_circles))
print("rand score:         ", rand_score(sk_dbscan_circles.labels_,custom_dbscan_circles))
print("completeness score: ", completeness_score(sk_dbscan_circles.labels_,custom_dbscan_circles))
print("\n")

print("credit card user dataset")
print("homogeneity score:  ", homogeneity_score(sk_dbscan_user_data.labels_,custom_dbscan_user_data))
print("rand score:         ", rand_score(sk_dbscan_user_data.labels_,custom_dbscan_user_data))
print("completeness score: ", completeness_score(sk_dbscan_user_data.labels_,custom_dbscan_user_data))
print("\n")






####################################    Adding your own Dataset     #################################### 


def dbscan_my_dataset():


	print("Do you want to show dbscan your own dataset. yes or no? ")

	x = input("My answer: ")
	
	if x == "yes":
		

		print("We will ask for your filepath, please ensure all columns are numbers")
		filepath = input("filepath: ")
		### Choose your own constants (edit) ###
		epsilon = 15   #choose your own
		min_points = 8 #choose your own


		### Choose your file (edit) ###
		# filepath = "User_Data.csv" #note all values must be numbers

		### Performing DBSCAN (no need to edit) ###
		X_input = np.array(pd.read_csv(filepath))
		custom_dbscan_input = DBSCAN(epsilon,min_points).predict(X_user_data)
		sk_dbscan_input = skDBSCAN(eps=epsilon, min_samples=min_points).fit(X_user_data)


		### Graphing DBSCANs (no need to edit if your graph is 2d) ###

		#to graph, you must edit the number of dimensions you want to see, we have by default a 2d graph
		plt.subplot(121)
		plt.scatter(X_input[:,0], X_input[:,1], c = custom_dbscan_input)
		plt.title('Custom DBSCAN Input')
		plt.subplot(122)
		plt.title('Sklearn DBSCAN Input')
		plt.scatter(X_input[:,0], X_input[:,1], c = sk_dbscan_input.labels_)


		### Printing closeness custom DBSCAN to sklearn DBSCAN (no need to edit if your graph is 2d) ###

		print("input dataset")
		print("homogeneity score:  ", homogeneity_score(sk_dbscan_input.labels_,custom_dbscan_input))
		print("rand score:         ", rand_score(sk_dbscan_input.labels_,custom_dbscan_input))
		print("completeness score: ", completeness_score(sk_dbscan_input.labels_,custom_dbscan_input))


		#showing graph
		plt.show()

dbscan_my_dataset()



















