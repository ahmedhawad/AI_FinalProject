import pandas as pd
from numpy import zeros_like
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler







dataset = "User_Data.csv"

df = pd.read_csv(dataset)



epsilon = .3
min_clusters = 4

X = df.values
X = StandardScaler().fit_transform(X)


db = DBSCAN(eps= epsilon, min_samples= min_clusters,).fit(X)
core_samples_mask = zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels_pred = db.labels_
n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
n_noise_ = list(labels_pred).count(-1)





print(db.labels_)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)





import matplotlib.pyplot as plt





ax = plt.axes(projection="3d")

ax.scatter3D(df["Age"], df["AnnualIncome"], df["AnnualIncome"], marker = ".", c= labels_pred)






plt.title(f"Total Number of Clusters :{n_clusters_}")

plt.show()