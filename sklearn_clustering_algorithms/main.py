
from AffinityPropagation import affinity_propagation
from MeanShift import mean_shift
from MiniBatchKMeans import mini_batch_kmeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import pandas as pd
from data import df
from DBSCAN import dbscan
x, labels_true = make_moons(n_samples=1000, noise=0.08)


x = StandardScaler().fit_transform(x)


def run_all(df):
    plt.figure(1)
    affinity_propagation(df)
    plt.figure(2)
    dbscan(df)
    plt.figure(3)
    mean_shift(df)
    plt.figure(4)
    mini_batch_kmeans(df)


if __name__ == "__main__":
    run_all(df)
    plt.show()
