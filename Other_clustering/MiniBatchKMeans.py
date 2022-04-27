import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from data import df, columns

if __name__ == "__main__":
    mmk = MiniBatchKMeans().fit(df)
    ax = plt.axes(projection="3d")
    ax.scatter3D(*columns, marker=".", c=mmk.labels_)
    plt.title("Mini Batch K-Means")
    plt.show()
