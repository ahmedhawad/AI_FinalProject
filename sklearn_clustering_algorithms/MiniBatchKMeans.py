import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from data import df, columns


def mini_batch_kmeans(X):
    mmk = MiniBatchKMeans()
    ax = plt.axes()
    ax.scatter(x= X[:,0], y= X[:,1], c = mmk.fit_predict(X))
    plt.title("Mini Batch K-Means")


if __name__ == "__main__":
    mini_batch_kmeans(df)
    plt.show()
