from asyncio import protocols
from sklearn.cluster import DBSCAN
from data import df, columns
import matplotlib.pyplot as plt


def dbscan(X):
    db = DBSCAN(eps=0.2)
    ax = plt.axes()
    ax.scatter(x= X[:,0], y= X[:,1], c = db.fit_predict(X))
    print("fit predict: ", db.fit_predict(X))
    plt.title("DBSCAN")


if __name__ == "__main__":
    dbscan(df)
    plt.show