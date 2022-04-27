from sklearn.cluster import AffinityPropagation
from data import df, columns
import matplotlib.pyplot as plt


def affinity_propagation(X):
    ap = AffinityPropagation()
    ax = plt.axes()
    ax.scatter(x= X[:,0], y= X[:,1], c = ap.fit_predict(X))
    plt.title("Affinity Propagation")


if __name__ == "__main__":
    affinity_propagation(df)
    plt.show()
