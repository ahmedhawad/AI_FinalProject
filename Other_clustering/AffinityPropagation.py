from sklearn.cluster import AffinityPropagation
from data import df, columns
import matplotlib.pyplot as plt
if __name__ == "__main__":
    ap = AffinityPropagation().fit(df)
    ax = plt.axes(projection="3d")
    ax.scatter3D(*columns, marker=".", c=ap.labels_)
    plt.title("Affinity Propagation")
    plt.show()
