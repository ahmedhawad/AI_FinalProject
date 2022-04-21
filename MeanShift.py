from data import df, columns
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

if __name__ == "__main__":
    ms = MeanShift(bandwidth=2).fit(df)
    ax = plt.axes(projection="3d")
    ax.scatter3D(*columns, marker=".", c=ms.labels_)
    plt.title("Mean Shift")
    plt.show()
