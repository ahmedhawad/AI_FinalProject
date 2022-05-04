from data import df, columns
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift


def mean_shift(X):
    ms = MeanShift(bandwidth=2)
    ax = plt.axes()
    ax.scatter(x= X[:,0], y= X[:,1], c = ms.fit_predict(X))
    plt.title("Mean Shift")


if __name__ == "__main__":
    mean_shift(df)
    plt.show()
