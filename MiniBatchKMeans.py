import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


if __name__ == "__main__":
    df = pd.read_csv("Use_Data.csv").replace({"Male": 1, "Female": 0})
    mmk = MiniBatchKMeans().fit(df)
    ax = plt.axes(projection="3d")
    ax.scatter3D(df["Age"], df["AnnualIncome"],
                 df["AnnualIncome"], marker=".", c=mmk.labels_)
    plt.show()
