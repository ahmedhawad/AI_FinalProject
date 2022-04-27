import numpy
import pandas as pd


def get_data(file):
    df = pd.read_csv(file)
    allowed_types = [numpy.int64, int, numpy.float32,
                     numpy.float16, numpy.float64, numpy.int32]
    for col, t in dict(df.dtypes).items():
        if t not in allowed_types and col in list(df.columns):
            df = df.drop(columns=col)
    return df


df = get_data("Use_Data.csv")
columns = [df["Age"], df["AnnualIncome"], df["SpendingScore"]]
