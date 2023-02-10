"""
The describe program displays the main information about a dataset
"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
# read csv files
import csv
# nd arrays
import numpy as np
# dataframes
import pandas as pd
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '.', 'classes'))
from Metrics import Metrics


# -----------------------------------------------------------------------------
# Program : Describe
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : no argument will be taken in account (display
    # usage if an argument is provided)
    if len(sys.argv) != 2:
        print("predict: wrong number of arguments\n"
              "Usage: python describe.py /path/to/dataset.csv", file=sys.stderr)
        sys.exit()

    dataset: str = sys.argv[1]

    # -------------------------------------------------------------------------
    # Open the dataset and load it
    # -------------------------------------------------------------------------
    try:
        df: pd.DataFrame = pd.read_csv(dataset)
    except:
        print("error when trying to read dataset", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Clean the data : duplicates / empty values / outliers
    # -------------------------------------------------------------------------
    # duplicated data : keep only the first
    df: pd.DataFrame = df[~df.duplicated()]

    # filter non numerical features
    numeric_features: list = [feature for feature in df.columns
                              if pd.api.types.is_numeric_dtype(df[feature])]
    df_num: pd.DataFrame = df[numeric_features].copy()

    # empty values : replace empty values by the mean of the feature, and add
    # a boolean feature to 'tag' the replaced values (1 if the value has been
    # re created, 0 else)
    for col in df_num.columns:
        # replace nan values with the mean of the feature
        tmp_mean = pd.Series(df_num[col]).sum() / df_num[col].count()
        df_num.loc[:, col] = df_num[col].fillna(value=tmp_mean)
        # compute the metrics
        metrics = Metrics(df_num[col].to_numpy().reshape(-1,1))
        mean = metrics.mean()
        print(f'mean: {mean}')

    # outliers
