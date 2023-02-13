"""Draw a histogram for each matter to compare house distribution"""
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
# plot
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '.', 'utils'))
from preprocessing import get_numeric_features, replace_empty_nan_mean

# -----------------------------------------------------------------------------
# Program : Histogram
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : no argument will be taken in account (display
    # usage if an argument is provided)
    if len(sys.argv) != 1:
        print("predict: wrong number of arguments\n"
              "Usage: python histogram.py", file=sys.stderr)
        sys.exit()

    # -------------------------------------------------------------------------
    # Open the dataset and load it
    # -------------------------------------------------------------------------
    try:
        df: pd.DataFrame = pd.read_csv("./datasets/dataset_train.csv")
    except:
        print("error when trying to read dataset", file=sys.stderr)
        sys.exit(1)


    # -------------------------------------------------------------------------
    # Clean the data : duplicates / empty values
    # -------------------------------------------------------------------------
    df_num: pd.DataFrame = get_numeric_features(df)
    replace_empty_nan_mean(df_num)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Call main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
