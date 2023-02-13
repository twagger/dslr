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
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'classes'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'utils'))
from Metrics import Metrics
from preprocessing import get_numeric_features, replace_empty_nan_mean


# -----------------------------------------------------------------------------
# Program : Describe
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : one argument will be taken in account (display
    # usage if anything else is provided)
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
    # Clean the data : duplicates / empty values / nan 
    # -------------------------------------------------------------------------
    df_num: pd.DataFrame = get_numeric_features(df)
    replace_empty_nan_mean(df_num)

    # -------------------------------------------------------------------------
    # Metrics : compute and display
    # -------------------------------------------------------------------------
    stacked = None
    for i, col in enumerate(df_num.columns):
        # Get metrics for a column
        metrics = Metrics(df_num[col].to_numpy().reshape(-1,1))
        # Display when 5 columns
        if i % 5 == 0:
            print(stacked) if stacked is not None else None
            previous = "\n\n\n\n\n\n\n\n"
            stacked = None
        # Stack
        current = metrics.__str__(first = i % 5 == 0, name=col, sp=15)
        stacked = '\n'.join([prev + curr for prev, curr in
                             zip(previous.split('\n'), current.split('\n'))])
        previous = stacked
    print(stacked) if stacked is not None else None
        

# -----------------------------------------------------------------------------
# Call main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
