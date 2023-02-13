"""Module to train a logistic regression model"""
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
# for plot
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'classes'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'utils'))
from MyLogisticRegression import MyLogisticRegression
from preprocessing import get_numeric_features, replace_empty_nan_mean
from standardization import normalize_xset


# -----------------------------------------------------------------------------
# Program : Train
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : one argument will be taken in account (display
    # usage if anything else is provided)
    if len(sys.argv) != 2:
        print("predict: wrong number of arguments\n"
              "Usage: python logreg_train.py /path/to/dataset.csv",
              file=sys.stderr)
        sys.exit()

    dataset: str = sys.argv[1]

    # -------------------------------------------------------------------------
    # Open the training dataset and load it
    # -------------------------------------------------------------------------
    try:
        df: pd.DataFrame = pd.read_csv(dataset)
    except:
        print("error when trying to read dataset", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Check the data : presence / type
    # -------------------------------------------------------------------------
    col_names = ["Index", "Hogwarts House", "First Name", "Last Name",
                 "Birthday", "Best Hand", "Arithmancy", "Astronomy",
                 "Herbology", "Defense Against the Dark Arts",
                 "Divination", "Muggle Studies", "Ancient Runes",
                 "History of Magic", "Transfiguration", "Potions",
                 "Care of Magical Creatures", "Charms", "Flying"]
    col_types = [int, object, object, object, object, object, float, float,
                 float, float, float, float, float, float, float, float, float,
                 float, float]
    col_check = zip(col_names, col_types)

    # check that the expected columns are here and check their type
    if not set(col_names).issubset(df.columns):
        print(f"Missing columns in '{dataset}' file", file=sys.stderr)
        sys.exit(1)
    for name, type_ in col_check:
        if not df[name].dtype == type_:
            print(f"Wrong column type in '{dataset} file", file=sys.stderr)
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Clean the data : duplicates / empty values / nan
    # -------------------------------------------------------------------------
    df_num: pd.DataFrame = get_numeric_features(df)
    replace_empty_nan_mean(df_num)

    # -------------------------------------------------------------------------
    # Logistic regression
    # -------------------------------------------------------------------------
    # 0. parameters for training
    alpha = 1e-1 # learning rate
    max_iter = 2000 # max_iter

    # drop correlated feature
    df_num.drop('Defense Against the Dark Arts', inplace=True, axis=1)

    # nb features
    nb_features = len(df_num.columns)

    # set X and y
    X = np.array(df_num).reshape(-1, nb_features)
    y = np.array(df['Hogwarts House']).reshape((-1, 1))

    # normalize to ease gradient descent
    X_norm, means, stds = normalize_xset(X)

    # Create label sets to train models
    y_trains = []
    for house in ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]:
        relabel_log = np.vectorize(lambda x: 1 if x == house else 0)
        y_trains.append(relabel_log(y))



    ## MULTITHREADING

    def threadLogReg (y_train):
        model = MyLogisticRegression(np.random.rand(nb_features + 1, 1),
                                     alpha = alpha, max_iter = max_iter)
        model.fit_(X_norm, y_train, plot=False)
        return model

    from multiprocessing.dummy import Pool as ThreadPool

    # Make the Pool of workers
    pool = ThreadPool(4)

    # Open the URLs in their own threads
    # and return the results
    models = pool.map(threadLogReg, y_trains)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    ## MULTITHREADING



    # save the models hyperparameters in parameters.csv
    try:
        with open('predictions/parameters.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["thetas", "means", "stds"])
            for model in models:
                thetas_str = ','.join([f'{theta[0]}' for theta in model.thetas])
                mean_str = ','.join([f'{mean}' for mean in means])
                std_str = ','.join([f'{std}' for std in stds])
                writer.writerow([thetas_str, mean_str, std_str])
    except:
        print("Error when trying to read 'predictions/parameters.csv'", file=sys.stderr)
        sys.exit(1)

# -----------------------------------------------------------------------------
# Call main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
