"""Module to train a logistic regression model"""
# -----------------------------------------------------------------------------
# Module imports
# -----------------------------------------------------------------------------
# system
import os
import sys
import argparse
# read csv files
import csv
# nd arrays
import numpy as np
# dataframes
import pandas as pd
# for plot
import matplotlib
import matplotlib.pyplot as plt
# multithreading and multiprocessing
import threading
from multiprocessing.dummy import Pool as ThreadPool
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'classes'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'utils'))
from MyLogisticRegression import MyLogisticRegression
from preprocessing import get_numeric_features, replace_empty_nan_mean
from standardization import normalize_xset
from validators import type_validator, shape_validator


# -----------------------------------------------------------------------------
# Program : Train
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : one argument will be taken in account (display
    # usage if anything else is provided)
    parser = argparse.ArgumentParser(description='train the logistic model')
    parser.add_argument('dataset', type=str, help='training dataset')
    parser.add_argument('--plot', action='store_const', const=True,
                        default=False, help='plot learning curves')
    parser.add_argument('--gd', type=str, default='GD',
                        choices=['GD', 'SGD', 'MBGD'],
                        help='gradient descent algorithm')
    parser.add_argument('--maxiter', type=int, default=1000,
                        choices=range(1, 100000), metavar='{1...100000}',
                        help='the number of iterations of the logistic regression')
    
    args = parser.parse_args()
    dataset: str = args.dataset
    PLOT: bool = args.plot
    GD_TYPE: str = args.gd

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
    max_iter = args.maxiter # max_iter

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

    # -------------------------------------------------------------------------
    # Plotting (plt in thread, with logistic regression in multiprocessing)
    # -------------------------------------------------------------------------
    if PLOT:
        # Create plot
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes = axes.flatten()

        # Create event to control supervisor function
        event = threading.Event()

        # Define plt supervisor function
        def supervisor ():
            while not event.is_set():
                # Redraw plot
                fig.canvas.draw()
                # Wait to make plot visible
                plt.pause(0.01)

        # Run supervisor to refresh plot
        t = threading.Thread(target=supervisor)
        t.start()

        # Logistic regression part
        # Make pool of workers
        pool = ThreadPool(4)

        # Define multiprocessed function
        def threadLogReg (y_train, ax):
            model = MyLogisticRegression(np.random.rand(nb_features + 1, 1),
                                         alpha = alpha, max_iter = max_iter)
            model.fit_(X_norm, y_train, plot = True, ax = ax, gd = GD_TYPE)
            return model

        # Run logreg function on each y_train in its own process
        models = pool.starmap(threadLogReg, zip(y_trains, axes))

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()

        # Set event to stop supervisor function and join
        event.set()
        t.join()

    # -------------------------------------------------------------------------
    # No plotting (just logistic regression in multiprocessing)
    # -------------------------------------------------------------------------
    else:
        # Make pool of workers
        pool = ThreadPool(4)

        # Define multiprocessed function
        def threadLogReg (y_train):
            model = MyLogisticRegression(np.random.rand(nb_features + 1, 1),
                                         alpha = alpha, max_iter = max_iter)
            model.fit_(X_norm, y_train, plot = False, gd = GD_TYPE)
            return model

        # Run logreg function on each y_train in its own process
        models = pool.map(threadLogReg, y_trains)

        # Close the pool and wait for the work to finish
        pool.close()
        pool.join()


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
        print("Error when trying to read 'predictions/parameters.csv'",
              file=sys.stderr)
        sys.exit(1)

# -----------------------------------------------------------------------------
# Call main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
