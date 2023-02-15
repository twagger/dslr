"""
Module to use a model to predict the price of a car according to its mileage
"""
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
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'classes'))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'utils'))
from MyLogisticRegression import MyLogisticRegression
from preprocessing import get_numeric_features, replace_empty_nan_mean
from standardization import normalize_xset
from metric_functions import accuracy_score_, precision_score_, \
                             recall_score_, f1_score_


# -----------------------------------------------------------------------------
# Program : Predict
# -----------------------------------------------------------------------------
def main():

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    dataset: str = sys.argv[1]
    parser = argparse.ArgumentParser(description='train the logistic model')
    parser.add_argument('dataset', type=str, help='training dataset')
    args = parser.parse_args()
    dataset: str = args.dataset

    # -------------------------------------------------------------------------
    # Create the multi classifier from the parameters.csv file
    # -------------------------------------------------------------------------
    # 1. Update model thetas, mean, std from file if it exists
    thetas = []
    means = []
    stds = []
    try:
        with open('predictions/parameters.csv', 'r') as file:
            reader = csv.DictReader(file) # DictRader will skip the header row
            # I just have one row max in this project
            for row in reader:
                thetas.append(np.array([float(theta) for theta
                              in row['thetas'].split(',')]).reshape(-1, 1))
                # get mean and standard deviation used in standardization
                means.append(np.array([float(mean) for mean
                             in row['means'].split(',')]).reshape(-1, 1))
                stds.append(np.array([float(std) for std 
                             in row['stds'].split(',')]).reshape(-1, 1))

    except FileNotFoundError:
        sys.exit(1)
    except SystemExit:
        sys.exit(1)
    except ValueError as exc:
        print(f"Error when trying to read 'predictions/parameters.csv' : {exc}",
              file=sys.stderr)
        sys.exit(1)

    # 2. create 4 models with the proper thetas, one per class to classify
    models = []
    for i in range(4):
        models.append(MyLogisticRegression(thetas[i]))

    # -------------------------------------------------------------------------
    # Open the test dataset and load it
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
    col_types = [int, None, object, object, object, object, float, float,
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

    # drop correlated feature and label
    df_num.drop('Defense Against the Dark Arts', inplace=True, axis=1)
    df_num.drop('Hogwarts House', inplace=True, axis=1)

    # replace empty / nan values
    replace_empty_nan_mean(df_num)

    # nb features
    nb_features = len(df_num.columns)

    # set X and y
    X = np.array(df_num).reshape(-1, nb_features)
    # autotest ---- To remove for correction
    df_2: pd.DataFrame = pd.read_csv("data/dataset_truth_1.csv")
    y = np.array(df_2['Hogwarts House']).reshape((-1, 1))

    # normalize with means and stds from training
    X_norm, _, _ = normalize_xset(X, means, stds)

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------
    # prediction with the best group of models on test set
    predict = np.empty((X.shape[0], 0))
    for model in models:
        predict = np.c_[predict, model.predict_(X_norm)]
    predict = np.argmax(predict, axis=1).reshape((-1, 1))

    # -------------------------------------------------------------------------
    # Compare with test set labels
    # -------------------------------------------------------------------------
    # relabel the test set y to fit with the classes number
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    y_nums = y.copy()
    for num, house in enumerate(houses):
        y_nums[y_nums == house] = num
    
    # print the metrics of the multi classifier
    for i, house in enumerate(houses):
        print(f'{house.upper():-<40}')
        print(f"{'Accuracy score : ':20}"
              f"{accuracy_score_(y_nums, predict, pos_label=i)}")
        print(f"{'Precision score : ':20}"
              f"{precision_score_(y_nums, predict, pos_label=i)}")
        print(f"{'Recall score : ':20}"
              f"{recall_score_(y_nums, predict, pos_label=i)}")
        print(f"{'F1 score : ':20}{f1_score_(y_nums, predict, pos_label=i)}")
   
    # output a prediction file
    try:
        with open('predictions/houses.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "Hogwarts House"])
            for index, prediction in enumerate(predict):
                writer.writerow([index, houses[int(prediction)]])
    except:
        print("Error when trying to read 'houses.csv'", file=sys.stderr)
        sys.exit(1)


# -----------------------------------------------------------------------------
# Call main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
