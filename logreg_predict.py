"""
Module to use a model to predict the price of a car according to its mileage
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
# for plot
import matplotlib.pyplot as plt
# user modules
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'classes'))
from MyLogisticRegression import MyLogisticRegression


# -----------------------------------------------------------------------------
# Program : Predict
# -----------------------------------------------------------------------------

def main():

    # -------------------------------------------------------------------------
    # Argument management
    # -------------------------------------------------------------------------
    # argument management : one argument will be taken in account (display
    # usage if anything else is provided)
    if len(sys.argv) != 2:
        print("predict: wrong number of arguments\n"
              "Usage: python logreg_predict.py /path/to/dataset.csv",
              file=sys.stderr)
        sys.exit()

    dataset: str = sys.argv[1]

    
    # -------------------------------------------------------------------------
    # Linear regression
    # -------------------------------------------------------------------------
    # 1. Update model thetas, mean, std from file if it exists
    thetas = []
    means = []
    std = []
    try:
        with open('parameters.csv', 'r') as file:
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

            # check that thetas are valid
            if np.isnan(thetas).any() is True:
                print('Something when wrong during the training, '
                      'the parameters are invalid.', file=sys.stderr)
                sys.exit(1)

    except FileNotFoundError:
        sys.exit(1)
    except SystemExit:
        sys.exit(1)
    except:
        print("Error when trying to read 'parameters.csv'", file=sys.stderr)
        sys.exit(1)

    # 2. create 4 models, one per class to classify
    models = []
    for i in range(4):
        models.append(MyLogisticRegression(thetas[i]))

    # 2. predict one value (the prompted one) with standardized mileage
    mileage_norm = (mileage - mean) / std
    predicted_price = MyLR.predict_(np.array([[mileage_norm]]))

    # -------------------------------------------------------------------------
    # Display prediction
    # -------------------------------------------------------------------------
    # 3. display the predicted value to the user
    print(f'For a mileage of {mileage},'
          f' the predicted price is {predicted_price[0][0]}')

    # -------------------------------------------------------------------------
    # Plot prediction aside with the training dataset data
    # -------------------------------------------------------------------------
    # open and load the training dataset
    try:
        df = pd.read_csv("./data.csv")
    except:
        print("Error when trying to read dataset", file=sys.stderr)
        sys.exit(1)

    # check that the expected columns are here and check their type
    if not set(['km', 'price']).issubset(df.columns):
        print("Missing columns in 'data.csv' file", file=sys.stderr)
        sys.exit(1)
    if not (df.km.dtype == float or df.km.dtype == int) \
            or not (df.price.dtype == float or df.price.dtype == int):
        print("Wrong column type in 'data.csv' file", file=sys.stderr)
        sys.exit(1)

    # set x and y
    x = np.array(df['km']).reshape((-1, 1))
    y = np.array(df['price']).reshape((-1, 1))

    # normalize x data
    x_norm = (x - mean) / std
    y_hat = MyLR.predict_(x_norm)

    # plot
    plt.figure()
    plt.scatter(x, y, marker='o', label='training data')
    plt.scatter(mileage, predicted_price[0][0], marker='o',
                label='predicted data')
    plt.plot(x, y_hat, color='red', label='prediction function')
    plt.legend()
    plt.show()

# -----------------------------------------------------------------------------
# Call main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()