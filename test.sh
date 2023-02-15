#!/bin/bash

test_logreg () {
    # Remove previous predictions
    rm -rf predictions/parameters.csv
    rm -rf predictions/houses.csv
    # Train model
    python logreg_train.py data/dataset_train.csv $1 2> /dev/null
    # Predict houses
    python logreg_predict.py data/dataset_test.csv > /dev/null
    # Evaluate the work
    python evaluate.py > /dev/null
    # Check the return and print
    if [[ $? -eq 0 ]]
    then
        echo "python logreg_train.py data/dataset_train.csv $1  :  ✅"
    else
        echo "python logreg_train.py data/dataset_train.csv $1  :  ❌"
    fi
}

# GD
echo && echo "GD :"
test_logreg "--gd GD   --maxiter 100        "
test_logreg "--gd GD   --maxiter 1000       "
test_logreg "--gd GD   --maxiter 10000      "
test_logreg "--gd GD   --maxiter 99999      "
# MBGD
echo && echo "MBGD :"
test_logreg "--gd MBGD --maxiter 5          "
test_logreg "--gd MBGD --maxiter 20         "
test_logreg "--gd MBGD --maxiter 100        "
# SGD
echo && echo "SGD :"
test_logreg "--gd SGD  --maxiter 1          "
test_logreg "--gd SGD  --maxiter 5          "
test_logreg "--gd SGD  --maxiter 10         "
# Plot
echo && echo "Plotting :"
test_logreg "--gd GD   --maxiter 100  --plot"
test_logreg "--gd MBGD --maxiter 5    --plot"
test_logreg "--gd SGD  --maxiter 1    --plot"