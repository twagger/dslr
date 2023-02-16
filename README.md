# Welcome to dslr 🧙

![](https://metro.co.uk/wp-content/uploads/2017/06/pri_43899462.jpg?quality=90&strip=all&zoom=1&resize=644%2C338)

This project is about **implementing a logistic regression model** to classify student of **Hogwarts** to their respective houses, according to their marks in the differents subjects.

As we have **four houses**, we will need to implement a **multi-classifier** based on four models that use the principe of **one versus all classifiation**.

A first and very important step of this project is **data analysis** and understanding. This first part is essential if we want to prepare the data properly so the model can be efficiently trained.

# Installation instructions

This project is made with Python. It does not need specific compilation or deployement instructions. Just clone in on your local machine and use it !

```sh
git clone git@github.com:twagger/dslr.git
cd dslr
```

# Basic usage instruction

*Consult dataset metrics*
```sh
python3 describe.py data/dataset_train.csv
```

*Draw histogram with data*
```sh
python3 histogram.py
```

*Draw scatter plot with data*
```sh
python3 scatter_plot.py
```

*Draw pair plot with data*
```sh
python3 pair_plot.py
```

*Train multiclass logistic model*
```sh
python3 logreg_train.py data/dataset_train.py
```

*Predict Hogwarts houses with the trained model*
```sh
python3 logreg_predict.py data/dataset_test.py
```

# Main features (to detail)

* Data metrics
* Visual analysis
* Logistic regression

# Extra features (to detail)

* multithreading
* plot learning curves while training the models
* progress bar and training status
* Model metrics

# Libraries used

* Numpy
* Matplotlib
* Seaborn
* tqdm
* threading

# Resources

* [Coursera](https://www.coursera.org/learn/machine-learning) : Andrew Ng's supervised learning course
* [tqdm](https://github.com/tqdm/tqdm) : tqdm documentation
* [Python doc](https://docs.python.org/3/library/threading.html#module-threading) : Documentation about threading in python
* [Python doc](https://docs.python.org/3/library/concurrent.futures.html) : Documentation about parallel tasks in python
* [Matplotlib doc](https://matplotlib.org/stable/api/index) Matplotlib general documentation

# Authors

👨 **Thomas WAGNER**

* Github: [@twagger](https://github.com/twagger/)

👨 **César Claude**

* Github: [@cclaude42](https://github.com/cclaude42)

# Credits

* 🖼️ Illustrative image from : **Harry Potter and the Philosopher's Stone**