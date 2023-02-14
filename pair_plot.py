import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


classes = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
           'Flying']


# Pair plot
def pair_plot (filename : str):
    # Read data
    df = pd.read_csv(filename)

    # Select features
    df = df[classes]

    # Draw pairplot
    sns.pairplot(df)

    # Show plot
    plt.show()


if __name__ == "__main__":
    pair_plot("data/dataset_train.csv")