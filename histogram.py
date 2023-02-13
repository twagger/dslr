import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


classes = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
           'Flying']

houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

colors = ['firebrick', 'lime', 'royalblue', 'gold']


# Histogram
def histogram (filename : str):
    # Read data
    df = pd.read_csv(filename)

    # Create plot grid
    _, axes = plt.subplots(nrows=3, ncols=5)

    # Turn grid into array of length [number of features]
    axes = axes.flatten()[:len(classes)]

    for ax, class_ in zip(axes, classes):
        # Set title
        ax.title.set_text(class_)
        # Draw histogram for each house
        for house, color in zip(houses, colors):
            ax.hist(df.loc[df['Hogwarts House'] == house, class_], color=color, alpha=0.75, label=house)

    # Set house legend in corner
    handles, labels = axes[0].get_legend_handles_labels()
    plt.legend(handles, labels, loc='center')

    # Show plot
    plt.show()


if __name__ == "__main__":
    histogram("data/dataset_train.csv")