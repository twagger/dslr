import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


classes = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
           'Flying']


# Scatter plot
def scatter_plot (filename : str):
    # Read data
    try:
        df = pd.read_csv(filename)
    except:
        print("error when trying to read dataset", file=sys.stderr)
        sys.exit(1)

    # Create plot grid
    _, axes = plt.subplots(nrows=len(classes), ncols=len(classes))

    # Turn grid into array
    axes = axes.flatten()

    # Compare each class to each class
    for i, class_x in enumerate(classes):
        for j, class_y in enumerate(classes):
            # Select the right subplot
            ax = axes[i * len(classes) + j]

            # Draw scatter plot
            ax.scatter(df[class_x], df[class_y], s=8, edgecolor='white', linewidth=0.4)
            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

            # Set column titles
            if i == 0:
                # Alternate placement for readability
                if j % 2 == 0:
                    ax.set_title(class_y, fontsize=12)
                else:
                    plt.text(0.5, 1.5, class_y, horizontalalignment='center', fontsize=12, transform = ax.transAxes)
            
            # Set line titles
            if j == 0:
                # Alternate placement for readability
                if i % 2 == 0:
                    ax.set_ylabel(class_x, rotation=90, fontsize=8)
                else:
                    plt.text(-0.3, 0.5, class_x, rotation=90, verticalalignment='center', fontsize=8, transform = ax.transAxes)

    # Show plot
    plt.show()


if __name__ == "__main__":
    scatter_plot("data/dataset_train.csv")