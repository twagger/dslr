import sys, csv, math
import pandas as pd
import numpy as np


# Analyze functions
def mean (data : pd.DataFrame):
    tot = 0
    for i in range(data.size):
        tot += data.iloc[i]
    return tot / data.size

def stddev (data : pd.DataFrame, mean : float):
    tot = 0
    for i in range(data.size):
        tot += (data.iloc[i] - mean) ** 2
    return math.sqrt(tot / data.size)

def percentile (data : pd.DataFrame, percentile : float) -> float:
    i = float(percentile * (data.size - 1))
    if i.is_integer():
        return data.iloc[int(i)]
    else:
        return data.iloc[math.floor(i)] * (1 - i % 1) + data.iloc[math.ceil(i)] * (i % 1)

def analyze_feature (feature : pd.DataFrame) -> dict:
    feature = feature.dropna().sort_values()
    m = mean(feature)
    return {
        "name": feature.name,
        "count": feature.size,
        "mean": m,
        "std": stddev(feature, m),
        "min": percentile(feature, 0),
        "25%": percentile(feature, 0.25),
        "50%": percentile(feature, 0.5),
        "75%": percentile(feature, 0.75),
        "max": percentile(feature, 1)
    }


# Display
def display (data):
    SIZE = 5
    # For each block of SIZE
    for i in range(0, len(data), SIZE):
        slce = data[i:i+SIZE]

        # Display feature names
        line = f"{0:<6}"
        for desc in slce:
            if len(desc['name']) > 14:
                line += f"{desc['name'][:11]:>13s}..."
            else:
                line += f"{desc['name']:>16s}"
        print(line)

        # For each legend
        for legend in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
            # Start line with legend
            line = f"{legend:6s}"
            # Append to line each feature's stat
            for desc in slce:
                if -1000000 <= desc[legend] <= 1000000:
                    line += f"{desc[legend]:16.6f}"
                else:
                    line += f"{desc[legend]:16.6e}"
            # Print line
            print(line)

        print()


# Describe
def describe (filename : str):
    # Read data
    df = pd.read_csv(filename)

    # Get analysis for each numeric column
    stats = [analyze_feature(df[col]) for col in df if pd.api.types.is_numeric_dtype(df[col])]

    # Format print
    display(stats)

def main ():
    if len(sys.argv) > 1:
        describe(sys.argv[1])
    else:
        print("Please provide a CSV file to read :", file=sys.stderr)
        print("    python3 describe.py filename.csv", file=sys.stderr)

if __name__ == "__main__":
    main()