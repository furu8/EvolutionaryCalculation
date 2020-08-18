import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_graph(df):
    plt.plot(df['t'], np.log(df['fg']))
    plt.show()

def main():
    argv = sys.argv
    df = read_df(argv[1])
    plot_graph(df)

if __name__ == "__main__":
    main()