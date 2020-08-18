import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_graph(pso_df, de_df):
    plt.plot(pso_df['t'], np.log(pso_df['fg']))
    plt.plot(de_df['t'], np.log(de_df['fg']))
    plt.xlim(-1, 1001)
    plt.ylim(-10, 10)
    plt.show()

def main():
    # argv = sys.argv

    pso_sphere2_df = read_df('pso_sphere2.csv')
    pso_sphere20_df = read_df('pso_sphere20.csv')
    pso_rastrigin2_df = read_df('pso_rastrigin2.csv')
    pso_rastrigin20_df = read_df('pso_rastrigin20.csv')

    de_sphere2_df = read_df('de_sphere2.csv')
    de_sphere20_df = read_df('de_sphere20.csv')
    de_rastrigin2_df = read_df('de_rastrigin2.csv')
    de_rastrigin20_df = read_df('de_rastrigin20.csv')

    plot_graph(pso_sphere2_df, de_sphere2_df)
    plot_graph(pso_sphere20_df, de_sphere20_df)
    plot_graph(pso_rastrigin2_df, de_rastrigin2_df)
    plot_graph(pso_rastrigin20_df, de_rastrigin20_df)

if __name__ == "__main__":
    main()