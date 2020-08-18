import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_df(path):
    df = pd.read_csv(path)
    return df

def plot_graph(name, pso_df, de_df):
    plt.title(name)
    plt.plot(pso_df['t'], np.log(pso_df['fg']), label='pso')
    plt.plot(de_df['t'], np.log(de_df['fg']), label='de')
    plt.xlim(-1, 1001)
    plt.ylim(-12, 7)
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(name)
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

    plot_graph('sphere2',pso_sphere2_df, de_sphere2_df)
    plot_graph('sphere20',pso_sphere20_df, de_sphere20_df)
    plot_graph('rastrigin2',pso_rastrigin2_df, de_rastrigin2_df)
    plot_graph('rastrigin20',pso_rastrigin20_df, de_rastrigin20_df)

if __name__ == "__main__":
    main()