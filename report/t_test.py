import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_df(path):
    df = pd.read_csv(path)
    return df

def extract_col_parameter(df, col):
    idx_list = [0, 1, 4, 5]
    return [df[col][idx] for idx in idx_list]

def calc_test(pso_mean_list, de_mean_list, pso_var_list, de_var_list, n):
    [print(pso_mean_list[i], de_mean_list[i], pso_var_list[i], de_var_list[i]) for i in range(4)]
    return [pso_mean_list[i] - de_mean_list[i] / np.sqrt((pso_var_list[i] + de_var_list[i]) / (n - 1)) for i in range(4)]

def judge_test(t_list):
    up_limit = 1.972017478
    un_limit = -1.972017478
    name_list = ['sphere2', 'rastrigin2', 'sphere20', 'rastrigin20']

    for i, t in enumerate(t_list):
        print(name_list[i], end=': ')
        if t < un_limit or up_limit < t:
            print('棄却')
        else:
            print('受容')

def main():
    pso_n = 100
    de_n = 100
    df = (pso_n-1) + (de_n-1) # 自由度

    pso_df = read_df('pso.csv')
    de_df = read_df('de.csv')

    print(pso_df)
    print(de_df)
    
    pso_mean_list = extract_col_parameter(pso_df, 'fg_mean')
    de_mean_list = extract_col_parameter(de_df, 'fg_mean')
    pso_var_list = extract_col_parameter(pso_df, 'fg_var')
    de_var_list = extract_col_parameter(de_df, 'fg_var')

    t_list = calc_test(pso_mean_list, de_mean_list, pso_var_list, de_var_list, pso_n)
    
    judge_test(t_list)

if __name__ == "__main__":
    main()