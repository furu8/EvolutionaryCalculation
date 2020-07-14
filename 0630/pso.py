import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def sphere_func(D, X, i):
    ans = 0
    for d in range(D):
        Xd = X[i][d]
        ans += Xd ** 2
    return ans

def rastringin_func(D, X, i):
    ans = 0
    for d in range(D):
        Xd = X[i][d]
        ans += ((Xd**2) - 10*np.cos(2*np.pi*Xd) + 10)
    return ans

def main(M, D, c, w, fx):
    Tmax = 1000          # 最大繰り返し回数
    Cr = 1e-5            # 終了条件
    x_min, x_max = -5, 5 # 範囲

    X = (x_max - x_min) * np.random.rand(M,D) + x_min # 位置
    V = np.zeros((M,D)) # 速度
    F = np.zeros(M)     # 評価関数を格納
    Fp = np.full(M, float('inf'))      # pbest
    Xp = np.full((M, D), float('inf'))
    Fg = float('inf')                  # gbest
    Xg = np.full(D, float('inf'))

    cnvg_plot = []

    for t in range(1, Tmax+1):
        for i in range(0, M):
            if fx == 'sphere':
                F[i] = sphere_func(D, X, i)
            elif fx == 'rastringin':
                F[i] = rastringin_func(D, X, i)
            else:
                print('引数fxが間違っています')

            if F[i] < Fp[i]:
                Fp[i] = F[i]
                Xp[i] = X[i]
                if Fp[i] < Fg:
                    Fg = Fp[i]
                    Xg = X[i]
        if Fg < Cr:
            cnvg_plot.append([t, Fg])
            break
        for i in range(M):
            r1 = np.random.rand()
            r2 = np.random.rand()
            V[i] = w*V[i] + c*r1*(Xp[i] - X[i]) + c*r2*(Xg - X[i])
            X[i] = X[i] + V[i]
           
    # print("終了時刻t={}".format(t))
    # print("解の目的関数値Fg={}".format(Fg))
    # print("解Xg={}".format(Xg))
    return t, Fg, cnvg_plot

if __name__ == "__main__":
    M = 30                  # 粒子数
    D_list = [2]     # 解の次元
    c = 1.494               # PSOのパラメータ
    w = 0.729               # PSOのパラメータ
    fx_list = ['sphere', 'rastringin']


    time_list = np.array([])
    fg_list = np.array([])
    cnvg_plot_list = []
    
    ans_d_list = []
    ans_fx_list = []
    ans_fg_mean_list = []
    ans_fg_var_list = []
    ans_time_mean_list = []
    
    df = pd.DataFrame(columns=['d', 'fx_type', 'fg_mean', 'fg_var', 'time_mean'])
    
    for d in D_list:
        for fx in fx_list:
            for i in range(100): 
                t, Fg, cnvg = main(M, d, c, w, fx)
                time_list = np.append(time_list,t)
                fg_list = np.append(fg_list,Fg)
                cnvg_plot_list.append(cnvg)
            ans_d_list.append(d)
            ans_fx_list.append(fx)
            ans_fg_mean_list.append(fg_list.mean())
            ans_fg_var_list.append(fg_list.var())
            ans_time_mean_list.append(time_list.mean())


    df['d'] = ans_d_list
    df['fx_type'] = ans_fx_list
    df['fg_mean'] = ['{:.3e}'.format(m) for m in ans_fg_mean_list]
    df['fg_var'] = ['{:.3e}'.format(v) for v in ans_fg_var_list]
    df['time_mean'] = ans_time_mean_list

    # df.to_csv('pso.csv')
    print(df)
        