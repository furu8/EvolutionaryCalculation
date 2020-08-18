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

def main(D, fx):
    M = 30               # 粒子数
    c = 1.494            # PSOのパラメータ
    w = 0.729            # PSOのパラメータ
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

    plot_t_list = []
    plot_fg_list = []

    for t in range(1, Tmax+1):
        plot_t_list.append(t)
        plot_fg_list.append(Fg)
        for i in range(M):
            if fx == 'sphere':
                F[i] = sphere_func(D, X, i)
            elif fx == 'rastrigin':
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
            break
        for i in range(M):
            r1 = np.random.rand()
            r2 = np.random.rand()
            V[i] = w*V[i] + c*r1*(Xp[i] - X[i]) + c*r2*(Xg - X[i])
            X[i] = X[i] + V[i]
           
    return t, Fg, plot_t_list, plot_fg_list

if __name__ == "__main__":
    D_list = [2, 5, 20]     # 解の次元
    fx_list = ['sphere', 'rastrigin']

    time_list = np.array([])
    fg_list = np.array([])
    
    ans_d_list = []
    ans_fx_list = []
    ans_fg_mean_list = []
    ans_fg_var_list = []
    ans_time_mean_list = []

    ans_plot_t_list = np.array([])
    ans_plot_fg_list = np.array([])
    
    df = pd.DataFrame(columns=['d', 'fx_type', 'fg_mean', 'fg_var', 'time_mean'])
    
    for d in D_list:
        for fx in fx_list:
            print(d, fx)
            for i in range(10): 
                print(i+1, end=' ')
                t, Fg, plot_t_list, plot_fg_list = main(d, fx)
                time_list = np.append(time_list, t)
                fg_list = np.append(fg_list, Fg)
                plt.plot(plot_t_list, plot_fg_list)
                ans_plot_t_list = np.append(ans_plot_t_list, plot_t_list)
                ans_plot_fg_list = np.append(ans_plot_fg_list, plot_fg_list)
            print(ans_plot_t_list.shape)
            ans_d_list.append(d)
            ans_fx_list.append(fx)
            ans_fg_mean_list.append(fg_list.mean())
            ans_fg_var_list.append(fg_list.var())
            ans_time_mean_list.append(time_list.mean())
            # plt.show()

    df['d'] = ans_d_list
    df['fx_type'] = ans_fx_list
    df['fg_mean'] = ['{:.3e}'.format(m) for m in ans_fg_mean_list]
    df['fg_var'] = ['{:.3e}'.format(v) for v in ans_fg_var_list]
    df['time_mean'] = ans_time_mean_list

    # df.to_csv('pso.csv')
    print(df)
        