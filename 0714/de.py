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
    Cr = 0.9             # 終了条件
    x_min, x_max = -5, 5 # 範囲

    X = (x_max - x_min) * np.random.rand(M,D) + x_min # 位置
    Xnew = (x_max - x_min) * np.random.rand(M,D) + x_min # 位置
    V = np.zeros((M,D)) # 速度
    U = np.zeros(d)     # 解候補
    F = np.zeros(M)     # 評価関数を格納
    Ftmp = 0     # 評価関数を格納
    Fend = 1e-5 # 終了条件
    Fbest = np.full(M, float('inf'))      # pbest
    Xbest = np.full((M, D), float('inf'))

    cnvg_plot = []

    for t in range(1, Tmax+1):
        for i in range(M):
            for j in range(D):
                # 3個体選ぶ
                
                # Vを作成
                
                pass
            
            Jr = np.random.rand()

            for j in range(D):
                ri = np.random.rand()
                if ri < Cr or j == Jr:
                    U[j] = V[j]
                else:
                    U[j] = X[i][j]
            
            if fx == 'sphere':
                Ftmp = sphere_func(D, X, i)
            elif fx == 'rastringin':
                Ftmp = rastringin_func(D, X, i)
            else:
                print('引数fxが間違っています')

            if Ftmp < F[i]:
                F[i] = Ftmp
                # XnewをUで上書き
                Xnew[i] = U
                # Fbest、Xbest更新
                Fbest = Ftmp
                Xbest = X[i]
            else:
                # XnewをXで上書き
                Xnew[i] = X

        if Fbest < Fend:
            cnvg_plot.append([t, Fg])
            break
           
    # print("終了時刻t={}".format(t))
    # print("解の目的関数値Fg={}".format(Fg))
    # print("解Xg={}".format(Xg))
    return t, Fg, cnvg_plot

if __name__ == "__main__":
    M = 30                # 個体数
    D_list = [2]     # 解の次元
    c = 0.9               # DEのパラメータ
    Fw = 0.5              # DEのパラメータ
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
                t, Fg, cnvg = main(M, d, c, Fw, fx)
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
        