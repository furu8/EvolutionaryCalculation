import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def init_fx(M, D, X, fx):
    f = np.zeros(M)
    
    if fx == 'sphere':
        for m in range(M):
            ans = 0
            for d in range(D):
                Xd = X[m][d]
                ans += Xd ** 2
            f[m] = ans
    elif fx == 'rastrigin':
        for m in range(M):
            ans = 0
            for d in range(D):
                Xd = X[m][d]
                ans += ((Xd**2) - 10*np.cos(2*np.pi*Xd) + 10)
            f[m] = ans
    else:
        print('引数fxが間違っています')

    return f

def sphere_func(D, U):
    ans = 0
    for d in range(D):
        Xd = U[d]
        ans += Xd ** 2
    return ans

def rastringin_func(D, U):
    ans = 0
    for d in range(D):
        Xd = U[d]
        ans += ((Xd**2) - 10*np.cos(2*np.pi*Xd) + 10)
    return ans

def main(D, fx):
    M = 30                # 個体数
    c = 0.9               # DEのパラメータ
    Fw = 0.5              # DEのパラメータ
    Tmax = 1000          # 最大繰り返し回数
    Cr = 0.9             # 終了条件
    x_min, x_max = -5, 5 # 範囲

    X = (x_max - x_min) * np.random.rand(M,D) + x_min # 位置
    Xnew = np.zeros((M,D)) 
    V = np.zeros(D) # 速度
    U = np.zeros(D)     # 解候補
    F = np.zeros(M)     # 評価関数を格納
    Ftmp = 0     # 評価関数を格納
    Fend = 1e-5 # 終了条件
    Fbest = float('inf')
    Xbest = np.full(D, float('inf'))

    plot_t_list = []
    plot_fg_list = []

    # 初期値によるF = f(x^0ベクトル) の計算
    F = init_fx(M, D, X, fx)

    for t in range(1, Tmax+1):
        plot_t_list.append(t)
        plot_fg_list.append(Fbest)
        for i in range(M):
            # 3個体選ぶ
            a, b, c = np.random.choice(np.arange(M), 3, replace=False)
            V = X[a] + Fw * (X[b] - X[c])
            # 突然変異
            Jr = np.random.randint(D)
            # 交叉
            for j in range(D):
                ri = np.random.rand()
                if ri < Cr or j == Jr:
                    U[j] = V[j]
                else:
                    U[j] = X[i][j]
            
            if fx == 'sphere':
                Ftmp = sphere_func(D, U)
            elif fx == 'rastrigin':
                Ftmp = rastringin_func(D, U)
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
                Xnew[i] = X[i]

        # XをXnewで上書き     
        X = Xnew
        if Fbest < Fend:
            break
           
    # print("終了時刻t={}".format(t))
    # print("解の目的関数値Fbest={}".format(Fbest))

    return t, Fbest, plot_t_list, plot_fg_list

if __name__ == "__main__":
    D_list = [2]   # 解の次元
    
    fx_list = ['sphere', 'rastrigin']

    time_list = np.array([])
    fg_list = np.array([])
    cnvg_plot_list = []
    
    ans_d_list = []
    ans_fx_list = []
    ans_fg_mean_list = []
    ans_fg_var_list = []
    ans_time_mean_list = []

    ans_plot_t_mean_list = []
    ans_plot_fg_mean_list = []
    
    df = pd.DataFrame(columns=['d', 'fx_type', 'fg_mean', 'fg_var', 'time_mean'])
    
    for d in D_list:
        for fx in fx_list:
            t_df = pd.DataFrame()
            fg_df = pd.DataFrame()
            print(d, fx)
            for i in range(10): 
                t, Fg, plot_t_list, plot_fg_list = main(d, fx)
                time_list = np.append(time_list, t)
                fg_list = np.append(fg_list, Fg)
                t_df = pd.concat([t_df, pd.Series(plot_t_list)], axis=1)
                fg_df = pd.concat([fg_df, pd.Series(plot_fg_list)], axis=1)
            
            ans_d_list.append(d)
            ans_fx_list.append(fx)
            ans_fg_mean_list.append(fg_list.mean())
            ans_fg_var_list.append(fg_list.var())
            ans_time_mean_list.append(time_list.mean())
            ans_plot_t_mean_list.append(t_df.T.mean())
            ans_plot_fg_mean_list.append(fg_df.T.mean())
         
        print("解の目的関数値Fg={}".format(ans_fg_mean_list))


    df['d'] = ans_d_list
    df['fx_type'] = ans_fx_list
    df['fg_mean'] = ['{:.3e}'.format(m) for m in ans_fg_mean_list]
    df['fg_var'] = ['{:.3e}'.format(v) for v in ans_fg_var_list]
    df['time_mean'] = ans_time_mean_list

    # df.to_csv('de.csv')
    print(df)

    plt.plot(ans_plot_t_mean_list[1], ans_plot_fg_mean_list[1])
    plt.show()