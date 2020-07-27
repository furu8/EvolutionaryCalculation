import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main(M, D, Pm, Tmax, fx):
    Wmax  = 15 # 例題1は15、例題2は20
    weight = np.zeros(D)
    value = np.zeros(D)
    X = np.random.randint(0,2,(m,d)) # 染色体 0 or 1
    Xnext = np.zeros((M,D)) # 次の染色体
    F = np.zeros(M)     # 評価関数を格納
    Fbest = float('inf')
    Xbest = np.full(D, float('inf'))
    cnvg_plot = []

    for t in range(1, Tmax+1):
        # 各個体の評価値F[i]を計算
        # 最良値Fbestと最適解Xbest[]の更新

        # M個の子個体を生成
        for i in range(M):
            # ルーレット洗濯で親個体p1,p2を選ぶ
            # 交叉する次元d1,d2をランダム生成
            
            # d1 < d2となるように入替
            if d1 > d2:
                d1, d2 = d2, d1
            # 二点交叉により、子Xnext[i][]を生成
            for d in range(D):
                if d <= d1 or d > d2:
                    Xnext[i][d] = X[p1][d]
                else:
                    Xnext[i][d] = X[p2][d]
            for d in range(D):
                if np.random.rand() < Pm:
                    Xnext[i][d] = 1 - Xnext[i][d]
    
        # XにXnextを上書き
        X = Xnext
    
    print('解の目的関数値 Fg = {}'.format(Fbest))
    print('最適解 Xbest = {}'.format(Xbest))            

    return t, Fbest, cnvg_plot

if __name__ == "__main__":
    M = 20                # 個体数
    D_list = [2, 5, 20]   # 解の次元
    Pm = 0.05             # 突然変異確率
    Tmax = 100            # 最大繰り返し回数
    fx_list = ['sphere', 'rastrigin']

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
                t, Fg, cnvg = main(M, d, Pm, Tmax, fx)
                time_list = np.append(time_list,t)
                fg_list = np.append(fg_list,Fg)
                cnvg_plot_list.append(cnvg)
                # break
            ans_d_list.append(d)
            ans_fx_list.append(fx)
            ans_fg_mean_list.append(fg_list.mean())
            ans_fg_var_list.append(fg_list.var())
            ans_time_mean_list.append(time_list.mean())
         
        print("解の目的関数値Fg={}".format(ans_fg_mean_list))


    df['d'] = ans_d_list
    df['fx_type'] = ans_fx_list
    df['fg_mean'] = ['{:.3e}'.format(m) for m in ans_fg_mean_list]
    df['fg_var'] = ['{:.3e}'.format(v) for v in ans_fg_var_list]
    df['time_mean'] = ans_time_mean_list

    # df.to_csv('de.csv')
    print(df)
        