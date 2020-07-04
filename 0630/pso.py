import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sphere_func(D, X, i):
    ans = 0
    for d in range(D):
        Xd = X[i][d]
        print(Xd)
        ans += Xd ** 2
    return ans

def rastringin_func(D, X, i):
    ans = 0
    for d in range(D):
        Xd = X[i][d]
        ans += Xd**2 - 10*np.cos(2*np.pi*Xd) + 10
    return ans

def main(M, D, c, w):
    Tmax = 1000          # 最大繰り返し回数
    Cr = 1e-5            # 終了条件
    x_min, x_max = -5, 5 # 範囲

    X = (x_max - x_min) * np.random.rand(M,D) + x_min # 位置
    V = np.zeros((M,D)) # 速度
    F = np.zeros(M)     # 評価関数を格納
    Fp = np.full(M, float('inf'))      # pbest
    Xp = np.full((M, D), float('inf'))
    Fg = float('inf')                             # gbest
    Xg = np.full(D, float('inf'))

    for t in range(1, Tmax+1):
        for i in range(0, M):
            # F[i] = sphere_func(D, X, i)
            F[i] = rastringin_func(D, X, i)
            if F[i] < Fp[i]:
                Fp[i] = F[i]
                for d in range(D):
                    Xp[i][d] = X[i][d]
                if Fp[i] < Fg:
                    Fg = Fp[i]
                    for d in range(D):
                        Xg[d] = X[i][d]
        if Fg < Cr:
            break
        for i in range(M):
            for d in range(D):
                r1 = np.random.rand()
                r2 = np.random.rand()
                V[i][d] = w*V[i][d] + c*r1*(Xp[i][d] - X[i][d]) + c*r2*(Xg[d] - X[i][d])
                X[i][d] = X[i][d] + V[i][d]
                
    print("終了時刻t={}".format(t))
    print("解の目的関数値Fg={}".format(Fg))
    print("解Xg=[")
    for d in range(1, D-1):
        print("{}".format(Xg[d]))
    print("]\n")

if __name__ == "__main__":
    M = 30     # 粒子数
    D = 5      # 解の次元
    c = 1.494  # PSOのパラメータ
    w = 0.729  # PSOのパラメータ
    main(M, D, c, w)    