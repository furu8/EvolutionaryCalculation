import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy

class GA:
    def __init__(self, M, D, W, V, Wmax):
        self.M = M
        self.D = D
        self.weight = W
        self.value = V
        self.wmax  = Wmax          # 例題1は15、例題2は20
        self.pm = 0.05             # 突然変異確率
        self.tmax = 100            # 最大繰り返し回数

        self.x = np.random.randint(0,2,(M,D)) # 染色体 0 or 1
        self.xnext = np.zeros((M,D)) # 次の染色体
        self.f = np.zeros(M)     # 評価関数を格納
        self.fbest = max(self.f) # 郡全体の最適解(評価関数)(初期化)
        self.xbest = self.x[np.argmax(self.f)] # 郡全体の最適解(染色体)(初期化)

        self.cnvg_plot = []

    def evaluate_fitness_value(self):
        for i in range(self.M):
            self.f[i] = 0
            w = 0
            for d in range(self.D):
                if self.x[i][d] == 0: # 染色体が0だったら無視
                    continue
                else:
                    self.f[i] += self.value[d]
                    w += self.weight[d]
            # 制限を超えたら小さい値を適当に
            if w > self.wmax:
                self.f[i] = 1

    def calc_ga(self):
        for t in range(1, self.tmax+1):
            # 各個体の評価値F[i]を計算
            self.evaluate_fitness_value()
            # 最良値Fbestと最適解Xbest[]の更新
            if max(self.f) > self.fbest:
                self.fbest = max(self.f)
                self.xbest = np.copy(self.x[np.argmax(self.f)])
            # M個の子個体を生成
            for i in range(self.M):
                # ルーレット選択で親個体p1,p2を選ぶ
                sumf_list = self.f / sum(self.f)
                p1, p2 = np.random.choice(range(self.M), 2, replace=False, p=sumf_list)
                # 交叉する次元d1,d2をランダム生成
                d1, d2 = np.random.choice(range(self.D), 2, replace=False)
                # d1 < d2となるように入替
                if d1 > d2:
                    d1, d2 = d2, d1
                # 二点交叉により、子Xnext[i][]を生成
                for d in range(self.D):
                    if d <= d1 or d > d2:
                        self.xnext[i][d] = self.x[p1][d]
                    else:
                        self.xnext[i][d] = self.x[p2][d]
                # 突然変異
                for d in range(self.D):
                    if np.random.rand() < self.pm:
                        self.xnext[i][d] = 1 - self.xnext[i][d]
        
            # XにXnextを上書き
            self.x = np.copy(self.xnext)   

if __name__ == "__main__":
    M = 20                # 個体数
    wmax_list = [15, 20]
    W_list = [[7, 5, 1, 9, 6], [3, 6, 5, 4, 8, 5, 3, 4, 8, 2]]
    V_list = [[50, 40, 10, 70, 55], [70, 120, 90, 70, 130, 80, 40, 50, 30, 70]]

    w = 0
    for W, V in zip(W_list, V_list):
        print(W)
        count = 0
        D = len(W) # 解の次元数
        for i in range(100):
            ga = GA(M, D, np.array(W), np.array(V), wmax_list[w])
            ga.calc_ga()
            print(i, ga.fbest, ga.xbest)
            if ga.fbest == 125 or ga.fbest == 420:
                count += 1
        w += 1
        print(D, count)