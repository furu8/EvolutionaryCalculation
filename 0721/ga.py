import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class GA:
    def __init__(self, M, D):
        self.M = M
        self.D = D
        self.weight = np.array([7, 5, 1, 9, 6])
        self.value = np.array([50, 40, 10, 70, 55])
        self.wmax  = 15 # 例題1は15、例題2は20
        self.pm = 0.05             # 突然変異確率
        self.tmax = 100            # 最大繰り返し回数

        self.x = np.random.randint(0,2,(M,D)) # 染色体 0 or 1
        self.xnext = np.zeros((M,D)) # 次の染色体
        self.f = np.zeros(M)     # 評価関数を格納
        self.fbest = max(self.f) # 郡全体の最適解(評価関数)(初期化)
        self.xbest = self.x[np.argmax(self.f)] # 郡全体の最適解(染色体)(初期化)

        self.cnvg_plot = []

    def evaluate_ans(self):
        for i in range(self.M):
            w = 0
            for d in range(self.D):
                if self.x[i][d] == 0: # 染色体が0だったら無視
                    continue
                else:
                    self.f[i] += self.value[d]
                    w += self.weight[d]
            if w > self.wmax:
                self.f[i] = 1
        print(self.f)

    def calc_ga(self):
        for t in range(1, self.tmax+1):
            # 各個体の評価値F[i]を計算
            print(t)
            self.evaluate_ans()
            break
                
        #     # 最良値Fbestと最適解Xbest[]の更新
        #     # M個の子個体を生成
        #     for m in range(M):
        #         # ルーレット洗濯で親個体p1,p2を選ぶ
        #         # 交叉する次元d1,d2をランダム生成
                
        #         # d1 < d2となるように入替
        #         if d1 > d2:
        #             d1, d2 = d2, d1
        #         # 二点交叉により、子Xnext[i][]を生成
        #         for d in range(D):
        #             if d <= d1 or d > d2:
        #                 Xnext[i][d] = X[p1][d]
        #             else:
        #                 Xnext[i][d] = X[p2][d]
        #         for d in range(D):
        #             if np.random.rand() < Pm:
        #                 Xnext[i][d] = 1 - Xnext[i][d]
        
        #     # XにXnextを上書き
        #     X = Xnext
        
        # print('解の目的関数値 Fg = {}'.format(Fbest))
        # print('最適解 Xbest = {}'.format(Xbest))            

        # return t, Fbest, cnvg_plot

    # def main():
        # time_list = np.array([])
        # fg_list = np.array([])
        # cnvg_plot_list = []

        # ans_d_list = []
        # ans_fx_list = []
        # ans_fg_mean_list = []
        # ans_fg_var_list = []
        # ans_time_mean_list = []
        
        # df = pd.DataFrame(columns=['d', 'fx_type', 'fg_mean', 'fg_var', 'time_mean'])
            
        # for d in D_list:
        #     for i in range(100): 
        #         ga = GA(M,D)
        #         t, self.fbest, cnvg_plot = ga.calc_ga()
        #         time_list = np.append(time_list,t)
        #         fg_list = np.append(fg_list,Fg)
        #         cnvg_plot_list.append(cnvg)
        #         # break
        #     ans_d_list.append(d)
        #     ans_fx_list.append(fx)
        #     ans_fg_mean_list.append(fg_list.mean())
        #     ans_fg_var_list.append(fg_list.var())
        #     ans_time_mean_list.append(time_list.mean())
            
        # print("解の目的関数値Fg={}".format(ans_fg_mean_list))

        # f['d'] = ans_d_list
        # df['fx_type'] = ans_fx_list
        # df['fg_mean'] = ['{:.3e}'.format(m) for m in ans_fg_mean_list]
        # df['fg_var'] = ['{:.3e}'.format(v) for v in ans_fg_var_list]
        # df['time_mean'] = ans_time_mean_list

        # # df.to_csv('de.csv')
        # print(df)

        
        # t, Fbest, cnvg_plot = 
        


if __name__ == "__main__":
    M = 20                # 個体数
    D_list = [5, 10]      # 解の次元

    ga = GA(M,5)
    ga.calc_ga()
    # print(t)
    print(ga.fbest)