import numpy as np
import copy

class LinearRegression:
    def __init__(self, phi='poly',*,K=None,alpha=None, beta=None, max_iterate=10000, end_eps=1.0e-6):
        if K is None:
            assert("input K!!!")
        self.K = K
        self.alpha = alpha
        self.fix_alpha = True
        if self.alpha is None:
            self.fix_alpha = False
        self.beta = beta
        self.fix_beta = True
        if self.beta is None:
            self.fix_beta = False
        if phi =='poly':
            self.phi =( lambda k,x: x**k)
        else:
            self.phi = phi
        self.weight_expv = None
        self.weight_var = None
        self.Phi = None
        self.max_iterate = max_iterate
        self.end_eps = end_eps

    def fit(self, data_X, data_Y):
        if self.fix_alpha and self.fix_beta:
            return
        N = len(data_Y)
        Phi = np.zeros((self.K,N))
        IK = np.identity(self.K)
        for k in range(self.K):
            for n in range(N):
                Phi[k][n] = self.phi(k,data_X[n])
        eigens, _ = np.linalg.eig(Phi.T @ Phi)
        #エビデンス最大化を反復法で計算する
        #初期化
        alpha = self.alpha if self.fix_alpha else 1.0
        beta = self.beta if self.fix_beta else 1.0
        #反復法
        for i in range(self.max_iterate):
            tmp_alpha = alpha
            tmp_beta = beta
            gamma = 0.0
            for n in range(N):
                gamma += eigens[n] / (eigens[n] + beta * alpha)
            C = Phi @ Phi.T + alpha*beta*IK
            weight_expv = data_Y@Phi.T @ np.linalg.inv(C)
            if not self.fix_alpha:
                alpha = gamma / np.inner(weight_expv, weight_expv)
            if not self.fix_beta:
                tmp = data_Y - weight_expv @ Phi
                beta = np.inner(tmp,tmp)/(N-gamma)
            if abs(tmp_alpha - alpha) < self.end_eps and abs(tmp_beta - beta)<self.end_eps:
                break
        #パラメータを格納
        self.alpha = self.alpha if self.fix_alpha else alpha
        self.beta = self.beta if self.fix_beta  else beta
        self.weight_expv = weight_expv
        self.weight_var = np.linalg.inv(C)
        self.Phi = Phi

        

    def score(self,data_X,data_Y):
        N = len(data_Y)
        if self.Phi is None:
            Phi = np.zeros((self.K,N))
            for k in range(self.K):
                for n in range(N):
                    Phi[k][n] = self.phi(k,data_X[n])
        output = data_Y @ (np.identity(N) - self.Phi.T @ self.weight_var @ self.Phi) @ data_Y.T
        output /= self.beta
        output +=(N-self.K)*np.log(self.beta)
        tmp = np.log(np.linalg.det(self.weight_var)) + self.K*np.log(self.alpha)
        return ((tmp - output) / 2).real

    def predict(self, x):
        output = 0.0
        for k in range(self.K):
            output += self.weight_expv[k] * self.phi(k,x)
        return output

    def Ktuning(self,data_X,data_Y,maxK=None):
        if maxK is None:
            maxK = self.K
        best_model=None
        best_score=None
        for K in range(2,maxK):
            model = copy.deepcopy(self)
            model.K = K
            model.fit(data_X,data_Y)
            score=model.score(data_X,data_Y)
            if best_score is None or best_score < score:
                best_score = score
                best_model = model
        return best_model, best_score


