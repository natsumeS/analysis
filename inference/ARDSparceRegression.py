import numpy as np

class ARDSparceRegression:
	def __init__(self, phi='poly', K=5):
		self.K=K
		self.alpha = np.array([1.0 for i in range(K)])
		self.beta = 1.0
		self.k_list = np.array([i for i in range(K)])
		if phi =='poly':
			self.phi =( lambda k,x: x**k)
		else:
			self.phi = phi
		self.weight_expv = None
		self.weight_var = None
		self.Phi = None

	def fit(self, data_X, data_Y,*,max_iterate=10000, end_eps=1.0e-6, is_nan_inf=1.0e5):
		N = len(data_Y)
		Phi = np.zeros((self.K,N))
		IK = np.identity(self.K)
		for k in range(self.K):
			for n in range(N):
				Phi[k][n] = self.phi(k,data_X[n])
        #エビデンス最大化を反復法で計算する
        #初期化
		alpha = np.array([1.0 for i in range(self.K)])
		beta = 1.0
		A = np.zeros((self.K,self.K))
		for i in range(self.K):
			A[i][i] = alpha[i]
        #反復法
		for i in range(max_iterate):
			tmp_alpha = alpha
			tmp_beta = beta
			C = A + Phi @ Phi.T / beta
			invC = np.linalg.inv(C)
			weight_expv = data_Y @ Phi.T @ invC / beta
			gamma = 0.0
			for i in range(self.K):
				tmp_gamma = 1.0 - alpha[i] * invC[i][i]
				gamma += tmp_gamma
				alpha[i] = tmp_gamma / (weight_expv[i] * weight_expv[i])
				A[i][i] = alpha[i]
            #alphaで発散しているものを削除
			nan_bools = alpha < is_nan_inf
			A = A[nan_bools,:]
			A = A[:,nan_bools]
			Phi = Phi[nan_bools,:]
			IK = IK[nan_bools,:]
			IK = IK[:,nan_bools]
			weight_expv = weight_expv[nan_bools]
			invC = invC[nan_bools,:]
			invC = invC[:,nan_bools]
			alpha = alpha[nan_bools]
			tmp_alpha = tmp_alpha[nan_bools]
			self.K = np.sum(nan_bools)
			self.k_list = self.k_list[nan_bools]
            #
			tmp = data_Y - weight_expv @ Phi
			beta = (np.inner(tmp,tmp))/(N - gamma)
			if np.linalg.norm(tmp_alpha-alpha,ord=2) < end_eps and abs(tmp_beta - beta)<end_eps:
				break
        #パラメータを格納
		self.alpha = alpha
		self.beta = beta
		self.weight_expv = weight_expv
		self.weight_var = invC
		self.Phi = Phi

	def score(self,data_X,data_Y):
		N = len(data_Y)
		if self.Phi is None:
			Phi = np.zeros((self.K,N))
			for k,k2 in enumerate(self.k_list):
				for n in range(N):
					Phi[k][n] = self.phi(k2,data_X[n])
		output = data_Y @ (np.identity(N) - self.Phi.T @ self.weight_var @ self.Phi) @ data_Y.T
		output /= self.beta
		output += N*np.log(self.beta)
		tmp = 0.0
		for i in range(self.K):
			tmp += np.log(self.alpha[i])
		tmp += np.log(np.linalg.det(self.weight_var))
		return (tmp - output) / 2

	def predict(self, x):
		output = 0.0
		for i,k in enumerate(self.k_list):
			output += self.weight_expv[i] * self.phi(k,x)
		return output