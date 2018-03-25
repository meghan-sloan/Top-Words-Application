import numpy as np
import pdb

class NMF():
    '''
    k--int number of topics
    max_iter -- int maximum number of iterations
    '''
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.W = None
        self.H = None
        self.max_iter = max_iter

    def fit(self, V):
        #pdb.set_trace()
        self.W = np.random.random(size=V.shape[0]*self.k).reshape(V.shape[0], self.k)

        for _ in range(self.max_iter):
            self.H = np.linalg.lstsq(self.W,V)[0]
            #pdb.set_trace()
            self.H = self.H * (self.H > 0)
            self.W = np.linalg.lstsq(self.H.T, V.T)[0].T
            self.W = self.W * (self.W > 0)
            cost = self.cost(V)
            # print(cost)
            if cost < 0.1:
                return

    def fit_transform(self,V):
        self.fit(V)
        return self.W, self.H

    def cost(self, V):
        return np.linalg.norm(V-self.W.dot(self.H))



if __name__ == '__main__':
    pass
