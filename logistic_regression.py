import numpy as np

# data
sales = np.array([[1, 0, 0, 1],[1, 1, 0, 0]])
prices = np.array([[37.8, 39.3, 45.9, 41.3]])


class LogisticRegressor:
    def __init__(self, shape=None):
        self.weights = np.random.rand(shape[0], shape[1] + 1) if shape is not None else None

    # cost function
    def total_cost(self, x, y):
        z = self.weights @ x
        p = 1 / (1 + np.exp(-z))
        return np.sum(-(y*np.log(p)+(1-y)*np.log(1-p))) / len(y[0])

    # gradient descent
    def fit(self, x, y, alpha=0.0001, iters=2000):
        DATALEN = len(y[0])
        NOUT = len(y)
        NVARS = len(x)
        if self.weights is None:
            self.weights = np.zeros((NOUT, NVARS + 1))
        xt = np.concatenate((x, np.ones((1, DATALEN))))
        for i in range(iters):
            z = self.weights @ xt
            p = 1 / (1 + np.exp(-z))
            e = p-y
            self.weights -= (e @ xt.T / DATALEN) * alpha
        return self.total_cost(xt, y)

    # prediction
    def predict(self, x):
        xp = np.concatenate((x, np.ones((1, len(x[0])))))
        z = self.weights @ xp
        return 1/(1+np.exp(-z))


lr = LogisticRegressor()
print(lr.fit(prices, sales, alpha=0.00001))
print(lr.predict(np.array([[24.5]])))
