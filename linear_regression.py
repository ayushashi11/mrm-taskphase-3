import numpy as np

# data
sales = np.array([[22.1, 10.4, 18.3, 18.5]])
prices = np.array([[37.8, 39.3, 45.9, 41.3]])


class LinearRegressor:
    def __init__(self, shape=None):
        self.weights = np.random.rand(shape[0], shape[1] + 1) if shape is not None else None

    # cost function
    def total_cost(self, x, y):
        return np.sum((self.weights @ x - y) ** 2) / 2 / len(y[0])

    # gradient descent
    def fit(self, x, y, alpha=0.0001, iters=2000):
        DATALEN = len(y[0])
        NOUT = len(y)
        NVARS = len(x)
        if self.weights is None:
            self.weights = np.random.rand(NOUT, NVARS + 1)
        xt = np.concatenate((x, np.ones((1, DATALEN))))
        for i in range(iters):
            error = (self.weights @ xt) - y
            self.weights -= (error @ xt.T / DATALEN) * alpha
        return self.total_cost(xt, y)

    # prediction
    def predict(self, x):
        xp = np.concatenate((x, np.ones((1, len(x[0])))))
        return self.weights @ xp


lr = LinearRegressor()
print(lr.fit(prices, sales, alpha=0.00001))
print(lr.predict(np.array([[24.5]])))
