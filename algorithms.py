import numpy as np
from scipy.special import expit
from collections import Counter

# Utility
def one_hot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh


# 1. Linear Regression OVR

class LinearRegressionOVR:
    def __init__(self, lambda_reg=1e-2):
        self.lambda_reg = lambda_reg

    def fit(self, X, y):
        Y = one_hot(y)
        n, d = X.shape
        A = X.T @ X + self.lambda_reg * np.eye(d)
        self.W = np.linalg.solve(A, X.T @ Y)

    def predict(self, X):
        logits = X @ self.W
        return np.argmax(logits, axis=1)


# 2. Logistic Regression OVR

class LogisticRegressionOVR:
    def __init__(self, lr=0.5, iters=200):
        self.lr = lr
        self.iters = iters

    def fit(self, X, y):
        n, d = X.shape
        K = 10
        Y = one_hot(y)
        self.W = np.zeros((d, K))

        for _ in range(self.iters):
            logits = X @ self.W
            preds = expit(logits)
            grad = X.T @ (preds - Y) / n
            self.W -= self.lr * grad

    def predict(self, X):
        logits = X @ self.W
        return np.argmax(logits, axis=1)


# 3. K-Means Classifier

class KMeansClassifier:
    def __init__(self, k=50, iters=20):
        self.k = k
        self.iters = iters

    def fit(self, X, y):
        n, d = X.shape
        idx = np.random.choice(n, self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.iters):
            dists = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            labels = np.argmin(dists, axis=1)
            for j in range(self.k):
                pts = X[labels == j]
                if len(pts) > 0:
                    self.centroids[j] = pts.mean(axis=0)

        # Assign majority class to each cluster
        self.cluster_label = np.zeros(self.k, dtype=int)
        for j in range(self.k):
            pts = np.where(labels == j)[0]
            if len(pts):
                self.cluster_label[j] = Counter(y[pts]).most_common(1)[0][0]

    def predict(self, X):
        dists = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        lbl = np.argmin(dists, axis=1)
        return self.cluster_label[lbl]


# 4. Gaussian Model

class GaussianClassifier:
    def fit(self, X, y):
        self.means = []
        self.vars = []
        for c in range(10):
            Xc = X[y == c]
            self.means.append(Xc.mean(axis=0))
            self.vars.append(Xc.var(axis=0) + 1e-5)

        self.means = np.array(self.means)
        self.vars = np.array(self.vars)

    def predict(self, X):
        log_likelihoods = []
        for c in range(10):
            diff = (X - self.means[c]) ** 2 / self.vars[c]
            ll = -0.5 * np.sum(diff + np.log(self.vars[c]), axis=1)
            log_likelihoods.append(ll)

        log_likelihoods = np.array(log_likelihoods).T
        return np.argmax(log_likelihoods, axis=1)


# Ensemble (Weighted Voting)

class EnsembleClassifier:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        votes = np.zeros((len(X), 10))
        for model, w in zip(self.models, self.weights):
            pred = model.predict(X)
            for i, p in enumerate(pred):
                votes[i, p] += w
        return np.argmax(votes, axis=1)
