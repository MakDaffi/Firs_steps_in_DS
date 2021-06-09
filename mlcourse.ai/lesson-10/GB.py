import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score

class GradientBoosting(BaseEstimator):

    def sigma(self, z):
        z = z.reshape([z.shape[0], 1])
        z[z > 100] = 100
        z[z < -100] = -100
        return 1. / (1 + np.exp(-z))


    def log_los_grad(self, y, p):
        y = y.reshape([y.shape[0], 1])
        p = p.reshape([p.shape[0], 1])
        return (p - y) / (p * (1 - p))


    def mse_grad(self, y, p):
        return 2 * (p - y.reshape([y.shape[0], 1])) / y.shape[0]


    def __init__(self, n_estimators=10, learning_rate=0.01, max_depth=3, random_state=17, loss='mse', debug=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.initialization = lambda y: np.mean(y) * np.ones([y.shape[0], 1])
        self.debug = debug

        if loss == 'log_loss':
            self.objective = log_loss
            self.objective_grad = self.log_los_grad
        elif loss == 'mse':
            self.objective = mean_squared_error
            self.objective_grad = self.mse_grad

        self.trees_ = []
        self.loss_by_iter = []

        if self.debug:
            self.residuals = []
            self.temp_pred = []


    def fit(self, x, y):
        self.x = x
        self.y = y
        b = self.initialization(y)

        prediction = b.copy()

        for t in range(self.n_estimators):

            resid = - self.objective_grad(y, prediction)

            if self.debug:
                self.residuals.append(resid)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(x, resid)

            b = tree.predict(x).reshape([x.shape[0], 1])

            if self.debug:
                self.temp_pred.append(b)

            prediction += self.learning_rate * b

            self.loss_by_iter.append(self.objective(y, prediction))

        self.train_pred = prediction

        if self.loss_name == 'log_loss':
            self.train_pred = self.sigma(self.train_pred)


    def predict_proba(self, x):

        pred = np.mean(self.y) * np.ones([x.shape[0], 1])

        for t in range(self.n_estimators):
            pred += self.learning_rate * self.trees_[t].predict(x).reshape([x.shape[0], 1])

        if self.loss_name == 'log_loss':
            return self.sigma(pred)
        else:
            return pred


    def predict(self, x):

        pred_probs = self.predict_proba(x)

        if self.loss_name == 'log_loss':
            max_accuracy = 0
            best_thres = 0

            for thres in np.linspace(0.01, 1.01, 100):
                acc = accuracy_score(self.y, self.train_pred > thres)

                if acc >= max_accuracy:
                    max_accuracy = acc
                    best_thres = thres

            return pred_probs > best_thres
        else:
            return pred_probs
