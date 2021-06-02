# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression


# Визуализация Логистической регресии
def plot_boundary(clf, x, grid_step=.01, poly_featurizer=None):
    x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step))
    z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z)


data = pd.read_csv('mlcourse.ai-master/data/microchip_tests.txt', header=None, names=('test1', 'test2', 'released'))
print(data)
x = data.iloc[:, :2].values
y = data.iloc[:, 2].values

# Построение полиномов по x до 7 степени (Для случая, когда нельзя провести линию)
poly = PolynomialFeatures(degree=7)
x_poly = poly.fit_transform(x)
# C - параметр сложности модели
C = 10
logit = LogisticRegression(C=C, n_jobs=-1, random_state=17)
logit.fit(x_poly, y)
plot_boundary(logit, x, grid_step=.01, poly_featurizer=poly)
# Предсказание
print(logit.predict(poly.transform(np.c_[[0.75, 0.5], [0.5, 0.5]])))

# Визуализация точек
plt.scatter(x[y == 1, 0], x[y == 1, 1], c='green', label='Выпущенные')
plt.scatter(x[y == 0, 0], x[y == 0, 1], c='red', label='Бракованные')
plt.xlabel("Тест 1")
plt.ylabel("Тест 2")
plt.title('2 теста микрочипов')
plt.legend()
plt.show()
