import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Построение графиков зависимости одного признака от остальных
def f(df, s):
    for i, col in enumerate(df.columns[:]):
        plt.subplot(4, 3, i + 1)
        plt.scatter(df[col], df[s])
        plt.title(col)


# Обучение моделей и оценка качества
def train_validate_report(model, x_train, y_train, x_test, y_test, feature_names, forest=False):
    model.fit(x_train, y_train)
    print("MSE:")
    print(np.sqrt(mean_squared_error(y_test, model.predict(x_test))))
    # Весы признаков
    coef = model.feature_importances_ if forest else model.coef_
    print("Weighers:")
    print(pd.DataFrame(coef, feature_names, columns=['coef']).sort_values(by='coef', ascending=False))


# Загрузка данных
df = pd.read_csv("mlcourse.ai-master/data/bikes_rent.csv")

# Начальный анализ данных
# sns.violinplot(df['season'], df['cnt'])
sns.heatmap(df.corr())

# Подгтовка классов моделей
linreg = LinearRegression()
lasso = Lasso(random_state=17)
ridge = Ridge(random_state=17)
lasso_cv = LassoCV(random_state=17)
ridge_cv = RidgeCV()
forest = RandomForestRegressor(random_state=17, n_estimators=100)

# Подготовка данных для обучения
x, y = df.drop('cnt', axis=1).values, df['cnt'].values
train_part_size = int(.7 * x.shape[0])

x_train, x_test = x[:train_part_size, :], x[train_part_size:, :]
y_train, y_test = y[:train_part_size], y[train_part_size:]
# Масштабирование данных
scaler = StandardScaler()
x_scale_train = scaler.fit_transform(x_train)
x_scale_test = scaler.transform(x_test)

print("Linear Regression: ")
train_validate_report(linreg, x_scale_train, y_train, x_scale_test, y_test, df.columns[:-1])
print("LASSO: ")
train_validate_report(lasso, x_scale_train, y_train, x_scale_test, y_test, df.columns[:-1])
print("Ridge: ")
train_validate_report(ridge, x_scale_train, y_train, x_scale_test, y_test, df.columns[:-1])
print("Random forest: ")
train_validate_report(forest, x_scale_train, y_train, x_scale_test, y_test, df.columns[:-1], True)

plt.show()
