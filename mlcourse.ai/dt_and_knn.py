import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Подготовка данных для обучения
df = pd.read_csv('mlcourse.ai-master/data/telecom_churn.csv', sep=',')
print(df.head())
df.drop(['State', 'Voice mail plan'], axis=1, inplace=True)
print(df['International plan'].head())
df['International plan'] = df['International plan'].map(lambda x: 1 if (x == "Yes") else 0)
print(df['International plan'].head())
y = df['Churn'].astype('int')
x = df.drop('Churn', axis=1)
print(x.shape, y.shape)

# Разбиение подборки на две (обучающающая и тестовая)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)
# Метод дерева решений
first_tree = DecisionTreeClassifier(random_state=17)
print(np.mean(cross_val_score(first_tree, x_train, y_train, cv=10)))
# Метод ближайших соседей
first_knn = KNeighborsClassifier()
print(np.mean(cross_val_score(first_knn, x_train, y_train, cv=10)))

# Настройка max_depth для дерева
tree_pr = {'max_depth': np.arange(1, 11), 'max_features': [0.5, 0.7, 1]}
tree_grid = GridSearchCV(first_tree, tree_pr, cv=10, n_jobs=-1)
tree_grid.fit(x_train, y_train)
print(tree_grid.best_score_, tree_grid.best_params_)
a = tree_grid.predict(x_test)
print(accuracy_score(y_test, a))

knn_pr = {"n_neighbors": range(5, 30, 5)}
knn_grid = GridSearchCV(first_knn, knn_pr, cv=10, n_jobs=-1)
knn_grid.fit(x_train, y_train)
print(knn_grid.best_score_, knn_grid.best_params_)
b = knn_grid.predict(x_test)
print(accuracy_score(y_test, b))
