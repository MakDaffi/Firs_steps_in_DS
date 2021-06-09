import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


# Подготовка поля для отрисовки графиков
fig, axes = plt.subplots(2, 1)


# Визуализация PCA
def vis_pca(res, kmeans):
    axes[0].scatter(res[:, 0], res[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
    plt.title('PCA')


# Визуализация Total points vs. Total assists
def vis1(df1, kmeans):
    axes[1].scatter(df1['pts'], df1['ast'], c=kmeans.labels_, s=50, cmap='viridis')
    plt.xlabel('Total points')
    plt.ylabel('Total assitances')


# Обучение модели для задачи кластеризации
df = pd.read_csv("mlcourse.ai-master/data/nba_2013.csv")
kmeans = KMeans(n_clusters=5, random_state=1)
df1 = shuffle(df._get_numeric_data().dropna(axis=0))
pca = PCA(n_components=2)
res = pca.fit_transform(df1)
kmeans.fit(res)

# Применение вышеописанных функций визуализации
vis1(df1, kmeans)
vis_pca(res, kmeans)

plt.show()
