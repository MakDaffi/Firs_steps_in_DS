import pandas as pd
import numpy as np

data = pd.read_csv('mlcourse.ai-master/data/beauty.csv', sep=';')

print(data)
# Выводим краткий анализ данных(медиана, минимальный элемент, максимальный элимент и т.д.)
print(data.describe())
# Обращение по индексу к столбцу(индексом является его название)
print(data['exper'].head(10))
# Вывод на экран указанных столбцов и указанных строк(столбцы указывают по названиям)
print(data.loc[0:5, ['wage', 'looks']])
# Вывод на экран указанных столбцов и указанных строк(столбцы указывают по индексам)
print(data.iloc[:, 0:3])
# Анализ для определенных данных
# Анализ среднего заработка женщин и мужчин
print([data[data['female'] == 1]['wage'].mean(), data[data['female'] == 0]['wage'].mean()])
print(data[(data['female'] == 0) & (data['married'] == 1)]['looks'].mean())

a = data.groupby(['looks', 'female'])

# Анализ данных по категориальным признакам
for look, sub_dat_frame in a:
    print(look)
    print(sub_dat_frame['wage'].median())

print(data.groupby(['looks'])[['wage', 'exper']].agg(np.median))
print(pd.crosstab(data['female'], data['married']))

data['is_reach'] = (data['wage'] > data['wage'].quantile(.75)).astype('int64')

print(data.head())
