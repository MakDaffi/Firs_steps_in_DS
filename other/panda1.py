import pandas as pd
import random as rd

# Способы задачи серии
m_s = pd.Series([5, 6, 7, 8, 9, 10])
m_s1 = pd.Series([5, 6, 7, 8, 9, 10], index=['a', 'b', 'c', 'd', 'e', 'f'])
m_s2 = pd.Series({'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6})


# Примеры операций над сериями
m_s1[['a', 'c', 'f']] = -1
m_s1[['b', 'd', 'e']] -= 1
print(m_s1[(m_s1 > 0)])


# Серии и ее индексам можно задать имя
m_s2.name = 'numbers'
m_s2.index.name = 'letters'
print(m_s2)


x = []
y = []
for i in range(6):
    x.append(rd.uniform(1, 27))
    y.append(rd.randint(5, 30))
# DataFrame - основной объект библиотеки pandas, по факту представляющий из себя таблицу данных
# По факту таблица представляет собой совокупность нескольких серий
df = pd.DataFrame({
    'Monkeys': ['Ivan', 'Sergey', 'Roma', 'Andrew', 'Nikita', 'Kolya'],
    'Bananas': y,
    'Age': x
})


# Можно выводить определенные колонки таблицы, зная их названия
print(df.Age)
print(df.Bananas)
print(df.Monkeys)


# Индексы DataFrame так же как и серии можно переименовать
df.index = ['I', 'S', 'R', 'A', 'N', 'K']
print(df)
