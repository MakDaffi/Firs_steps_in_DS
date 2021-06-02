import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('mlcourse.ai-master/data/telecom_churn.csv')
print(df.head())

# Признаки по одному
# Количественные

# Гистограма по определенному признаку
# df['Total day minutes'].hist()

# Ящик с усами
# sns.boxplot(df['Total day minutes'])

# Гистограма по всем признакам
# df.hist()

# Категориальные и бинарные признаки

print(df['State'].value_counts().head())
print(df['Churn'].value_counts())

# sns.countplot(df['Churn'])
# sns.countplot(df['State'])

a = df[df['State'].isin(df['State'].value_counts().head().index)]
# sns.countplot(a['State'])

# Взаимодействие признаков

# Количественные с количественными
feat = [f for f in df.columns if 'charge' in f]
print(feat)
# df[feat].hist()
# sns.palplot(df[feat])
c = df[df['Churn']]
b = df[~df['Churn']]
# sns.scatterplot(c['Total eve charge'], c['Total night charge'], color='green', label='True')
# sns.scatterplot(b['Total eve charge'], b['Total night charge'], color='red', label='False')

df.drop(feat, axis=1, inplace=True)
# sns.heatmap(df.corr())

# Котегориальный и бинарный с количественным
# sns.boxplot('Churn', 'Total day minutes', data=df)
# sns.violinplot('Churn', 'Total day minutes', data=df)

# Категориальный с бинарным
sns.countplot('International plan', hue='Churn', data=df)

plt.show()
