import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score

reviews = shuffle(pd.read_csv("mlcourse.ai-master/data/IMDB Dataset.csv"))

reviews['sentiment'] = reviews['sentiment'].map(lambda x: 1 if x == "positive" else 0)

train_part_size = int(0.7 * reviews.shape[0])
reviews_train = reviews[:train_part_size]
reviews_test = reviews[train_part_size:]

cv = CountVectorizer()
x_train, y_train = cv.fit_transform(reviews_train['review']), reviews_train['sentiment']
x_test, y_test = cv.transform(reviews_test['review']), reviews_test['sentiment']

logit = LogisticRegression(random_state=17, n_jobs=-1)
sgd_logit = SGDClassifier(loss='log', max_iter=70, random_state=17, n_jobs=-1)

logit.fit(x_train, y_train)
sgd_logit.fit(x_train, y_train)
print(accuracy_score(y_test, logit.predict(x_test)), accuracy_score(y_test, sgd_logit.predict(x_test)))
