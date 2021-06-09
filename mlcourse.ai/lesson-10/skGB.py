import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv('mlcourse.ai-master/data/telecom_churn.csv')

se = LabelEncoder()
df["State"] = se.fit_transform(df["State"])
df["International plan"] = df["International plan"].map(lambda x: 1 if x == "Yes" else 0)
df["Churn"] = df["Churn"].map(lambda x: 1 if x else 0)
df["Voice mail plan"] = df["Voice mail plan"].map(lambda x: 1 if x == "Yes" else 0)
size = int(.7 * df.shape[0])
x_train, y_train = df.drop("Churn", axis=1)[:size], df["Churn"][:size]
x_test, y_test = df.drop("Churn", axis=1)[size:], df["Churn"][size:]

gb = GradientBoostingClassifier()
gb.fit(x_train, y_train)

print("Accuracy: {0:.2f}".format(accuracy_score(y_test, gb.predict(x_test))))
print("Precision: {0:.2f}".format(precision_score(y_test, gb.predict(x_test))))
print("Recall: {0:.2f}".format(recall_score(y_test, gb.predict(x_test))))
print("F1: {0:.2f}".format(f1_score(y_test, gb.predict(x_test))))
