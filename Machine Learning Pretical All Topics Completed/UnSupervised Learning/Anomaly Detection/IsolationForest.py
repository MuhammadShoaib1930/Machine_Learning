import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest


data = pd.read_csv(r'datasets\Iris.csv')
x = data.iloc[:,:-1]
x = x.drop(columns=['Id'])
y = LabelEncoder().fit_transform(data['Species'])

model = IsolationForest()
model.fit(x)
predictions = model.predict(x)  # -1 = outlier, 1 = inlier
predictions
