import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN


data = pd.read_csv(r'datasets\Iris.csv')
x = data.iloc[:,:-1]
x = x.drop(columns=['Id'])
y = LabelEncoder().fit_transform(data['Species'])

model = DBSCAN(eps=0.3, min_samples=3)
model.fit(x)
labels = model.labels_

sns.scatterplot(x)
plt.plot(y,color='blue')
plt.plot(labels,color='green')
plt.show()
confusion_matrix(y,labels)
