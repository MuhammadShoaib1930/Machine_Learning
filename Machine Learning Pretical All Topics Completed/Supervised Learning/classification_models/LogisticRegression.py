import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, minmax_scale ,LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score ,confusion_matrix
data = pd.read_csv(r'datasets\Iris.csv')
x = data.iloc[:,:-1]
y = data['Species']
mapY = {'Iris-setosa':0, 'Iris-versicolor':0, 'Iris-virginica':1}
y = y.map(mapY)
y.unique()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f"accuracy_score={accuracy_score(y_test,y_pred)}, f1_score={f1_score(y_test,y_pred)},recall_score={recall_score(y_test,y_pred)},precision_score={precision_score(y_test,y_pred)}")
confusion_matrix(y_test,y_pred)
