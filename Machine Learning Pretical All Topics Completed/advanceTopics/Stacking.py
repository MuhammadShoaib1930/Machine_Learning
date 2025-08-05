# StackingClassifier
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, minmax_scale ,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

base_models = [
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC(probability=True))
]

meta_model = LogisticRegression()


data = pd.read_csv(r'datasets\Iris.csv')
x = data.iloc[:,:-1]
x = x.drop(columns=['Id'])
y = data['Species']
y.unique()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)


model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(f"accuracy_score={accuracy_score(y_test,predictions)}")
print(model.predict([[10,10,10,10]]))
sns.scatterplot(x_test)
plt.plot(y_test,color='blue')
plt.plot(predictions,color='green')
plt.show()
confusion_matrix(y_test,predictions)

