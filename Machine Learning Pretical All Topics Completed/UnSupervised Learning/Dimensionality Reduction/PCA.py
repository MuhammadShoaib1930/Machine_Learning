import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


data = pd.read_csv(r'datasets\Iris.csv')
x = data.iloc[:,:-1]
x = x.drop(columns=['Id'])
y = LabelEncoder().fit_transform(data['Species'])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(x)
X_reduced