import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
pd.set_option('display.max_rows', None)
data = pd.read_csv(r'datasets\college_student_placement_dataset.csv')
data['Internship_Experience'] = data['Internship_Experience'].map({'Yes':1,'No':0}).astype(int)
data['Placement'] = data['Placement'].map({'Yes':1,'No':0}).astype(int)

x = data.iloc[:,[1,2,4,5,6,7,8,9]]
y = data['CGPA']
x_train , x_test,y_tran, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
model = GradientBoostingRegressor()
model.fit(x_train,y_tran)
print(f'MAE={mean_absolute_error(y_test, model.predict(x_test))*100}, R2={r2_score(y_test, model.predict(x_test))*100}, RMSE={root_mean_squared_error(y_test, model.predict(x_test))*100}, RMSE={mean_squared_error(y_test, model.predict(x_test))*100}')
tt_score = model.score(x_test,y_test)
tn_score = model.score(x_train,y_tran)
print(f'test_score={tt_score}, train_score={tn_score}, difference={abs(tt_score-tn_score)}')

p_data = model.predict(x_test)
newData = x_test
newData['Actual_CGPA']=y_test
newData['Predict_CGPA'] = p_data
print(newData)
