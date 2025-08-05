import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge, Lasso , ElasticNet
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error
pd.set_option('display.max_rows', None)
data = pd.read_csv(r'datasets\college_student_placement_dataset.csv')
data['Internship_Experience'] = data['Internship_Experience'].map({'Yes':1,'No':0}).astype(int)
data['Placement'] = data['Placement'].map({'Yes':1,'No':0}).astype(int)

x = data.iloc[:,[1,2,4,5,6,7,8,9]]
y = data['CGPA']
py = PolynomialFeatures(degree=3)

x = py.fit_transform(x)
x_train , x_test,y_tran, y_test = train_test_split(x,y,test_size=0.25,random_state=90)
model = Lasso()
model.fit(x_train,y_tran)
print(f'MAE={mean_absolute_error(y_test, model.predict(x_test))*100}, R2={r2_score(y_test, model.predict(x_test))*100}, RMSE={root_mean_squared_error(y_test, model.predict(x_test))*100}, RMSE={mean_squared_error(y_test, model.predict(x_test))*100}')
tt_score = model.score(x_test,y_test)
tn_score = model.score(x_train,y_tran)
print(f'test_score={tt_score}, train_score={tn_score}, difference={abs(tt_score-tn_score)}')

# Ridge random_state = 96 Degree = 3
# MAE=24.64459966674159, R2=96.30219512939019, RMSE=28.654050265723143, RMSE=8.210545966305885
# test_score=0.9630219512939019, train_score=0.9624822943991378, difference=0.0005396568947640956

# Lasso  random_state = 90 Degree = 2
# 90 0.9421203483507332
# MAE=30.869864380753143, R2=93.16549031529343, RMSE=38.83535272517845, RMSE=15.081846212890254
# test_score=0.9316549031529342, train_score=0.9367161687229466, difference=0.005061265570012408

#LinearRegression random_state = 96 Degree = 2
# 96 0.963613432643402
# MAE=24.7837724372021, R2=96.26701239171342, RMSE=28.701318396829944, RMSE=8.23765677716209
# test_score=0.9626701239171342, train_score=0.9620356611212224, difference=0.0006344627959118387