# First install:
# !pip install scikit-optimize
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()

# Bayesian Search Space
param_space = {
    'n_estimators': (10, 200),
    'max_depth': (3, 10)
}

# Bayes Search
opt = BayesSearchCV(model, param_space, n_iter=10, cv=3)
opt.fit(X, y)

print("Best Params:", opt.best_params_)

