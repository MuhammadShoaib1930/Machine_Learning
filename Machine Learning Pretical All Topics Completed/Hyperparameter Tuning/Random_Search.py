from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scipy.stats import randint

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()

# Parameter distribution
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(3, 10)
}

# Random Search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(X, y)

print("Best Params:", random_search.best_params_)