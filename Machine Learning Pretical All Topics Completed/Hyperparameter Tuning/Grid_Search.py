
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Model and parameter grid
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 3, 5]
}

# Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("Best Params:", grid_search.best_params_)
