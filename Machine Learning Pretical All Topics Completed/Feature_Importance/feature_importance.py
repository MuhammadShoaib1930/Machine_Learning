
#  Feature Importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importance
importance = model.feature_importances_
df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
print(df.sort_values(by='Importance', ascending=False))
