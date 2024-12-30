import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
import joblib

data = pd.read_csv('data.csv')

X = data.drop('true_prob', axis=1)
X = X.drop('state_name', axis=1) 
y = data['true_prob']

# Split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

# Define search space
param_space = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20),
    'max_features': (0.1, 1.0)
}

# Define the model
rf_regressor = RandomForestRegressor(random_state=42)

# Bayesian optimization with 50 steps
np.int = int
opt = BayesSearchCV(
    estimator=rf_regressor,
    search_spaces=param_space,
    n_iter=50,
    random_state=42,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=False
)

# print(X_train, y_train)

# Fit the model
opt.fit(X_train, y_train)
joblib.dump(opt.best_estimator_, 'qraft.pkl')

# Load the saved model
best_model = joblib.load('qraft.pkl')

# Accuracy of the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)