import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV

# load the data
data = pd.read_csv('data.csv')

# remove the y
X = data.drop('true_probability', axis=1)
X = X.drop('state_name', axis=1)
y = data['true_probability']

# split in training and testing data (15% to testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# parameters
param_space = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20),
    'max_features': (0.1, 1.0)
}

# defining the model
rf = RandomForestRegressor(random_state=42)

# use bayesian optimization
np.int = int
opt = BayesSearchCV(
    estimator=rf,
    search_spaces=param_space,
    n_iter=50, 
    random_state=42,
    cv=5,  
    scoring='neg_mean_squared_error',  # scoring metric
    return_train_score=False
)

# print(X_train, y_train)

# fitting
opt.fit(X_train, y_train)


# save the model and load it
joblib.dump(opt.best_estimator_, 'qraft1.pkl')
best_model = joblib.load('qraft1.pkl')

# testing
y_pred = best_model.predict(X_test)

# computing mse
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)