import statsmodels.api as sm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Prepare the target variable: Average like count per user
user_avg_likes = {}
for username, posts in username2posts_train.items():
    avg_likes = np.mean([post.get("like_count", 0) or 0 for post in posts])
    user_avg_likes[username] = avg_likes

# Create the target array
y_like_counts = [user_avg_likes[uname] for uname in train_usernames]
y_like_counts = [0 if np.isnan(val) else val for val in y_like_counts]

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(
    x_post_train, y_like_counts, test_size=0.2, random_state=42
)

lgbm_regressor = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.2,
    max_depth=30,
    min_data_in_leaf=30,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    objective="poisson",  # Poisson objective for count data
    random_state=42
)

# Train the model
lgbm_regressor.fit(x_train, y_train)

# Predict on validation set
y_pred_val = lgbm_regressor.predict(x_val)

# Ensure non-negative predictions
y_pred_val = np.maximum(y_pred_val, 0)

# Evaluate the model
mse = log_mse_like_counts(y_val, y_pred_val)
print(f"MSE on validation set: {mse}")

# Predict on test set
y_test_pred = lgbm_regressor.predict(x_post_test)

# Ensure non-negative predictions
y_test_pred = np.maximum(y_test_pred, 0)

# Create a DataFrame for test predictions
test_predictions = pd.DataFrame({
    "username": test_usernames,
    "predicted_like_count": y_test_pred
})

print(test_predictions.head())