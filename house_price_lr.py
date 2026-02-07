# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 2: Load Dataset
data = pd.read_csv("train.csv")
print(data.head())


# STEP 3: Select Features and Target
features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = data['SalePrice']


# STEP 4: Handle Missing Values
features = features.fillna(features.mean())


# STEP 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)


# STEP 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


# STEP 7: Predict Prices
y_pred = model.predict(X_test)


# STEP 8: Evaluate Model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))


# STEP 9: Model Coefficients
coefficients = pd.DataFrame({
    "Feature": features.columns,
    "Coefficient": model.coef_
})
print(coefficients)


# STEP 10: Predict New House Price
new_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})

predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])


# STEP 11: Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()