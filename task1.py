# 🏠 House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv('C:/Users/BHAKTI/OneDrive/Desktop/ML Internship/house-prices-advanced-regression-techniques/train.csv')  # Make sure 'train.csv' is in the same directory
print("Data Loaded Successfully")

# Preview Data
print(data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].head())

# Select Features and Target
features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = data['SalePrice']

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# Model Coefficients
coefficients = pd.DataFrame(model.coef_, features.columns, columns=["Coefficient"])
print("\nModel Coefficients:")
print(coefficients)
# 🏠 House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')  # Updated path
print("Data Loaded Successfully")

# Preview Data
print(data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].head())

# Select Features and Target
features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = data['SalePrice']

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# Model Coefficients
coefficients = pd.DataFrame(model.coef_, features.columns, columns=["Coefficient"])
print("\nModel Coefficients:")
print(coefficients)
