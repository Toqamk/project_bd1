#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from scipy.stats import zscore
df = pd.read_csv("ford.csv")



# Import the necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# Based on the correlation matrix, we can select relevant features for our model
selected_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

# Create the feature matrix X and the target variable y
X = df[selected_features]
y = df['price']

# Perform feature scaling using z-score normalization
X_scaled = X.apply(zscore)




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)





random_forest_reg = RandomForestRegressor(random_state=42)





# Random Forest Regression
random_forest_reg.fit(X_train, y_train)
random_forest_pred = random_forest_reg.predict(X_test)
random_forest_mae = mean_absolute_error(y_test, random_forest_pred)
random_forest_rmse = mean_squared_error(y_test, random_forest_pred, squared=False)
random_forest_r2 = r2_score(y_test, random_forest_pred)





print("Random Forest Regression:")
print("MAE:", random_forest_mae)
print("RMSE:", random_forest_rmse)
print("R^2 Score:", random_forest_r2)




# Save the results to a text file
with open("randomforest.txt", 'w') as file:

    file.write("Random Forest Regression:\n")
    file.write(f"MAE: {random_forest_mae}\n")
    file.write(f"RMSE: {random_forest_rmse}\n")
    file.write(f"R^2 Score: { random_forest_r2}\n")




