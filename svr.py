#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from scipy.stats import zscore
df = pd.read_csv("ford.csv")




# Import the necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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





svr = SVR()





# Support Vector Regression
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
svr_mae = mean_absolute_error(y_test, svr_pred)
svr_rmse = mean_squared_error(y_test, svr_pred, squared=False)
svr_r2 = r2_score(y_test, svr_pred)



print("Support Vector Regression:")
print("MAE:", svr_mae)
print("RMSE:", svr_rmse)
print("R^2 Score:", svr_r2)


 #Save the results to a text file
with open("svr.txt", 'w') as file:

    file.write("Support Vector Regression:\n")
    file.write(f"MAE: {svr_mae}\n")
    file.write(f"RMSE: {svr_rmse}\n")
    file.write(f"R^2 Score: {svr_r2}\n")






