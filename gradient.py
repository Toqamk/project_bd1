#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.stats import zscore
df = pd.read_csv("ford.csv")


# In[2]:


# Import the necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[3]:


# Based on the correlation matrix, we can select relevant features for our model
selected_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

# Create the feature matrix X and the target variable y
X = df[selected_features]
y = df['price']

# Perform feature scaling using z-score normalization
X_scaled = X.apply(zscore)


# In[4]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[5]:


gradient_boosting_reg = GradientBoostingRegressor(random_state=42)


# In[6]:


gradient_boosting_reg = GradientBoostingRegressor()
gradient_boosting_reg.fit(X_train, y_train)

# Make predictions
gradient_boosting_pred = gradient_boosting_reg.predict(X_test)

# Calculate metrics
gradient_boosting_mae = mean_absolute_error(y_test, gradient_boosting_pred)
gradient_boosting_rmse = mean_squared_error(y_test, gradient_boosting_pred, squared=False)
gradient_boosting_r2 = r2_score(y_test, gradient_boosting_pred)


# In[7]:


# Print the metrics
print("Gradient Boosting Regression:")
print("MAE:", gradient_boosting_mae)
print("RMSE:", gradient_boosting_rmse)
print("R^2 Score:", gradient_boosting_r2)


# In[ ]:
# Save the results to a text file
with open("gradient.txt", 'w') as file:

    file.write("Gradient Boosting Regression:\n")
    file.write(f"MAE: {gradient_boosting_mae}\n")
    file.write(f"RMSE: {gradient_boosting_rmse}\n")
    file.write(f"R^2 Score: {gradient_boosting_r2}\n")




