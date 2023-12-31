#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.stats import zscore
df = pd.read_csv("ford.csv")


# In[2]:


# Import the necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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


linear_reg = LinearRegression()


# In[6]:


# Linear Regression
linear_reg.fit(X_train, y_train)
linear_reg_pred = linear_reg.predict(X_test)
linear_reg_mae = mean_absolute_error(y_test, linear_reg_pred)
linear_reg_rmse = mean_squared_error(y_test, linear_reg_pred, squared=False)
linear_reg_r2 = r2_score(y_test, linear_reg_pred)


# In[7]:


print("Linear Regression:")
print("MAE:", linear_reg_mae)
print("RMSE:", linear_reg_rmse)
print("R^2 Score:", linear_reg_r2)
print()


# In[ ]:
# Save the results to a text file
with open("linear.txt", 'w') as file:

    file.write("Linear Regression:\n")
    file.write(f"MAE: {linear_reg_mae}\n")
    file.write(f"RMSE: {linear_reg_rmse}\n")
    file.write(f"R^2 Score: {linear_reg_r2}\n")





