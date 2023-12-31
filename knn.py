#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from scipy.stats import zscore
df = pd.read_csv("ford.csv")


# In[8]:


# Import the necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor



# In[9]:


# Based on the correlation matrix, we can select relevant features for our model
selected_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

# Create the feature matrix X and the target variable y
X = df[selected_features]
y = df['price']

# Perform feature scaling using z-score normalization
X_scaled = X.apply(zscore)


# In[10]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[11]:


knn_reg = KNeighborsRegressor()


# In[12]:


# K-Nearest Neighbors Regression
knn_reg.fit(X_train, y_train)
knn_pred = knn_reg.predict(X_test)
knn_mae = mean_absolute_error(y_test, knn_pred)
knn_rmse = mean_squared_error(y_test, knn_pred, squared=False)
knn_r2 = r2_score(y_test, knn_pred)


# In[13]:


print("K-Nearest Neighbors Regression:")
print("MAE:", knn_mae)
print("RMSE:", knn_rmse)
print("R^2 Score:", knn_r2)


# In[ ]:
# Save the results to a text file
with open("knn.txt", 'w') as file:

    file.write("K-Nearest Neighbors Regression:\n")
    file.write(f"MAE: {knn_mae}\n")
    file.write(f"RMSE: {knn_rmse}\n")
    file.write(f"R^2 Score: {knn_r2}\n")




