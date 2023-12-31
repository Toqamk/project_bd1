#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.stats import zscore
df = pd.read_csv("ford.csv")


# In[2]:


# Import the necessary libraries for modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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


decision_tree_reg = DecisionTreeRegressor(random_state=42)


# In[6]:


# Decision Tree Regression
decision_tree_reg.fit(X_train, y_train)
decision_tree_pred = decision_tree_reg.predict(X_test)
decision_tree_mae = mean_absolute_error(y_test, decision_tree_pred)
decision_tree_rmse = mean_squared_error(y_test, decision_tree_pred, squared=False)
decision_tree_r2 = r2_score(y_test, decision_tree_pred)


# In[7]:


print("Decision Tree Regression:")
print("MAE:", decision_tree_mae)
print("RMSE:", decision_tree_rmse)
print("R^2 Score:", decision_tree_r2)



# In[ ]:
# Save the results to a text file
with open("decisiontree.txt", 'w') as file:
    file.write("Decision Tree Regression:\n")
    file.write(f"MAE: {decision_tree_mae}\n")
    file.write(f"RMSE: {decision_tree_rmse}\n")
    file.write(f"R^2 Score: {decision_tree_r2}\n")




