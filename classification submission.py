#!/usr/bin/env python
# coding: utf-8

# ### CLASSIFICATION SUBMISSION

# In[1]:


#import necessary libraries
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Data_for_UCI_named.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df = df.drop("stab", axis=1)


# ### Splitting the data

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Split the dataset
X = df.drop('stabf', axis=1)
y = df['stabf']

# Perform train-test split with a random state of 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---
# 
# 
# ### Question 14
# What is the accuracy on the test set using the random forest classifier? In 4 decimal places.

# In[6]:


# Train Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train_scaled, y_train)
rf_predictions = rf_classifier.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Validation Accuracy:", round(rf_accuracy, 4))


# ---
# 
# 
# ### Question 15
# What is the accuracy on the test set using the XGboost classifier? In 4 decimal places.

# In[7]:


# Convert class labels to binary values
y_train_binary = y_train.replace({'stable': 0, 'unstable': 1})

# Train XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=1)
xgb_classifier.fit(X_train, y_train_binary)
y_pred = xgb_classifier.predict(X_test)


# In[8]:


# Convert predicted labels to string format
y_pred_str = np.where(y_pred == 0, 'stable', 'unstable')

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred_str)
print("Test Accuracy:", round(accuracy, 4))


# ---
# 
# 
# ### Question 16
# 
# What is the accuracy on the test set using the LGBM classifier? In 4 decimal places.
# 

# In[9]:


# Train LightGBM classifier
lgb_classifier = lgb.LGBMClassifier(random_state=1)
lgb_classifier.fit(X_train_scaled, y_train)
lgb_predictions = lgb_classifier.predict(X_test_scaled)
lgb_accuracy = accuracy_score(y_test, lgb_predictions)

print(f"Accuracy on the test set: {lgb_accuracy:.4f}")


# ---
# 
# 
# ### Question 17
# Using the ExtraTreesClassifier as your estimator with cv=5, n_iter=10, scoring = 'accuracy', n_jobs = -1, verbose = 1 and random_state = 1. What are the best hyperparameters from the randomized search CV?
# 

# In[10]:


#import necessary libraries
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV

# Create the ExtraTreesClassifier
estimator = ExtraTreesClassifier(random_state=1)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator, param_distributions=param_grid, cv=5, n_iter=10, scoring='accuracy', n_jobs=-1, verbose=1, random_state=1)

# Fit the RandomizedSearchCV object to the data
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")


# ---
# 
# 
# ### Question 18
# 
# Train a new ExtraTreesClassifier Model with the new Hyperparameters from the RandomizedSearchCV (with random_state = 1). Is the accuracy of the new optimal model higher or lower than the initial ExtraTreesClassifier model with no hyperparameter tuning?

# In[11]:


#import necessary libraries
from sklearn.ensemble import ExtraTreesClassifier

# Create a new ExtraTreesClassifier model with optimal hyperparameters
new_model = ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=1)

# Train the new model on the training data
new_model.fit(X_train, y_train)

# Evaluate the accuracy of the new model on the test data
new_accuracy = new_model.score(X_test, y_test)

# Evaluate the accuracy of the initial model on the test data
initial_accuracy = lgb_classifier.score(X_test, y_test)

# Compare the accuracies
if new_accuracy > initial_accuracy:
    print("The accuracy of the new optimal model is higher than the initial model.")
elif new_accuracy < initial_accuracy:
    print("The accuracy of the new optimal model is lower than the initial model.")
else:
    print("The accuracy of the new optimal model is the same as the initial model.")


# ---
# 
# 
# ### Question 20
# Find the feature importance using the optimal ExtraTreesClassifier model. Which features are the most and least important respectively?
