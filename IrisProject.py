
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:11:07 2021

@author: Sinem
"""
#We will give 3 of 4 different sizes of the flower and find the 4th dimension. (I want to find the sepal length)
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import numpy as np

iris = load_iris()
print(iris.feature_names)

print(iris.data[:10])
#features in the iris dataset
feature_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
print(feature_df.head())
#statistical description of our data (we transposed it with .T to make it look nicer)
print(feature_df.describe().T)
#features info
print(feature_df.info())
#We can look at the correlation between the attributes.
sns.pairplot(feature_df)

#training model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


#feature data frame output
print(feature_df.head())


X = feature_df.select_dtypes("float64").drop("sepal length (cm)", axis=1)
y= feature_df["sepal length (cm)"]
#controls
print(X.head())

print(y.head())

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state =7)
#linear models
linear_model = LinearRegression()
ridge_model = Ridge() #L2 
lasso_model = Lasso() #L1 
# training
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
#performance
print(linear_model.score(X_train, y_train))
print(ridge_model.score(X_train, y_train))
print(lasso_model.score(X_train, y_train))
#comparison
lin_pred = linear_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)
#dictionary
pred_dict = {"Linear": lin_pred, "Ridge":ridge_pred, "Lasso": lasso_pred}
#predictions
print(pred_dict)
for key, value in pred_dict.items():
  print("Model:", key)
  print("R2 Score:", r2_score(y_test, value))
  print('Mean Absolute Error:', mean_absolute_error(y_test, value))
  print('Mean Squared Error:', mean_squared_error(y_test, value))
  print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, value)))
  print()
#custom prediction(predict sepal length)
value_to_predict = [[3.4, 1.2, 0.3]]
lin_pred = linear_model.predict(value_to_predict)
ridge_pred = ridge_model.predict(value_to_predict)
lasso_pred = lasso_model.predict(value_to_predict)

pred_dict = {"Linear": lin_pred, "Ridge":ridge_pred, "Lasso": lasso_pred}
print(pred_dict)

#We take the estimation according to whichever is better in terms of success performance.

