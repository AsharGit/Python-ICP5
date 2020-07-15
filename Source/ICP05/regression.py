import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

train = pd.read_csv('winequality-red.csv')

# Handling missing and null values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

# Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

# Using absolute value we print the 3 most correlated features to quality
corr = numeric_features.corr().abs()
print(corr['quality'].sort_values(ascending=False)[1:4], '\n')

# Target value quality
y = data['quality']
X = data.drop(['quality'], axis=1)

# Training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=.33)


# Build a linear model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance and visualize results
print("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)


print('RMSE is: \n', mean_squared_error(y_test, predictions))

# visualize
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b')  # alpha helps to show overlapping data
plt.xlabel('Wine features')
plt.ylabel('Quality')
plt.title('Linear Regression Model')
plt.show()
