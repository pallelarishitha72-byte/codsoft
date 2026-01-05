import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("Advertising.csv")

print ("Preview of the dataset:")
print (data.head());
print ("-" * 40)
X = data[['TV']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print ("Training data size:", X_train.shape)
print ("Testing data size:", X_test.shape)
print ("-" * 40)
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed.")
print ("-" * 40)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("-" * 40)
plt.plot(X_test, y_pred, color='red', label='Predicted Sales')

plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("TV Advertising Budget vs Sales")
plt.legend()
plt.show()