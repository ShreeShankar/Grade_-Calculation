# Grade_-Calculation
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
data = {
    'Marks': [75, 85, 60, 90, 70, 50, 80, 95, 65, 55],
    'Grade': [4, 5, 3, 5, 4, 2, 4, 5, 3, 2]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['Marks']]
y = df['Grade']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Marks')
plt.ylabel('Grade')
plt.title('Grade Prediction Based on Marks')
plt.show()

