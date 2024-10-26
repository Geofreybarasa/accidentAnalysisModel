import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data1.csv')

# Display dataset overview
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
if df.isnull().values.any():
    print("Warning: The dataset contains NaN values. Please handle them before proceeding.")

# Encode categorical variables
df['weather_conditions'] = df['weather_conditions'].astype('category').cat.codes

# Prepare features and target variable
X = df[['vehicles_involved', 'weather_conditions']]
y = df['severity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Make predictions on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Example prediction
example_data = np.array([[3, 1]])  # Example: 3 vehicles, clear weather (encoded as 1)
predicted_severity = model.predict(example_data)
print(f'Predicted Accident Severity: {predicted_severity[0]:.2f}')

# Generate predictions for visualization
vehicles_range = np.arange(1, 10)
weather_conditions_range = np.arange(0, 3)

predictions = [(vehicles, weather, model.predict([[vehicles, weather]])[0])
               for vehicles in vehicles_range for weather in weather_conditions_range]

predictions_df = pd.DataFrame(predictions, columns=['vehicles_involved', 'weather_conditions', 'predicted_severity'])

# Create subplots for visualization
plt.figure(figsize=(18, 12))

# Plot 1: Actual vs Predicted
plt.subplot(2, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
plt.xlabel('Actual Accident Severity', fontsize=14)
plt.ylabel('Predicted Accident Severity', fontsize=14)
plt.title('Actual vs Predicted Accident Severity', fontsize=16)
sns.regplot(x=y_test, y=y_pred, scatter=False, color='red', line_kws={'label': 'Line of Best Fit'})
plt.legend()

# Plot 2: Residuals Distribution
plt.subplot(2, 2, 2)
sns.histplot(y_test - y_pred, kde=True, color='purple', bins=20)
plt.xlabel('Residuals', fontsize=14)
plt.title('Distribution of Residuals', fontsize=16)

# Plot 3: Residuals vs Predicted
plt.subplot(2, 2, 3)
sns.scatterplot(x=y_pred, y=y_test - y_pred, color='orange', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Accident Severity', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Predicted Accident Severity', fontsize=16)

# Plot 4: Multiple Predictions
plt.subplot(2, 2, 4)
sns.scatterplot(data=predictions_df, x='vehicles_involved', y='predicted_severity', hue='weather_conditions', 
                palette='viridis', alpha=0.8, edgecolor='w', s=100)
plt.xlabel('Number of Vehicles Involved', fontsize=14)
plt.ylabel('Predicted Accident Severity', fontsize=14)
plt.title('Predicted Accident Severity for Various Conditions', fontsize=16)
plt.legend(title='Weather Conditions', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# Display model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Example usage of the prediction function
print(f'Predicted Accident Severity for 3 vehicles and clear weather: {model.predict([[3, 1]])[0]:.2f}')

# Create a pie chart for performance metrics
fig, ax = plt.subplots(figsize=(8, 8))
metrics = ['Mean Squared Error', 'R^2 Score', 'Average Predicted Severity']
values = [mse, r2, np.mean(predictions_df['predicted_severity'])]

# Normalize values for pie chart
total = sum(values)
values = [v / total for v in values] if total > 0 else values  # Prevent division by zero

ax.pie(values, labels=metrics, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Performance Metrics Distribution', fontsize=16)
plt.show()
