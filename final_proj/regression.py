import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load the data
file_path = 'gen2.csv'
data = pd.read_csv(file_path)

# Preparing the data
X = data.drop(['fitness_score'], axis=1)
y = data['fitness_score']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple regression models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Polynomial Regression (Degree 2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Polynomial Regression (Degree 3)': make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
}

results = []
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Append results
    results.append({
        'Model': name,
        'MSE': mse,
        'R^2 Score': r2
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for name in models.keys():
    model = models[name]
    if 'Polynomial' not in name:
        continue  # Skip non-polynomial for simplicity in plotting
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, label=name)

plt.xlabel('True Fitness Score')
plt.ylabel('Predicted Fitness Score')
plt.title('Fitness Score Prediction')
plt.legend()
plt.grid(True)
plt.show()

print(results_df)

# Analyzing Coefficients for Linear Regression
linear_model = models['Linear Regression']
coefficients = pd.DataFrame(linear_model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)