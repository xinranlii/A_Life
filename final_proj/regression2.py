import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

file_path = 'gen4.csv'
data = pd.read_csv(file_path)

X = data[['new_upper_mass', 'new_lower_mass', 'new_upper_gear', 'new_lower_gear', 'new_upper_joint_range_min', 'new_lower_joint_range_min', 'random_num_upper_legs', 'random_num_lower_legs']]
y = data['fitness_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple regression models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MSE': mse,
        'R^2 Score': r2
    })

results_df = pd.DataFrame(results)

# Plotting
# fig, ax = plt.subplots(figsize=(10, 6))
# for name in models.keys():
#     model = models[name]
#     if 'Polynomial' not in name:
#         continue  # Skip non-polynomial for simplicity in plotting
#     y_pred = model.predict(X_test)
#     plt.scatter(y_test, y_pred, label=name)

# plt.xlabel('True Fitness Score')
# plt.ylabel('Predicted Fitness Score')
# plt.title('Fitness Score Prediction')
# plt.legend()
# plt.grid(True)
# plt.show()

print(results_df)


linear_model = models['Linear Regression']
coefficients = pd.DataFrame(linear_model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

predictor_vars = ['new_upper_mass', 'new_lower_mass', 'new_upper_gear', 'new_lower_gear', 'new_upper_joint_range_min', 'new_lower_joint_range_min', 'random_num_upper_legs', 'random_num_lower_legs']

for var in predictor_vars:
    X = data[[var]]
    y = data['fitness_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
    plt.title(f'{var} vs. Fitness Score')
    plt.xlabel(var)
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.show()