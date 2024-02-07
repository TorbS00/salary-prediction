from src.data_loader import load_dataset
from src.data_preprocessor import preprocess_data
from src.model_trainer import train_linear_model, train_polynomial_model
from src.model_evaluator import evaluate_model
from src.result_printer import print_results


# Load the dataset
dataset_path = '../data/raw/data.csv'
df = load_dataset(dataset_path)


# Prepare the data
X = df[['rating']]
y = df['salary']
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Train linear regression model
linear_model = train_linear_model(X_train, y_train)

# Evaluate linear model
linear_mape = evaluate_model(linear_model, X_test, y_test)

# Print linear model results
print("Linear Regression Model:")
print_results(linear_model.intercept_, linear_model.coef_[0], linear_mape)

# Train polynomial regression models for degrees 2, 3, and 4
best_mape, best_degree = float('inf'), None
for degree in range(2, 5):
    polynomial_model = train_polynomial_model(X_train, y_train, degree)
    polynomial_mape = evaluate_model(polynomial_model, X_test, y_test)
    if polynomial_mape < best_mape:
        best_mape = polynomial_mape
        best_degree = degree

# Print best polynomial model results
print(f"Best Polynomial Regression Model (Degree {best_degree}):")
print_results(None, None, best_mape)