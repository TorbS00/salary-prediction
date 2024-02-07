from sklearn.metrics import mean_absolute_percentage_error


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape
