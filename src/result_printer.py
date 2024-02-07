def print_results(intercept, slope, mape):
    if intercept is not None and slope is not None:
        print(f"Intercept: {intercept:.5f}")
        print(f"Slope: {slope:.5f}")
    else:
        print("Model coefficients:")
        print("Not applicable for polynomial regression.")

    print(f"MAPE: {mape:.5f}")