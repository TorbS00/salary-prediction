from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def train_linear_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_polynomial_model(x_train, y_train, degree):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train, y_train)
    return model
