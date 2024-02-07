from sklearn.model_selection import train_test_split


def preprocess_data(x, y, test_size=0.3, random_state=100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
