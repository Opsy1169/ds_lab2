import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def generated_data():
    mean = 0
    std = np.sqrt(0.01)
    eps = np.random.normal(mean, std, 100)
    x = np.linspace(0, 1, 100)
    a = 2
    b = 0.1
    y = calc_y(x, a, b, eps)
    perform_regression_and_plot(x, y)


def test_data():
    x, y = read_data_from_file()
    perform_regression_and_plot(x, y)


def perform_regression_and_plot(x, y):
    reg = perform_regression(x, y)
    x_line = [np.min(x), np.max(x)]
    y_line = calc_y(x_line, reg.coef_, reg.intercept_)
    plot_data(x, y, x_line, y_line)


def read_data_from_file():
    file = open("01-soybean-data.txt")
    protein = []
    oil = []
    for line in file:
        splitted_line = line.split()
        protein.append(float(splitted_line[9]))
        oil.append(float(splitted_line[10]))
    return np.array(protein), np.array(oil)


def plot_data(x, y, x_line, y_line):
    plt.plot(x, y, 'x')
    plt.plot(x_line, y_line)
    plt.axis('equal')
    plt.show()


def perform_regression(x, y):
    x_train = np.array(list(map(lambda x1: np.array([x1]), x))).reshape(-1, 1)
    x_train.reshape(-1, 1)
    reg = LinearRegression().fit(x_train, y)
    print('Determination coefficient: ', reg.score(x_train, y))
    print('a: ', reg.coef_)
    print('b: ', reg.intercept_)
    return reg


def calc_y(x, a, b, eps=None):
    y = []
    eps = np.zeros(len(x)) if eps is None else eps
    for i in range(len(x)):
        y.append(a * x[i] + b + eps[i])
    return np.array(y)


if __name__ == '__main__':
    # generated_data()
    test_data()
