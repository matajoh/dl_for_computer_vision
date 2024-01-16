import matplotlib.pyplot as plt
import numpy as np


def compare(func0, func1):
    x_values = np.arange(-50, 50) / 10.0
    y0_values = func0(x_values)
    y1_values = func1(x_values)
    plt.ylim([-1.1, 1.1])
    plt.plot(x_values, y0_values, x_values, y1_values)


def perceptron(x):
    return np.sign(x)


def tanh(x):
    return np.tanh(x)


def dtanh(y, dy):
    return (1 - y**2) * dy


def relu(x):
    return np.where(x > 0, x, 0)


def main():
    plt.rc('font', size=15)
    plt.figure(figsize=(4, 4))
    compare(perceptron, tanh)
    plt.show()

    plt.figure(figsize=(4, 4))
    x_values = np.arange(-50, 50) / 10.0
    plt.ylim(-5, 5)
    plt.plot(x_values, relu(x_values))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
