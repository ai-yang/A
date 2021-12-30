import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
from planar_utils import plot_decision_boundary, sigmoid, load_extra_datasets


def load_planar_dataset():
    np.random.seed(1)
    m = 400
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype="uint8")
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    return X, Y


def layer_size(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_y


def initialize_param(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    par = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return par


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]
    temp = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(temp) / m
    return cost


def back_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dz2 = A2 - Y
    dw2 = np.dot(dz2, A1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = np.multiply(np.dot(W2.T, dz2), 1 - np.power(A1, 2))
    dw1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    grads = {"dW1": dw1,
             "db1": db1,
             "dW2": dw2,
             "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=0.025):
    w1 = parameters["W1"]
    b1 = parameters["b1"]
    w2 = parameters["W2"]
    b2 = parameters["b2"]
    dw1 = grads["dW1"]
    db1 = grads["db1"]
    dw2 = grads["dW2"]
    db2 = grads["db2"]
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    parameters = {"W1": w1,
                  "b1": b1,
                  "W2": w2,
                  "b2": b2}
    return parameters


def nn_model(X, Y, n_h, iteration=5000):
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[1]
    parameters = initialize_param(n_x, n_h, n_y)
    w1 = parameters['W1']
    w2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    for i in range(0, iteration):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = back_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if i % 100 == 0:
            print("Cost after %i iteration: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    prediction = (A2 > 0.5)
    return prediction


# X: training data  Y: color(1 or 0) corresponding to each data
X, Y = load_planar_dataset()
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, iteration=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()
