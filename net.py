from typing import Union
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

def plot(data: ndarray, label: ndarray) -> None:
    """Takes a 2-dimensional array and scatter plots it"""
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="x", ylabel="y")
    for i, point in enumerate(data):
        if label[i] == 0:
            ax.scatter(point[0], point[1], color="skyblue", edgecolors="white")
        else:
            ax.scatter(point[0], point[1], color="gold", edgecolors="white")
    plt.show()

def one_hot_encode(labels: ndarray) -> ndarray:
    """Converts 1-dimensional array into a 2-dimensional one-hot representation"""
    result = []
    for label in labels:
        if label:
            result.append([0, 1])
        else:
            result.append([1, 0])
    return np.array(result)

def softmax(x: Union[ndarray, list]) -> ndarray:
    """Takes an array as input, returns the softmax of each element. Note that the output values always add up to 1, meaning this is a valid probability distribution."""
    result = []

    for instance in x:

        # takes largest value in input vector, subtracts it from the current item in vector, then exponentiates the result
        exp = np.exp(instance - np.max(instance))

        # divides result by the sum of the imput vector, resulting in a probability
        result.append(exp / exp.sum())

    return np.array(result)

def reLU(x: ndarray) -> ndarray:
    """Rectified Linear Unit (ReLU). 
    Sets all elements of the array that are less than 0 to 0, leaves other elements as is.
    This introduces a nonlinearity to the network, allowing nodes to model more complex behaviour."""
    x[x<0] = 0
    return x

def plot_reLU() -> None:
    """Shows a plot of the reLU function"""
    x = np.linspace(-10, 10, 100)
    y = reLU(np.linspace(-10, 10, 100))
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="ReLU")
    ax.plot(x, y, color="skyblue")
    plt.show()

def init_network(n_features: int = 2, n_class: int = 2, n_hidden: int = 64) -> dict[str, ndarray]:
    """Initialises a neural network"""
    model = {
        "W1": np.random.randn(n_features, n_hidden),
        "b1": np.random.randn(n_hidden),
        "W2": np.random.randn(n_features, n_hidden),
        "b2": np.random.randn(n_class)
    }
    return model

def forward(model: dict[str, ndarray], input_data: ndarray) -> tuple[ndarray, ndarray]:
    W1, W2 = model["W1"], model["W2"]
    b1, b2 = model["b1"], model["b2"]

    # input matrix is multipled with the first weights matrix, then the second bias matrix is added
    a1 = input_data @ W1 + b1
    
    # reLU activation is applied to the output of the first X*W+b operation (where X = input matrix, W = weights matrix, b = bias matrix)
    z1 = reLU(a1)

    # input matrix is multipled with the second weights matrix, then the second bias matrix is added
    a2 = z1 @ W2 + b2
    
    # softmax axtivation is applied to the output of the second X*W+b operation
    z2 = softmax(a2)

    return z1, z2

def fit(model: dict[str, ndarray], input_data: ndarray, label: ndarray, batch_size: int, iter_num: int) -> dict[str, ndarray]:
    """Batches the training data. I don't quite understand what is going on with the perumutation function yet."""
    for epoch in range(iter_num):
        p = np.random.permutation(len(label))
        input_data, label = input_data[p], label[p]
        
        for i in range(0, len(label), batch_size):
            batch_data, batch_label = input_data[i:i + batch_size], label[i:i+batch_size]
            model = sgd(model, batch_data, batch_label)

    return model

def sgd(model: dict[str, ndarray], data: ndarray, label: ndarray, alpha: float = 1e-4) -> dict[str, ndarray]:
    """Stochastic gradient descent -  applies necessary changes to weights as advised by backward()"""

    grad = backward(model, data, label)
    for layer in grad.keys():
        model[layer] += alpha * grad[layer]
    return model

def backward(model: dict[str, ndarray], data: ndarray, label: ndarray) -> dict[str, ndarray]:
    """Backward pass - calculates partial derivatives"""
    z1, z2 = forward(model, data)
    label = one_hot_encode(label)
    db2_temp = label - z2
    db2 = np.sum(db2_temp, axis=0)
    dW2 = z1.T @ db2_temp
    db1_temp = db2_temp @ model["W1"].T
    db1_temp[z1 <= 0] = 0
    db1 = np.sum(db1_temp, axis=0)
    dW1 = data.T @ db1_temp
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def test(
    train_data: ndarray, 
    train_label: ndarray, 
    test_data: ndarray, 
    test_label: ndarray, 
    batch_size: int, 
    iter_num: int, 
    n_experiment: int
    ) -> None:

    acc_lst = []

    for k in range(n_experiment):
        model = init_network()
        model = fit(model, train_data, train_label, batch_size=batch_size, iter_num=iter_num)
        _, pred_label = forward(model, test_data)
        pred_label = np.array([np.argmax(pred) for pred in pred_label])
        acc_lst.append((pred_label == test_label).sum() / test_label.size)

    acc_lst = np.array(acc_lst)
    print("Mean accuracy: {0:.5g}, Standard Deviation: {1:.5g}".format(acc_lst.mean(), acc_lst.std()))

test(X_train, y_train, X_test, y_test, 10, 10, 100)
