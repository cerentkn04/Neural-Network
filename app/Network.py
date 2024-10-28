import numpy as np
import random
from read import load_and_display_mnist


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return (z > 0).astype(float)


class NeuralNetwork:
    def __init__(self, sizes, l2_lambda=0.1, dropout_prob=0.5):
        """sizes is a list containing the number of neurons in each layer."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.l2_lambda = l2_lambda
        self.dropout_prob = dropout_prob
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a, dropout=False):
        """Return the output of the network if 'a' is input."""
        activations = [a]
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            z = np.dot(w, a) + b
            if i < self.num_layers - 2:  # For hidden layers, use ReLU and dropout
                a = relu(z)
                if dropout:
                    mask = (np.random.rand(*a.shape) > self.dropout_prob)
                    a *= mask
                    a /= (1 - self.dropout_prob)  # Scale during training
                activations.append(a)
            else:  # For the output layer, use sigmoid
                a = sigmoid(z)
        return a

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = relu(z) if w.shape[1] != self.sizes[-1] else sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta, n):
        """Update weights and biases with L2 regularization."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1 - eta * self.l2_lambda / n) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the network using mini-batch stochastic gradient descent with L2 regularization."""
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, n)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {j} complete")

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, test_data):
        """Evaluate the number of correct outputs."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = load_and_display_mnist()
    training_data = [(x.reshape(784, 1) / 255.0, vectorized_result(y)) for x, y in zip(train_X, train_y)]
    test_data = [(x.reshape(784, 1) / 255.0, vectorized_result(y)) for x, y in zip(test_X, test_y)]

    network = NeuralNetwork([784, 128, 64, 10], l2_lambda=0.1, dropout_prob=0.5)
    network.SGD(training_data, epochs=30, mini_batch_size=32, eta=0.01, test_data=test_data)
