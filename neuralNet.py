import numpy as np

# Set the random seed to 0 for reproducibility and testing purposes
np.random.seed(0)

# Create a spiral dataset to train the neural network on (classification). From https://cs231n.github.io/neural-networks-case-study/
def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights to random values of shape (n_inputs, n_neurons). Multiply by 0.1 (the standard deviation) to reduce the size of the initial weights.
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to zeros of shape (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Take the dot product of inputs and weights and add biases to calculate the output of a forward pass
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        # Calculate the output values from inputs. The ReLU activation function returns 0 for any input value less than 0, and returns the input for any input value greater than or equal to 0.
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def forward(self, inputs):
        # Reduce the input values by subtracting the maximum value to avoid overflow when exponentiating. Then exponentiate the inputs to get the exponential values.
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the values by dividing by the sum of the exponential values. The sum of the output values is 1.
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities


# Create a spiral dataset with 100 points and 3 classes. The size of the dataset is 300 x 2, and the size of the labels is 300 x 1.
X, y = spiral_data(100, 3)

# Create a dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)
# Create a ReLU activation (to be used with the first dense layer):
activation1 = ActivationReLU()

# Create a second dense layer with 3 input features (as we take the output of the first layer here) and 3 output values (for the 3 classes of the spiral dataset)
dense2 = LayerDense(3, 3)
# Create a softmax activation (to be used with the second dense layer):
activation2 = ActivationSoftmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Perform a forward pass through activation function in the first layer
activation1.forward(dense1.output)
# Perform a forward pass through the second dense layer
dense2.forward(activation1.output)
# Perform a forward pass through the softmax activation function in the second layer
activation2.forward(dense2.output)

print(activation2.output[:5])
