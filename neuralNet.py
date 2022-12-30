import numpy as np
import nnfs

# Initialize the nnfs library for reproducibility and testing purposes
nnfs.init()

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
        # Initialize weights to random values of shape (n_inputs, n_neurons). Multiply by 0.01 (the standard deviation) to reduce the size of the initial weights.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to zeros of shape (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Keep track of the inputs for backpropagation
        self.inputs = inputs
        # Take the dot product of inputs and weights and add biases to calculate the output of a forward pass
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # The gradient of the loss with respect to the weights is the dot product of the inputs and the gradient of the loss with respect to the
        # outputs of the current layer.
        # The gradient of the loss with respect to the outputs of the current layer is the dvalues argument that is passed into the backward method.
        self.dweights = np.dot(self.inputs.T, dvalues)
        # The gradient of the loss with respect to the biases is the sum of the gradient of the loss with respect to the outputs of the current layer.
        # This is because the partial derivative of the loss with respect to the bias is 1.
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # The gradient of the loss with respect to the inputs is the dot product of the gradient of the loss with respect to the outputs of the
        # current layer and the transposed weights.
        # This is similar to the gradient of the loss with respect to the weights, but the order of the inputs and weights is reversed.
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    def forward(self, inputs):
        # Keep track of the inputs for backpropagation
        self.inputs = inputs
        # Calculate the output values from inputs. The ReLU activation function returns 0 for any input value less than 0, and returns the
        # input for any input value greater than or equal to 0.
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Create a copy of the dvalues array. This is because we will be modifying the values in the array.
        self.dinputs = dvalues.copy()
        # Set all of the negative values in the array to 0. This is because the ReLU activation function returns 0 for any input value less than 0.
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    def forward(self, inputs):
        # Keep track of the inputs for backpropagation
        self.inputs = inputs
        # Reduce the input values by subtracting the maximum value to avoid overflow when exponentiating. Then exponentiate the inputs to get the exponential values.
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the values by dividing by the sum of the exponential values. The sum of the output values is 1.
        probabilities = expValues / np.sum(expValues, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create an uninitialized array of the same shape as the gradient of the loss with respect to the outputs of the current layer.
        self.dinputs = np.empty_like(dvalues)

        for index, (singleOutput, singleDvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # Flatten the output array
            singleOutput = singleOutput.reshape(-1, 1)
            # Calculate the Jacobian matrix of the output. The Jacobian matrix is a matrix containing the sample-wise partial derivatives of the output values.
            # Create a diagonal matrix from the output array with values along the diagonal equal to the output values.
            # Then multiply the softmax outputs. Take the dot product of the output array and the transpose of the output array since the output array is a row vector.
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(
                singleOutput, singleOutput.T
            )
            # Calculate the sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobianMatrix, singleDvalues)


class Loss:
    def calculate(self, output, y):
        # Calculate the sample losses
        sampleLosses = self.forward(output, y)
        # Calculate the mean loss for the batch to use as the loss value for backpropagation
        dataLoss = np.mean(sampleLosses)
        return dataLoss


class LossCategoricalCrossEntropy(Loss):
    # Define the clip value to use when calculating the loss. This is used to avoid taking the log of 0, which would result in infinity.
    CLIP_VALUE = 1e-7

    def forward(self, yPred, yTrue):
        samples = len(yPred)
        # Clip the data to prevent taking the log of 0. Clip both sides to not drag mean towards any value.
        yPredClipped = np.clip(
            yPred,
            LossCategoricalCrossEntropy.CLIP_VALUE,
            1 - LossCategoricalCrossEntropy.CLIP_VALUE,
        )

        # Probabilities for target values if categorical labels
        if len(yTrue.shape) == 1:
            correctConfidences = yPredClipped[range(samples), yTrue]
        # Probabilities for target values if one-hot encoded labels
        elif len(yTrue.shape) == 2:
            correctConfidences = np.sum(yPredClipped * yTrue, axis=1)

        # Calculate the negative log likelihood for the loss
        negativeLogLikelihoods = -np.log(correctConfidences)
        return negativeLogLikelihoods

    def backward(self, dvalues, yTrue):
        # The number of samples
        samples = len(dvalues)
        # The number of labels in each sample
        labels = len(dvalues[0])

        # Convert categorical labels to one-hot encoded labels.
        if len(yTrue.shape) == 1:
            # Create an identity matrix of shape (labels, labels), and pass in the yTrue array to convert it into a
            # one-hot encoded array (this will shift the diagonal ones according to the indexes from the categorical labels).
            yTrue = np.eye(labels)[yTrue]

        # The gradient of the categorical cross-entropy loss with respect to the output of the softmax activation function ends up being: -yTrue / dvalues
        self.dinputs = -yTrue / dvalues
        # Normalize the gradient. They will eventually be summed
        self.dinputs /= samples


class ActivationSoftmax_LossCategoricalCrossEntropy:
    def __init__(self):
        # Create an activation property equal to a softmax activation function
        self.activation = ActivationSoftmax()
        # Create a loss property equal to a categorical cross-entropy loss function
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, yTrue):
        # Call the forward method of the softmax activation function
        self.activation.forward(inputs)
        # Store the output of the softmax activation function
        self.output = self.activation.output
        # Return the loss value using the output of the softmax activation function and the true labels
        return self.loss.calculate(self.output, yTrue)

    def backward(self, dvalues, yTrue):
        # The number of samples
        samples = len(dvalues)

        # Check to see if the labels are one-hot encoded
        if len(yTrue.shape) == 2:
            # If so, convert them into discrete values by taking the index of the largest value in each row
            yTrue = np.argmax(yTrue, axis=1)

        # Copy the gradient from the next layer
        self.dinputs = dvalues.copy()
        # Calculate the gradient by selecting the values from dinputs that correspond to the true labels and subtracting 1 from them.
        # This will result in the gradient being 0 for the correct label and -1 for the incorrect labels.
        self.dinputs[range(samples), yTrue] -= 1
        # Normalize the gradient. This will eventually be summed
        self.dinputs /= samples


class OptimizerSGD:
    def __init__(self, learningRate=1.0, decay=0.0, momentum=0.0):
        # Set the initial learning rate. This will be used as a reference, and won't be updated during training.
        self.learningRate = learningRate
        # The self.currentLearningRate property will be used to update the learning rate during training
        self.currentLearningRate = learningRate
        # Set the decay rate, which will determine how much the learning rate will be reduced during each update
        self.decay = decay
        # Keep track of the number of parameter updates that have been done
        self.iterations = 0
        # Set the momentum value. A higher value weights the previous update more
        self.momentum = momentum

    def preUpdateParams(self):
        # If decay isn't 0, then reduce the learning rate by multiplying it by
        # 1 / (1 + decay * iterations), which decreases over time
        if self.decay:
            self.currentLearningRate = self.learningRate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def updateParams(self, layer):
        # If momentum is not 0, and is being used
        if self.momentum:
            # Check to see if the layer already momentum properties for the weights
            if not hasattr(layer, "weightMomentums"):
                # If not, create it and initialize it with zeros
                layer.weightMomentums = np.zeros_like(layer.weights)
                # This is also done for the bias momentums since they wouldn't be created without the weight momentums
                layer.biasMomentums = np.zeros_like(layer.biases)

            # Calculate the value to update the weights by. This is done by multiplying the momentum value by the previous weight
            # momentums, and subtracting the current gradient multiplied by the learning rate. A higher value will allow the previous
            # weight update to have a larger affect on the current one (shifts it)
            weightUpdates = (
                self.momentum * layer.weightMomentums
                - self.currentLearningRate * layer.dweights
            )
            # Set the layer's weight momentums to the weight updates for the next iteration
            layer.weightMomentums = weightUpdates

            # Perform the same process for the bias updates as was done for the weight updates
            biasUpdates = (
                self.momentum * layer.biasMomentums
                - self.currentLearningRate * layer.dbiases
            )
            layer.biasMomentums = biasUpdates
        # If momentum is not being used
        else:
            # Update the layer's weights and biases by multiplying the learning rate by the gradients of the weights and biases
            weightUpdates = -self.currentLearningRate * layer.dweights
            biasUpdates = -self.currentLearningRate * layer.dbiases

        # Update the layer's weights and biases using weightUpdates and biasUpdates respectively
        layer.weights += weightUpdates
        layer.biases += biasUpdates

    def postUpdateParams(self):
        # Increment the number of iterations after updating the parameters
        self.iterations += 1


# Create a spiral dataset with 100 points and 3 classes. The size of the dataset is 300 x 2, and the size of the labels is 300 x 1.
X, y = spiral_data(100, 3)

# Create a dense layer with 2 input features and 64 output values
dense1 = LayerDense(2, 64)
# Create a ReLU activation (to be used with the first dense layer):
activation1 = ActivationReLU()

# Create a second dense layer with 3 input features (as we take the output of the first layer here) and 3 output values (for the 3 classes of the spiral dataset)
dense2 = LayerDense(64, 3)
# Create a combined activation and loss function object
lossActivation = ActivationSoftmax_LossCategoricalCrossEntropy()

# Create an optimizer object
optimizer = OptimizerSGD(decay=1e-3, momentum=0.9)

for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function in the first layer
    activation1.forward(dense1.output)
    # Perform a forward pass through the second dense layer
    dense2.forward(activation1.output)
    # Perform a forward pass through the combined activation and loss method, then store the calculated loss
    loss = lossActivation.forward(dense2.output, y)

    # Calculate the model's accuracy
    predictions = np.argmax(lossActivation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print the epoch number, accuracy, and loss every 100 epochs
    if not epoch % 100:
        print(
            f"""
        Epoch: {epoch}
        Accuracy: {accuracy:.3f}
        Loss: {loss:.3f}
        Learning Rate: {optimizer.currentLearningRate}
        """
        )

    # Perform a backward pass through the combined activation and loss method
    lossActivation.backward(lossActivation.output, y)
    # Perform a backward pass through the rest of the dense layers and activation functions
    dense2.backward(lossActivation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update the weights and biases using the optimizer
    optimizer.preUpdateParams()
    optimizer.updateParams(dense1)
    optimizer.updateParams(dense2)
    optimizer.postUpdateParams()
