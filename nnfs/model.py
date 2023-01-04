from neuralNet import (
    LayerDense,
    ActivationReLU,
    ActivationSoftmax_LossCategoricalCrossEntropy,
    OptimizerAdam,
)
from datasets import spiralData
import numpy as np

# Create a spiral dataset with 100 points and 3 classes. The size of the dataset is 300 x 2, and the size of the labels is 300 x 1.
X, y = spiralData(100, 3)

# Create a dense layer with 2 input features and 64 output values, with L1 and L2 regularization values of 5e-4
dense1 = LayerDense(2, 64, weightRegularizerL2=5e-4, biasRegularizerL2=5e-4)
# Create a ReLU activation (to be used with the first dense layer):
activation1 = ActivationReLU()

# Create a second dense layer with 3 input features (as we take the output of the first layer here) and 3 output values (for the 3 classes of the spiral dataset)
dense2 = LayerDense(64, 3)
# Create a combined activation and loss function object
lossActivation = ActivationSoftmax_LossCategoricalCrossEntropy()

# Create an optimizer object
optimizer = OptimizerAdam(learningRate=0.02, decay=5e-7)

for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function in the first layer
    activation1.forward(dense1.output)
    # Perform a forward pass through the second dense layer
    dense2.forward(activation1.output)
    # Perform a forward pass through the combined activation and loss method, then store the calculated data loss
    dataLoss = lossActivation.forward(dense2.output, y)

    # Calculate the regularization loss, if regularization is used
    regularizationLoss = lossActivation.loss.regularizationLoss(
        dense1
    ) + lossActivation.loss.regularizationLoss(dense2)

    # Calculate the overall loss, which is the sum of the data loss and the regularization loss
    loss = dataLoss + regularizationLoss

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
        Data Loss: {dataLoss:.3f}
        Regularization Loss: {regularizationLoss:.3f}
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


# Create and test the model on unseen test data
XTest, yTest = spiralData(100, 3)

dense1.forward(XTest)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = lossActivation.forward(dense2.output, yTest)

predictions = np.argmax(lossActivation.output, axis=1)
if len(yTest.shape) == 2:
    yTest = np.argmax(yTest, axis=1)
accuracy = np.mean(predictions == yTest)

print(f"validation, acc: {accuracy:.3f}, loss: {loss:.3f}")
