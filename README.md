Neural Network Mini-Library and House Price Prediction

Overview

This project consists of two parts:

1. Neural Network Mini-Library (part1_nn_lib.py): A basic implementation of a multi-layer neural network, with backpropagation, data preprocessing, training, and evaluation.
2. House Price Prediction (part2_house_value_regression.py): A neural network using PyTorch (or the mini-library) for predicting house prices in California.


Part 1: Neural Network Mini-Library (part1_nn_lib.py)

Description

The mini-library features implementations of neural network components like various layers (linear, sigmoid, ReLU), loss functions (MSE, Cross-Entropy), and a trainer for managing shuffling, batching, and parameter updates.

Running the Code

Executing part1_nn_lib.py demonstrates the following:

- Creation of linear and non-linear layers, like SigmoidLayer and ReluLayer.
- Building and testing a MultiLayerNetwork with specified dimensions, neuron counts, and activation functions.
- Training the network with a Trainer instance, demonstrating shuffling, batch processing, and evaluating validation loss.
- Applying data preprocessing using the Preprocessor class.

Example Usage

# Example of creating a neural network with specific input, hidden, and output layers.
# This network has 4 input features, 16 neurons in the first hidden layer, and 2 output neurons.
# It uses ReLU activation for the hidden layer and a sigmoid activation for the output layer.
network = MultiLayerNetwork(input_dim=4, neurons=[16, 2], activations=["relu", "sigmoid"])

# Performing a forward pass through the network using example inputs.
outputs = network(inputs)

# Executing a backward pass through the network for gradient computation.
grad_loss_wrt_inputs = network.backward(grad_loss_wrt_outputs)

# Updating network parameters based on the learning rate and computed gradients.
network.update_params(learning_rate)

# Training the neural network using the Trainer class.
# The Trainer manages batch processing, shuffling, loss computation, and parameter updates.
# Here, it's set up for mean squared error loss and a batch size of 32.
trainer = Trainer(network=network, batch_size=32, nb_epoch=10, learning_rate=1.0e-3, shuffle_flag=True, loss_fun="mse")
trainer.train(train_inputs, train_targets)

# Evaluating the trained network's performance on validation data.
print("Validation loss = ", trainer.eval_loss(val_inputs, val_targets))

# Utilizing the Preprocessor class for data normalization.
# This preprocessor applies min-max scaling to normalize dataset features to a [0, 1] range.
prep = Preprocessor(dataset)
normalized_dataset = prep.apply(dataset)  # Apply min-max normalization
original_dataset = prep.revert(normalized_dataset)  # Reverting normalization



Part 2: California House Prices Prediction (part2_house_value_regression.py)

Description

This part involves using a neural network for regression to predict house prices in California, with implementation in PyTorch.

Running the Code

Executing part2_house_value_regression.py performs the following:

- Training a neural network regressor on the California housing dataset.
- Evaluating the model's performance on test data.
- Demonstrating model saving, loading, and scoring functions.

Example Usage

# Initializing and training a regressor for house price prediction.
# The regressor is configured with specific hyperparameters and trained on the dataset.
regressor = Regressor(x_train, nb_epoch=1000)
regressor.fit(x_train, y_train)

# Saving the trained model for future predictions or evaluations.
save_regressor(regressor)


Installation and Execution

To run the code, ensure Python and the required libraries are installed. Install dependencies with:

pip install numpy torch pandas scikit-learn

Run the scripts in your Python environment:

python part1_nn_lib.py  # For Part 1
python part2_house_value_regression.py  # For Part 2

Ensure you have the necessary data files in the same directory for Part 2.


Additional Notes

- Part 1 provides a fundamental understanding of neural network components.
- Part 2 focuses more on practical application and can be extended for different regression tasks.

