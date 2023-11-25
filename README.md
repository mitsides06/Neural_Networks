# Neural Network Mini-Library and House Price Prediction

## Overview

This project is divided into two parts:

**1- Neural Network Mini-Library (part1_nn_lib.py):**
   - Implements a basic multi-layer neural network with backpropagation, data preprocessing, training, and evaluation.

**2- House Price Prediction (part2_house_value_regression.py):**
   - Utilises PyTorch (or the mini-library) to predict house prices in California.


## Part 1 Neural Network Mini-Library (part1_nn_lib.py)

### Description:

The mini-library includes implementations of neural network components such as various layers (linear, sigmoid, ReLU), loss functions (MSE, Cross-Entropy), and a Trainer for managing shuffling, batching, and updates.

### Running the Code:

Executing `part1_nn_lib.py` demonstrates creating layers, building and testing a network, training with a Trainer, and applying data preprocessing.

### Example Usage:

#### _Create a neural network_
```
network = MultiLayerNetwork(input_dim=4, neurons=[16, 2], activations=["relu", "sigmoid"])
```

#### _Forward and backward passes_
```
outputs = network(inputs)
grad_loss_wrt_inputs = network.backward(grad_loss_wrt_outputs)
```

#### _Update network parameters_
```
network.update_params(learning_rate)
```

#### _Train the network_
```
trainer = Trainer(network, batch_size=32, nb_epoch=10, learning_rate=1.0e-3, shuffle_flag=True, loss_fun="mse")
trainer.train(train_inputs, train_targets)
```

#### _Evaluate on validation data_
```
print("Validation loss = ", trainer.eval_loss(val_inputs, val_targets))
```

#### _Data normalisation_
```
prep = Preprocessor(dataset)
normalized_dataset = prep.apply(dataset)
original_dataset = prep.revert(normalized_dataset)
```


## Part 2: California House Prices Prediction (part2_house_value_regression.py)
 
### Description:

Implements a neural network for regression to predict California house prices using PyTorch.

### Running the Code:

Executing `part2_house_value_regression.py` involves training a regressor on the housing dataset, evaluating performance, and demonstrating saving/loading of the model.

### Example Usage:

#### _Initialize and train the regressor_
```
regressor = Regressor(x_train, nb_epoch=1000)
regressor.fit(x_train, y_train)
```

#### _Save the trained model_
```
save_regressor(regressor)
```


## Installation and Execution

### Installing dependencies:

`pip install numpy torch pandas scikit-learn`


### Running the scripts:

##### _For Part 1:_
`python3 part1_nn_lib.py`

##### _For Part 2:_
`python3 part2_house_value_regression.py`


## Additional Notes

- Part 1 offers a foundational understanding of neural network components.
- Part 2 focuses on practical application and can be adapted for other regression tasks.