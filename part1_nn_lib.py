import numpy as np
import pickle


def xavier_init(size, gain = 1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    @staticmethod
    def sigmoid(x):
        """
        Compute the sigmoid function, applied to the input x.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Sigmoid of the input array applied
                            element-wise.
        """
        # Element-wise sigmoid computation
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out), which is
                            the sigmoid of the input.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Compute the sigmoid of the input and cache it
        output = self.sigmoid(x)
        self._cache_current = output
        return output

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Compute the derivative of the sigmoid using the cached output
        sigmoid_output = self._cache_current
        sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)

        # Hadamard product of the incoming gradient with the sigmoid derivative
        return grad_z * sigmoid_derivative

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None # here we save last batched input x

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Apply ReLU to each element: if x > 0, output x; otherwise, output 0
        output = np.maximum(0.0, x)
        # Cache the output
        self._cache_current = output

        return output

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Compute the derivative of ReLU (1 where output > 0; 0 otherwise) by
        # converting booleans (True/False) for output > 0 to integers (1/0)
        output = self._cache_current
        relu_derivative = (output > 0).astype(np.int64)

        # Hadamard product of the incoming gradient with the ReLU derivative
        return grad_z * relu_derivative

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Use a uniform Xavier initialisation of the weights matrix
        self._W = xavier_init((n_in, n_out))
        # Initialise biases as a one-dimensional array of zeros
        self._b = np.zeros(n_out)

        # Cache for storing the input 'x' of the forward pass
        self._cache_current = None
        # Cache for storing the computed gradient of the loss wrt the weights
        self._grad_W_current = None
        # Cache for storing the computed gradient of the loss wrt the biases
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Affine transformation to get the output Z = XW + B
        # The biases '_b' are broadcasted to match the shape of 'x @ _W'
        z = (x @ self._W) + self._b

        # Store the input 'x' in the cache for use during the backward pass
        self._cache_current = x

        return z  # output z
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Retrieve the input 'x' used during the forward pass
        x = self._cache_current

        # Compute gradient of the function wrt to the layer parameters:
        # Compute gradient of the loss with respect to the weights ('_W')
        self._grad_W_current = x.T @ grad_z
        # Compute gradient of the loss with respect to the biases ('_b')
        self._grad_b_current = np.sum(grad_z, axis=0)

        # Compute and return gradient of the function wrt to the layer inputs:
        grad_input_x = grad_z @ self._W.T

        return grad_input_x
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Update weights using the learning rate and the gradient of weights
        self._W -= learning_rate * self._grad_W_current

        # Update biases using the learning rate and the gradient of biases
        self._b -= learning_rate * self._grad_b_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer
                represented as a list. The length of the list determines the
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        # Ensure number of layers matches number of activation functions
        assert len(neurons) == len(activations), \
            "Number of layers must match number of activations"

        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # List to store layers in the network
        self._layers = []
        no_of_layers = len(neurons)

        # Create all layers in the network using a single loop
        for i in range(no_of_layers):
            # Determine the number of input features for the current layer
            n_in = input_dim if i == 0 else neurons[i - 1]
            n_out = neurons[i]

            # Initialize the linear layer and add it to the network
            self._layers.append(LinearLayer(n_in, n_out))

            # Get the activation layer instance, only append if it's not None
            activation_layer = self._convert_to_activation(activations[i])
            if activation_layer:
                self._layers.append(activation_layer)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _convert_to_activation(self, string_):
        """
        Converts a string name to an activation layer instance or returns None
        for identity (linear) activation.

        Arguments:
            string_ {str} -- Name of the activation function. Supported values
                             are 'relu', 'sigmoid', and 'identity'
                             (linear activation).

        Returns:
            {Layer} -- An instance of the specified activation layer, or None
                       for identity activation. Returns a ReluLayer instance
                       for 'relu', a SigmoidLayer instance for 'sigmoid', and
                       None for 'identity'.

        Raises:
            ValueError: If the provided activation function name is not
                        'relu', 'sigmoid', or 'identity'.
        """
        # Convert the string to lowercase to handle different cases
        activation_name = string_.lower()

        if activation_name == "relu":
            return ReluLayer()
        elif activation_name == "sigmoid":
            return SigmoidLayer()
        # elif activation_name == "identity":
        #     return None
        else:
            return None
            # raise ValueError(f"Unrecognised activation function: {string_}")

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Initialise the result with the input data
        result = x

        # Sequentially apply forward pass of each network layer to the result
        for layer in self._layers:
            result = layer.forward(result)

        # Return the final output after all layers have been applied
        return result 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Initialise the result with the gradient from the final layer
        grad_result = grad_z

        # Sequentially apply backward pass of each layer in reverse order
        for layer in self._layers[::-1]:
            grad_result = layer.backward(grad_result)

        # Return the final gradient after all layers have been applied
        return grad_result
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Update the parameters of each layer using the stored gradients
        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Convert the string to lowercase to handle different cases
        loss_fun_name = self.loss_fun.lower()

        # Initialise the loss layer based on the specified loss function
        if loss_fun_name == "mse":
            self._loss_layer = MSELossLayer()
        elif loss_fun_name == "cross_entropy":
            self._loss_layer = CrossEntropyLossLayer()
        else:
            self._loss_layer = None
            # raise ValueError(f"Unrecognised loss function: {self.loss_fun}")
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Ensure both input_dataset & target_dataset are two-dimensional arrays
        if input_dataset.ndim == 1:
            input_dataset = np.reshape(input_dataset, (-1, 1))
        # if target_dataset.ndim == 1:
        #     target_dataset = np.reshape(target_dataset, (-1, 1))

        # Generate a random permutation of indices for the dataset
        randomised_indices = np.random.permutation(len(input_dataset))

        # Shuffle both datasets using the same random indices
        # This maintains alignment between each input and corresponding target
        shuffled_inputs = input_dataset[randomised_indices]
        shuffled_targets = target_dataset[randomised_indices]

        # Return the shuffled datasets
        return shuffled_inputs, shuffled_targets

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for epoch in range(self.nb_epoch):
            # Shuffle the dataset if the shuffle_flag is set to True
            if self.shuffle_flag == True:
                input_dataset, target_dataset = self.shuffle(input_dataset,
                                                             target_dataset)

            # Get the total number of data points in the training dataset
            no_of_data_points = len(input_dataset)

            # Iterate over mini-batches
            for i in range(0, no_of_data_points, self.batch_size):
                # Extract the current batch from the dataset
                batch_inputs = input_dataset[i:i + self.batch_size]
                batch_targets = target_dataset[i:i + self.batch_size]

                # Forward pass
                outputs = self.network.forward(batch_inputs)
                loss = self._loss_layer.forward(outputs, batch_targets)

                # Backward pass
                grad_loss = self._loss_layer.backward()
                self.network.backward(grad_loss)

                # Update parameters using one-step gradient descent
                self.network.update_params(self.learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        
        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Perform a forward pass and compute the loss
        outputs = self.network.forward(input_dataset)
        return self._loss_layer.forward(outputs, target_dataset)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Compute the minimum value for each feature in the dataset
        self._features_min = np.min(data, axis=0)

        # Compute the maximum value for each feature in the dataset
        self._features_max = np.max(data, axis=0)

        # Calculate the range for each feature
        self._features_range = self._features_max - self._features_min
        # Handle the case where max equal to min to avoid division by zero
        self._features_range[self._features_range == 0] = 1

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Apply min-max scaling to the data using the precomputed range
        return (data - self._features_min) / self._features_range

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Revert the min-max scaling using the precomputed range
        return data * self._features_range + self._features_min

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
    network = MultiLayerNetwork(input_dim=4, neurons=[16,16,4,4,2,2,4],
                                activations=["relu", "relu", "identity",
                                             "relu","sigmoid", "relu",
                                             "sigmoid"])
    inputs = np.array([[3,1,4,5], [7,3,9,0], [1,5,3,7], [6,2,8,9]])
    outputs = network.forward(inputs)
    
    print(outputs)


