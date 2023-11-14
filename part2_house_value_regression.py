import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import math


class Regressor(nn.Module):

    def __init__(self, x, hidden_layer_sizes:list[int], batch_size = 10, learning_rate = 0.001, activation_function = "relu", optimizer = "adam", 
                 nb_epoch = 400, reports_per_epoch = 10):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - hidden_layer_sizes {list[int]} -- A list of layer sizes.
            - batch_size {int} -- Size of the batch used for training.
            - learning_rate {float} -- Optimizer learning rate.
            - activation_function {str} -- Activation function to use for each layer.
            - optimizer {str} -- Optimizer to use for training.
            - nb_epoch {int} -- number of epochs to train the network.
            - reports_per_epoch {int} -- Number of training steps to use per epoch for reporting purposes.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        super(Regressor, self).__init__()
        self.standardiser = StandardScaler()
        self.lb = LabelBinarizer()   
        
        X, _ = self._preprocessor(x, training = True)
        self.data_size = X.shape[0]
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size 
        self.reports_per_epoch = reports_per_epoch
        
        self.activation_function = activation_function
        
        self.layers = nn.ModuleList([nn.Linear(self.input_size, hidden_layer_sizes[0])] + 
                                    [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]) for i in range(len(hidden_layer_sizes)-1)] 
                                    + [nn.Linear(hidden_layer_sizes[-1], self.output_size)])
        
        self.loss_fn = nn.MSELoss()
        
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
            
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        
        else: # Default to Adam
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    
    # Forward Function
    def forward(self, x):
        for layer in self.layers:
            if self.activation_function == "Sigmoid":
                x = nn.functional.sigmoid(layer(x))
                
            elif self.activation_function == "ReLU":
                x = nn.functional.relu(layer(x))
                
            elif self.activation_function == "Tanh":
                x = nn.functional.tanh(layer(x))
            
            else: # Default to ReLU
                x = nn.functional.relu(layer(x))
        return x
    
    
    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        """
        1. fillna
        2. standardise
        3. one hot encode
        4. return x, y
        """
        
        # Columns with categorical data in a list
        non_float_cols = [col for col in x.columns if x[col].dtype == "object"]
        
        # Separate data into categorical and numerical
        categorical_data = x[non_float_cols]
        numerical_data = x.drop(non_float_cols, axis = 1)
        
        # Fill empty values with random variables
        empty_filled_numerica_data = self.fill_empty_labels(numerical_data)

        # Standardisation
        if training: # Training data fits onto the standardiser model
            self.standardiser.fit(empty_filled_numerica_data)
            standardised_x = pd.DataFrame(self.standardiser.transform(empty_filled_numerica_data), columns = empty_filled_numerica_data.columns)
        else: # Testing data uses the standardiser model from training (it cannot fit onto the model)
            standardised_x = pd.DataFrame(self.standardiser.transform(empty_filled_numerica_data), columns = empty_filled_numerica_data.columns)
        
        # One Hot Encoding with LabelBinarizer
        one_hot_encoded_data = self.one_hot_encode(categorical_data, non_float_cols)
        
        # Concatenate the categorical and numerical data
        preprocess_data = pd.concat([standardised_x, one_hot_encoded_data], axis = 1)
        
        # Return preprocessed x and y, return None for y if it was None
        return preprocess_data, (y if isinstance(y, pd.DataFrame) else None)
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    # Correct empty feature instances with a normally distributed random variable using the feature mean and standard deviation
    def fill_empty_labels(self, raw_input):
        """
        Fill in the empty values of the raw input with a random variable 
        using the mean and standard deviation of the label

        Args:
            raw_input (pandas dataframe): dataframe containing the raw input data

        Returns:
            _type_: all nan are filled with random variables
        """
        for label in raw_input.columns: # loop for all labels
            if raw_input[label].dtype == "object": # if label is not a float, skip
                continue
            mean = raw_input[label].mean() # label mean 
            std_dev = raw_input[label].std() # label std dev
            mask = raw_input[label].isnull() # find where empty values exist
            num_empty = mask.sum()
            
            if num_empty > 0: # if empty values detected
                random_values = np.random.normal(mean, std_dev, size=num_empty).astype(int) # generate a list of varying random variables
                random_values[random_values < 0] = 0 # only total_bedrooms has empty values, ensure values are positive whole numbers
                
                raw_input.loc[mask, label] = random_values # assign a random value to each of the empty values
                
        return raw_input


    # One hot encode the categorical data
    def one_hot_encode(self, raw_input, non_float_cols):
        """
        One hot encode the categorical data

        Args:
            raw_input (pandas dataframe): dataframe containing the raw input data
            non_float_cols (list): list of all non float columns

        Returns:
            _type_: all categorical data is one hot encoded
        """
        if len(non_float_cols) != 0:
            
            encoded_dict = {}
            
            for non_float_col in non_float_cols:
                self.lb.fit(raw_input[non_float_col])
                
                encoded = self.lb.fit_transform(raw_input[non_float_col])

                for i, unique_label in enumerate(raw_input[non_float_col].unique()):
                    feature = "is_" + str(unique_label)
                    
                    encoded_dict[feature] = encoded[:, i]
            
            # Initialise the one_hot_encoded_data dataframe
            one_hot_encoded_data = pd.DataFrame(encoded_dict, columns=encoded_dict.keys())
            
            return one_hot_encoded_data
        else:
            return None


    # Pytorch dataloader    
    def _dataloader(self, x, y):
        """
        Convert the data into a dataloader (compatible with pytorch)

        Args:
            x (Pandas Dataframe): input data
            y (Pandas Dataframe): labels
            batch_size (int, optional): _description_. Defaults to 32.
        
        Returns:
            _type_: DataLoader object
        """
        data_set = TensorDataset(torch.Tensor(np.array(x)), torch.Tensor(np.array(y)))
        
        data_loader = DataLoader(data_set, batch_size = self.batch_size, shuffle = False)
        
        return data_loader
    
    
    def train_one_epoch(self, epoch_index, train_loader):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            # if i % (self.data_size)/(self.batch_size * self.reports_per_epoch) == (self.data_size)/(self.batch_size * 10) - 1:
            #     last_loss = running_loss / (self.data_size)/(self.batch_size * self.reports_per_epoch) # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     running_loss = 0.
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))

        return last_loss
    
    
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess the data
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        
        # Convert the data into a dataloader
        train_loader = self._dataloader(x_train, y_train)
        val_loader = self._dataloader(x_val, y_val)
        
        best_vloss = 1_000_000.
        
        for epoch in range(self.nb_epoch):
            print('EPOCH {}:'.format(epoch + 1))
            
            self.train(True)
            
            avg_loss = self.train_one_epoch(epoch, train_loader)
            
            running_vloss = 0.0
            
            self.eval()
            
            with torch.no_grad():
                for i, vdata in enumerate(val_loader):
                    vinputs, vlabels = vdata
                    voutputs = self(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss
                
            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {} valid_sqrt {}'.format(avg_loss, avg_vloss, math.sqrt(avg_vloss)))
            
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model'
            torch.save(self.state_dict(), model_path)
        
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
      
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    
    data = pd.read_csv("housing.csv")
    output_label = "median_house_value"
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    
    regressor = Regressor(x_train, [13,12])
    
    regressor.fit(x_train, y_train)

    