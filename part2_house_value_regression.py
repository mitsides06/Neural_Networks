import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

import pickle
import numpy as np
import pandas as pd

 
class NueralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation_function):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, self.hidden_layer_sizes[0]))
        for i in range(1, len(self.hidden_layer_sizes)):
            self.layers.append(nn.Linear(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]))
        self.layers.append(nn.Linear(self.hidden_layer_sizes[-1], self.output_size))


    def forward(self, x):
        
        if self.activation_function == 'relu':
            activation = torch.nn.ReLU()
        elif self.activation_function == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif self.activation_function == 'tanh':
            activation = torch.nn.Tanh()
        
        for layer in self.layers[:-1]:
            x = activation(layer(x))
        
        # No activation function on the last layer
        x = self.layers[-1](x)
        return x


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
    
    
    def early_stop(self, validation_loss):
        # validation loss is less than min validation loss
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            
        # validation loss is greater than min validation loss but less than min validation loss + min delta
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Regressor():

    def __init__(self, x, nb_epoch = 1000, hidden_layers = [64, 128, 256, 128, 64], activation_function = 'relu', 
                 optimizer="RMSprop", learning_rate = 0.001):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        self.numerical_x_mean = None
        self.categorical_x_mode = None
        
        self.label_binarizer = LabelBinarizer()
        self.standard_scaler = StandardScaler()
        
        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        
        self.hidden_layer_sizes = hidden_layers
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        
        self.model = NueralNetwork(self.input_size, self.output_size, self.hidden_layer_sizes, self.activation_function)
        
        self.early_stopping = EarlyStopping()
        
        return
    

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

        # Raise an exception if the model is not trained yet as we need to
        if not training and (self.categorical_x_mode is None or self.numerical_x_mean is None):
            raise Exception("You must train the model before testing it")
        
        # Remove rows with missing values
        if training and y is not None:
            x = x[y.notna().values]
            y = y[y.notna()]
        
        # Split numerical and categorical columns
        non_float_columns = [col for col in x.columns if x[col].dtype != np.float64]
        numerical_x = x.drop(non_float_columns, axis=1)
        categorical_x = x.loc[:, non_float_columns]

        # Fill missing values with mean or mode
        if training:
            self.numerical_x_mean = numerical_x.mean(axis=0)
            self.categorical_x_mode = categorical_x.mode(axis=0).iloc[0]
            
        numerical_x.fillna(self.numerical_x_mean, inplace=True)
        categorical_x.fillna(self.categorical_x_mode, inplace=True)
        
        # Standardize numerical columns and one-hot encode categorical columns
        if training:
            self.standard_scaler.fit(numerical_x)
            self.label_binarizer.fit(categorical_x)
        
        # Concatenate numerical and categorical columns
        numerical_x_df = pd.DataFrame(self.standard_scaler.transform(numerical_x), columns=numerical_x.columns)
        categorical_x_df = pd.DataFrame(self.label_binarizer.transform(categorical_x), columns=self.label_binarizer.classes_)
        
        preprocessored_x = pd.concat([numerical_x_df, categorical_x_df], axis=1)
    
        return torch.tensor(preprocessored_x.values).float(), torch.tensor(y.values).float() if y is not None else None

        
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

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

        loss = nn.MSELoss()
        
        if self.optimizer == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "RMSprop":
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "Adagrad":
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        else: # default
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for _ in range(self.nb_epoch):
            self.model.train(True)
            
            optimizer.zero_grad()
            predictions = self.model(x_train)
            mse = loss(predictions, y_train)
            mse.backward()
            optimizer.step()
            
            self.model.eval()
            
            with torch.no_grad():
                predictions = self.model(x_val)
                vloss = loss(predictions, y_val)
            
            # print("Training Loss: {}, Validation Loss: {}".format(mse, vloss))
            
            if self.early_stopping.early_stop(vloss):
                # print("Early stopping")
                break
            
        return self

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        with torch.no_grad():
            X, _ = self._preprocessor(x, training = False) # Do not forget

            output = self.model(X)
            
            return np.array(output)


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

        with torch.no_grad():
            
            X, Y = self._preprocessor(x, y = y, training = False)
            
            predictions = self.model(X)
            
            return mean_squared_error(np.array(Y), np.array(predictions), squared=False)
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
       
    def get_params(self, deep=True):
        return {"hidden_layers": self.hidden_layer_sizes, "activation_function": self.activation_function,
                "optimizer": self.optimizer, "learning_rate": self.learning_rate}


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
    
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Define the parameter grid to search
    param_grid = {
        'hidden_layers': [
            [16],
            [16, 16],
            [16, 16, 16],
            [16, 16, 16, 16],
            [32],
            [32, 32],
            [32, 32, 32],
            [32, 32, 32, 32],
            [64],
            [64, 64],
            [64, 64, 64],
            [64, 64, 64, 64],
            [128],
            [128, 128],
            [128, 128, 128],
            [128, 128, 128, 128],
            ],
        'activation_function': ['relu', 'sigmoid', 'tanh'],
        'optimizer': ['Adam', 'Adagrad', 'RMSprop'],
        'learning_rate': [0.001,0.005, 0.01, 0.05, 0.1]
    }
    
    best_score = float("inf")
    best_model = None
    counter = 0
    
    parameter_storage = []
    
    for hidden_layer in param_grid['hidden_layers']:
        for activation_function in param_grid['activation_function']:
            for optimizer in param_grid['optimizer']:
                for learning_rate in param_grid['learning_rate']:
                    print("Counter: {}".format(counter + 1))
                    print("Hidden Layers: {}, Activation Function: {}, Optimizer: {}, Learning Rate: {}".format(hidden_layer, activation_function, optimizer, learning_rate))
                    regressor = Regressor(x_train, hidden_layers = hidden_layer, activation_function = activation_function, 
                                          optimizer = optimizer, learning_rate = learning_rate)
                    regressor.fit(x_train, y_train)
                    score = regressor.score(x_val, y_val)
                    parameter_storage.append([len(hidden_layer) , hidden_layer[0], activation_function, optimizer, learning_rate, score])
                    
                    pd.DataFrame(parameter_storage, columns = ['Number of Hidden Layers', 'Number of Neurons per Layer', 'Activation Function', 'Optimizer', 'Learning Rate', 'Score']).to_csv("parameter_storage.csv", index=False)
                    
                    print("Score: {}".format(score))
                    print("\n")
                    if score < best_score:
                        best_score = score
                        best_model = regressor
                        save_regressor(regressor)

                    counter += 1
    
    best_model_score = best_model.score(x_test, y_test)
    print(f"Best Model Score on Test Data: {best_model_score}")
    
    return



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 1000)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    
    RegressorHyperParameterSearch()
    
    # example_main()
