import numpy as np
import math

class Vanila:
    """
    In NN class there are different ML models and functions available that helps you build and train your ML model.
    """
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = [] 
        self.init_method = None
        self.activation = None
        self.a_values = []
        self.z_values = []


    def add_vanila(self, number_of_neurons):
        if not isinstance(number_of_neurons, int):
            raise TypeError("The number of neurons must be an integer.")
        if number_of_neurons < 1:
            raise ValueError("The number of neurons must be at least 1.")
        if self.init_method == None:
            raise ValueError("In order to add a layer you have to set the initializtion method")

        if self.layers:
            previous_neurons = self.layers[-1]
            if self.init_method == "xavier":
                weights = np.random.uniform(-(np.sqrt(6/(previous_neurons+number_of_neurons))),
                                             np.sqrt(6/(previous_neurons+number_of_neurons)), (previous_neurons, number_of_neurons))
            else:
                weights = np.random.randn(previous_neurons, number_of_neurons) * np.sqrt(2 / previous_neurons) 

            self.weights.append(weights)
        
        self.biases.append(np.zeros((number_of_neurons, 1)))
        self.layers.append(number_of_neurons)

    def properties(self):
        print("It is a vanila neural network with "+ str(len(self.layers)) + " layers. The number of neurons are: "+ str(self.layers) + ".")
        if self.init_method is not None:
            print("The initializtion method is: "+ str(self.init_method))

    def set_init_method(self, init_method=""):
        """
        Choose between 'xavier' or 'he' initialization method.
        """
        if init_method == "xavier" or init_method == "he":
            self.init_method = init_method
        else:
            raise ValueError("For vanila model only xavier and he is available at this moment")        

    def print_weights(self):
        for i, weight in enumerate(self.weights):
            print(f"Weights for layer {i+1}:\n{weight}\n")

    def forward(self, input):
        self.a_values = []
        self.z_values = []


        if self.activation == None:
            raise ValueError("No activation function is set.")
        # self.z_values = input
        for f1 in range(len(self.layers)-1):
            # output = self.activation(np.dot(output, self.weights[f1]) + self.biases[f1+1].T)
            self.z_values.append(np.dot(input if f1 == 0 else self.z_values[-1], self.weights[f1]) + self.biases[f1+1].T)
            if f1 != len(self.layers)-2:
                self.a_values.append(self.activation(self.z_values[-1]))
            else:
                self.a_values.append(Activations.softmax(self.z_values[-1]))   #This is because no matter what i use in hidden layers i want to use softmax for last layer.
            

        # return output
        return self.a_values[-1]

    def set_activation(self, activation):
        activations = Activations()
        if activation == 'sigmoid':
            self.activation = activations.sigmoid
        elif activation == 'tanh':
            self.activation = activations.tanh
        elif activation == 'relu':
            self.activation = activations.ReLU
        else:
            raise ValueError("Supported activations: 'sigmoid', 'tanh', 'relu', 'softmax'.")
        

class Activations:
    def sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))  
        return y

    def tanh(self, x):
        y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  
        return y

    def ReLU(self, x):
        return np.maximum(0, x)  
    
    @staticmethod
    def softmax(x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)  # Fix stability issue for batches
        exp_values = np.exp(x_shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    

class Loss:
    def MSE(y_pred, y_true):
        """
        Calculate the Mean Squared Error (MSE) between true values and predictions.

        Parameters:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values

        Returns:
        float: MSE between y_true and y_pred
        """
        # Convert inputs to numpy arrays for element-wise operations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate squared differences, mean them
        mse = np.mean((y_true - y_pred) ** 2)
        return mse
    
    def MAE(y_pred, y_true):
        """
        Calculate the Mean Absolute Error (MAE) between true values and predictions.

        Parameters:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values

        Returns:
        float: MAE between y_true and y_pred
        """
        # Convert inputs to numpy arrays for element-wise operations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate squared differences, mean them
        mse = np.mean(np.abs(y_true - y_pred))
        return mse

    def huber_loss(y_pred, y_true, delta):
        abs_loss = np.abs(y_pred - y_true)
        loss = np.where(abs_loss < delta,
                        0.5 * (y_pred - y_true) ** 2,  
                        delta * (abs_loss - 0.5 * delta))  
        return np.mean(loss) 

    def BCE( y_pred, y_true):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1-y_true)* np.log(1-y_pred))

    def CCE(y_pred, y_true):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        return -np.mean(np.sum(y_true*np.log(y_pred), axis=1))


class train:
    def get_loss(y, y_pred, loss_method, ):
        return loss_method(y_pred = y_pred, y_true = y)
        
    def backpropagation(model , y, y_pred, X, loss_function, activation_function):
        # assuming we are using categorical cce and sigmoid
        delta = []
        # model_len = len(model.layers)-2
        # for f1 in range(len(model.layers)-1):
        #     print(f"f1 is: {f1}")
            # if f1 == 0: 
            #     res = np.array(-y_pred) / np.array(model.a_values[model_len-f1])
            #     res = np.dot(np.array(res) *((math.e**(-model.z_values[model_len-f1]))/((1+math.e**[model_len-f1])**2)))
            #     res = np.dot(np.array(res) ,np.array(model.weights[model_len-f1]))
            # if f1 == 0: 
            #     deltan = np.dot((-(np.array(y_pred) / np.array(model.a_values[model_len-f1]))).T,
            #                    (math.e**(-model.z_values[model_len-f1])/((1+math.e**(-model.z_values[model_len-f1])))**2))
            # else:
            #     deltan = np.dot(delta[-1], np.array(model.weights[model_len-f1+1]).T)
            #     deltan = np.dot(deltan, (math.e**(-model.z_values[(model_len-f1)])/((1+math.e**(-model.z_values[(model_len-f1)])))**2).T)
            # print(f"last delta shape is: {deltan.shape}")
            # delta.append(deltan)
        #     grad_w = 'a'
        #     grad_b = 'b'
        # return grad_w, grad_b
        return delta