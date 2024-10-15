import numpy

class Vanila:
    """
    In NN class there are different ML models and functions available that helps you build and train your ML model.
    """
    def __init__(self):
        self.layers = []
        self.init_method = None


    def add_vanila(self, number_of_neurons):
        if not isinstance(number_of_neurons, int):
            raise TypeError("The number of neurons must be an integer.")
        if number_of_neurons < 1:
            raise ValueError("The number of neurons must be at least 1.")
        self.layers.append(number_of_neurons)

    def properties(self):
        print("It is a vanila neural network with "+ str(len(self.layers)) + " layers. The number of neurons are: "+ str(self.layers) + ".")
        if self.init_method is not None:
            print("The initializtion method is: "+ str(self.init_method))

    def set_init_method(self, init_method=""):
        if init_method == "xavier" or init_method == "he":
            self.init_method = init_method
        else:
            raise ValueError("For vanila model only xavier and he is available at this moment")
