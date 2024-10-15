import numpy

class nn():
    """
    In NN class there are different ML models and functions available that helps you build and train your ML model.
    """
    def __init__(self):
        self.layers = []
        self.init_method = None


    def add_vanila(self, number_of_neurons):
        self.layers.append(number_of_neurons)

    

