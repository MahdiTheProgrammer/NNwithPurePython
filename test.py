import numpy
from lab import Vanila


model = Vanila()
model.add_vanila(number_of_neurons=128)
model.add_vanila(number_of_neurons=64)
model.add_vanila(number_of_neurons=10)
model.set_init_method("xavier")
model.properties()


