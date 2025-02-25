from lab import Vanila
import numpy as np

model = Vanila()
model.set_init_method("xavier")

model.add_vanila(number_of_neurons=128)
model.add_vanila(number_of_neurons=64)
model.add_vanila(number_of_neurons=10)

model.properties()

# model.print_weights()

model.set_activation(activation='relu')

x= model.forward(np.linspace(0,1,128))

print(x)

print(np.sum(x))