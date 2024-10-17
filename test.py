import numpy
from lab import Vanila


model = Vanila()
model.set_init_method("xavier")
model.add_vanila(number_of_neurons=20)
model.add_vanila(number_of_neurons=640)
model.add_vanila(number_of_neurons=10)
# model.print_weights()
# weights = model.weights
# print(weights[0])

model.set_activation('relu')
output = model.forward([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
print(len(output[0]))
# model.properties()


