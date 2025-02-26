from lab import Vanila
from lab import train
import numpy as np

model = Vanila()
model.set_init_method("xavier")

model.add_vanila(number_of_neurons=10)
model.add_vanila(number_of_neurons=2)
model.add_vanila(number_of_neurons=3)

# model.add_vanila(number_of_neurons=10)

model.properties()

model.print_weights()

model.set_activation(activation='relu')

input = np.linspace(0,1,10)

x= model.forward(input)

print(x)

print(np.sum(x))


delta ,dw, db = train.backpropagation (model = model, y_pred=x, y=[1,0,0],X=np.expand_dims(input,axis=0))
# print(x)
print(dw)
print(db)