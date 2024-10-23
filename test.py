import numpy
from lab import Vanila
from lab import train
from lab import Loss

model = Vanila()
model.set_init_method("xavier")
model.add_vanila(number_of_neurons=20)
model.add_vanila(number_of_neurons=640)
model.add_vanila(number_of_neurons=10)
# model.print_weights()
# weights = model.weights
# print(weights[0])

model.set_activation('relu')
y = model.forward([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y_pred = [1,2,3,4,5,6,7,8,9,10]
loss = train.get_loss(y=y,y_pred=y_pred, loss_method=Loss.CCE)
# print(loss)
# print(len(model.z_values[0]))
# model.properties()

epoch = 100
for epoch in range(epoch):
    y = model.forlward([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    y_pred = [1,2,3,4,5,6,7,8,9,10]
    loss = train.get_loss(y=y,y_pred=y_pred, loss_method=Loss.CCE)
    train.backpropagation(model,loss)
    


