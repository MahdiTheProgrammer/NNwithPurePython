import numpy
from lab import Vanila
from lab import train
from lab import Loss

model = Vanila()
model.set_init_method("xavier")
model.add_vanila(number_of_neurons=3)
model.add_vanila(number_of_neurons=4)
model.add_vanila(number_of_neurons=4)
model.add_vanila(number_of_neurons=3)
model.add_vanila(number_of_neurons=2)
# model.print_weights()
# weights = model.weights
# print(len(model.a_values))


model.set_activation('sigmoid')
y = model.forward([1,2,3])

y_pred = [1,2]
loss = train.get_loss(y=y,y_pred=y_pred, loss_method=Loss.CCE)
# print(loss)

print(len(model.layers))
print(len(model.z_values))
print(len(model.a_values))
print(len(model.weights[3]))
# model.properties()

epoch = 1
for epoch in range(epoch):
    y = model.forward([1,2,3])
    y_pred = [1,2]
    loss = train.get_loss(y=y,y_pred=y_pred, loss_method=Loss.CCE)
    grad_w= train.backpropagation( model , y, y_pred, [1,2,3])
    


