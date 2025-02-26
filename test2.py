from lab import Vanila
from lab import train
from lab import Loss
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt

model = Vanila()
model.set_init_method("xavier")

model.add_vanila(number_of_neurons=784)
model.add_vanila(number_of_neurons=512)
model.add_vanila(number_of_neurons=256)
model.add_vanila(number_of_neurons=128)
model.add_vanila(number_of_neurons=64)
model.add_vanila(number_of_neurons=32)
# model.add_vanila(number_of_neurons=49)
model.add_vanila(number_of_neurons=10)

model.set_activation(activation='relu')

loss_list = []
true = 0 
lr = 0.01
for f1 in tqdm(range(30000)):
    if f1%10000==0:
        lr = lr/10
        print(f'lr decreased: {lr}')
    r = random.randint(0,9)
    n = random.randint(0,9000)
    image = Image.open(f'archive/dataset/{r}/{r}/{n}.png')
    # image = Image.open(f'archive/dataset/{r}/{r}/{n}.png').convert('L')
    image = np.array(image)
    image = image[:,:,3]
    image = image / 255.0  
    image = image.flatten()
    # print(np.shape(image))
    image = image.reshape(1,784)
    x = model.forward(image)   
    if np.argmax(x) == r:
        true+=1
    y_true = np.zeros((1,10))
    y_true[0][r] = 1
    loss = Loss.CCE(y_pred=x, y_true=y_true)        
    train.backpropagation(model = model, y_pred=x, y=y_true,X=image, lr=lr)
    loss_list.append(loss)
    if f1%200==0:      
        print(loss)
        

print(true)
plt.plot(loss_list, marker='o', linestyle='-')

# Add labels and title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Time")

# Show the plot
plt.show()