import numpy as np

# load the data
x = np.loadtxt("pixels.txt", delimiter=",")
y_flat = np.loadtxt("digits.txt", delimiter=",")
L = len(x)
x = np.concatenate((np.ones((L,1)), x), axis=1)

# shuffle the data
import random
idx = random.sample(range(L),L)

x = x[idx]
y_flat = y_flat[idx]

# expand labels into binary array for classification
y = np.zeros((L,10))
for i in range(10):
  y[:,i] = (y_flat == i+1).astype(float)

# choose our training and test samples
test_size = 100
x_train = x[:-test_size]
y_train = y[:-test_size]
x_test = x[-test_size:]
y_test = y[-test_size:]

# create our neural net
from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation

# 20x20 pixel images + bias unit = 401 inputs
num_inputs = 401
# 1 hidden layer with 25 units (arbitrary)
hidden_units = 25
# 10 output categories
num_outputs = 10

model.add(Dense(hidden_units, activation='relu', input_dim=num_inputs))
model.add(Dense(num_outputs, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print loss_and_metrics
