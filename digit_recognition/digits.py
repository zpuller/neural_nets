import numpy as np

x = np.loadtxt("pixels.txt", delimiter=",")
y_flat = np.loadtxt("digits.txt", delimiter=",")
L = len(x)
x = np.concatenate((np.ones((L,1)), x), axis=1)

import random
idx = random.sample(range(L),L)

x = x[idx]
y_flat = y_flat[idx]

y = np.zeros((L,10))
for i in range(10):
  y[:,i] = (y_flat == i+1).astype(float)

x_train = x[:-100]
y_train = y[:-100]
x_test = x[-100:]
y_test = y[-100:]

from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation

model.add(Dense(25, activation='relu', input_dim=401))
model.add(Dense(10, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print loss_and_metrics
