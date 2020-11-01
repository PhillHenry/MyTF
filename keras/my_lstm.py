# LSTM example from https://machinelearningmastery.com/lstm-autoencoders/

import numpy as np
from math import pi
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import matplotlib.pyplot as plt


def raw_data(n_cycles, n_points):
    return np.sin(np.arange(n_points) * (n_cycles * 2 * pi / n_points))


def reshape(xs):
    return xs.reshape((1, (len(xs)), 1))


# define input sequence
n_in = 50
# reshape input into [samples, timesteps, features]
n_samples = 10

cycles_x3 = raw_data(3, n_in)
for i in range (n_samples - 1):
    data = raw_data((i + 1) / n_samples, n_in)
    cycles_x3 = np.concatenate((cycles_x3, data), axis=None)
cycles_x3 = cycles_x3.reshape(n_samples, n_in, 1)


#cycles_x3 = np.concatenate((raw_data(3, n_in), raw_data(2, n_in)), axis=None).reshape(n_samples, n_in, 1)

cycles_x1 = raw_data(1, n_in)
cycles_x1 = reshape(cycles_x1)

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in, 1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(cycles_x3, cycles_x3, epochs=300, verbose=0)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(cycles_x1, verbose=0)
predictions = yhat[0, :, 0]
print(predictions)

plt.plot(raw_data(1, n_in))
plt.plot(predictions)
plt.show()
