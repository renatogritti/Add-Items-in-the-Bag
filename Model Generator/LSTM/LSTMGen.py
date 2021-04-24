import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import MeanSquaredError


df = pd.read_csv("Model Generator/LSTM/model.csv")

dataraw = df.values.astype('float32')
dataset = dataraw

y = dataset[:,0]
X = dataset[:,1:]

train_y = np.array(y)
train = np.array(X)
train_X = np.reshape(train, (train.shape[0],1,train.shape[1]))
print(train_X.shape)

model = Sequential()

model.add(LSTM(250, input_shape = (1, 8)))
model.add(Dense(5))
model.add(Dense(4))

model.compile( optimizer='adam', loss="mean_absolute_error")
model.fit(train_X, train_y, epochs=1000, batch_size=20, verbose=1)


dados = [[0.0,0.0,0.0,4.0,0.0,0.0,0.0,4.0]]
adados = np.array(dados)
ar = np.reshape(adados, (adados.shape[0],1,adados.shape[1]))

previsao = model.predict(ar)

print(previsao)

model.save('lSTM.dat')




