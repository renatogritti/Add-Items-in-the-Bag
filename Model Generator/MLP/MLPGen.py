
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle


df = pd.read_csv("Model Generator/MLP/model.csv")

df.head()


TRAIN_SIZE = 1
train_size = int(len(df) * TRAIN_SIZE)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size, :], df.iloc[train_size:len(df),:]


y = train.iloc[:,0]
X = train.iloc[:,1:]


# 5,4 hidden layers (see model in Orange)
model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(5,4),
                           max_iter=180, shuffle=True,
                           activation='logistic')
model.fit(X, y)


print(model.loss_)


teste = [[0,2,0,2,0,2,0,2]]


result = model.predict(teste)
print(result)


file = open('Model Generator/MLP/MLP.dat', 'wb')
pickle.dump(model, file)
file.close()




