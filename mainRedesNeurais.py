import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Crie um callback para reduzir a taxa de aprendizado quando a perda de validação não melhorar
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.001)

# Adicione o callback à lista de callbacks
callbacks_list = [reduce_lr]

arq1 = 'data/treino_sinais_vitais_com_label.txt'
arq2 = 'data/textePequeno.txt'

col_names = ['pSist','pDiast','qPA', 'pulso', 'resp', 'gravid', 'classe']
data = pd.read_csv(arq1)
X = pd.get_dummies(data.drop(['classe', 'pSist', 'pDiast'], axis=1))
y = data['classe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
print(y_train.head())


model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dropout(0.8))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics='accuracy')


# Subtraia 1 se suas classes são 1, 2, 3 e 4 em vez de 0, 1, 2 e 3
y_train_one_hot = to_categorical(y_train - 1)

# Agora use y_train_one_hot em vez de y_train ao ajustar o modelo
model.fit(X_train, y_train_one_hot, epochs=50000, batch_size=128, callbacks=callbacks_list)

y_hat = model.predict(X_test)
y_hat = np.argmax(y_hat, axis=1) + 1
print(accuracy_score(y_test, y_hat))