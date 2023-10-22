from classes import *
import pandas as pd

arq1 = 'data/treino_sinais_vitais_com_label.txt'
arq2 = 'data/textePequeno.txt'

col_names = ['pSist','pDiast','qPA', 'pulso', 'resp', 'gravid', 'classe']
data = pd.read_csv(arq1, skiprows=1, header=None, names=col_names)
data = data.drop('pSist', axis=1)
data = data.drop('pDiast', axis=1)
print(data.head())

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test) 
from sklearn.metrics import accuracy_score
print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}')