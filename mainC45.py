from classes.descisionTreeClassifier import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

arq1 = 'data/treino_sinais_vitais_com_label.txt'
arq2 = 'data/textePequeno.txt'

col_names = ['pSist','pDiast','qPA', 'pulso', 'resp', 'gravid', 'classe']
data = pd.read_csv(arq1, skiprows=1, header=None, names=col_names)
data = data.drop('pSist', axis=1)
data = data.drop('pDiast', axis=1)
#data = data.drop('gravid', axis=1)
#print(data.head())

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
#print(X)
#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,random_state=41)

classifier = DecisionTreeClassifier(max_depth=2, min_samples_split=2, mode='entropy')
classifier.fit(X_train,Y_train)
classifier.print_tree()
Y_pred = classifier.predict(X_test) 
print(f'D:2 / S:2 / M:E -> Accuracy: {accuracy_score(Y_test, Y_pred)}')
