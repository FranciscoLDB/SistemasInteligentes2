from sklearn import datasets
from sklearn.model_selection import train_test_split
from randomForest import RandomForest
from sklearn.metrics import accuracy_score
import pandas as pd

arq1 = 'data/treino_sinais_vitais_com_label.txt'
arq2 = 'data/textePequeno.txt'

col_names = ['pSist','pDiast','qPA', 'pulso', 'resp', 'gravid', 'classe']
data = pd.read_csv(arq1, skiprows=1, header=None, names=col_names)
data = data.drop('pSist', axis=1)
data = data.drop('pDiast', axis=1)
data = data.drop('gravid', axis=1)
#print(data.head())

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
clf = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')