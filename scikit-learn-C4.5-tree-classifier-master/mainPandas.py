import pandas as pd
from sklearn.model_selection import train_test_split
from c45_v2 import C45
import numpy as np

#Prepara dados
nomeAC = "treino_sinais_vitais_com_label.txt"
nomeAS = "treino_sinais_vitais_sem_label.txt"
testePequeno = "textePequeno.txt"
df = pd.read_csv(testePequeno)
df = df.drop('i', axis=1)
nomeColunas = ['qPA','pulso','resp','gravid']
cClasse = np.array(df['classe'].to_list())
cQPA = df['qPA'].to_list()
cPulso = df['pulso'].to_list()
cResp = df['resp'].to_list()
cGravid = df['gravid'].to_list()
data = np.vstack((cQPA,cPulso,cResp,cGravid)).reshape(len(cQPA),-1)
nomeColunas = ['qPA','pulso','resp','gravid']


clf2 = C45(attrNames=nomeColunas)
X_train, X_test, y_train, y_test = train_test_split(data, cClasse, test_size=0.5)
clf2.fit(X_train, y_train)
print(f'Accuracy: {clf2.score(X_test, y_test)}')
clf2.printTree()