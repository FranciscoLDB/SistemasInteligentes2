import pandas as pd
import math
def calculate_entropy(x):
    ent=0
    for k in set(x):
        p_i=float(x.count(k))/len(x)
        ent=ent-p_i* math.log(p_i,2)
    
    print('x:')
    print(x)
    print('Entropy:')
    print(ent)
    return ent

def alculate_entropy(values):
    total_values = len(values)
    value_counts = {}  # Dicionário para contar a frequência de cada valor
    for value in values:
        if value not in value_counts:
            value_counts[value] = 1
        else:
            value_counts[value] += 1

    entropy = 0
    #print('value_counts: ')
    #print(value_counts)
    #print('value_counts.values(): ')
    #print(value_counts.values())
    for count in value_counts.values():
        proportion = count / total_values
        entropy -= proportion * math.log2(proportion)

    return entropy

def calculate_entropy2(values, minV= 0, maxV= 0):
    total_values = len(values)
    value_counts = {}  # Dicionário para contar a frequência de cada valor
    dif = abs(maxV - minV) + 1
    print('dif: ' + str(dif))
    if dif > 10:
        inter = dif/10
        for i in range(10):
            val = inter*(i+1)
            value_counts[val] = 0
        #print(value_counts)    
        for value in values:
            if value <= inter:
                value_counts[inter] += 1
            elif value <= inter*2:
                value_counts[inter*2] += 1
            elif value <= inter*3:
                value_counts[inter*3] += 1
            elif value <= inter*4:
                value_counts[inter*4] += 1
            elif value <= inter*5:
                value_counts[inter*5] += 1
            elif value <= inter*6:
                value_counts[inter*6] += 1
            elif value <= inter*7:
                value_counts[inter*7] += 1
            elif value <= inter*8:
                value_counts[inter*8] += 1
            elif value <= inter*9:
                value_counts[inter*9] += 1
            else:
                value_counts[inter*10] += 1
        #print(value_counts)  
        entropy = 0
        for count in value_counts.values():
            proportion = count / total_values
            #print('count: ' + str(count))  
            #print('total_values: ' + str(total_values))  
            #print('proportion: ' + str(proportion))
            if proportion > 0:
                entropy -= proportion * math.log2(proportion)
    else:
        for value in values:
            if value not in value_counts:
                value_counts[value] = 1
            else:
                value_counts[value] += 1

        entropy = 0
        for count in value_counts.values():
            proportion = count / total_values
            entropy -= proportion * math.log2(proportion)

    return entropy

nomeAC = "treino_sinais_vitais_com_label.txt"
nomeAS = "treino_sinais_vitais_sem_label.txt"
testePequeno = "textePequeno.txt"
df = pd.read_csv(testePequeno)
df = df.drop('i', axis=1)
#print(df.head())
cQPA = df['qPA'].to_list()
cPulso = df['pulso'].to_list()
cResp = df['resp'].to_list()
cGravid = df['gravid'].to_list()
cClasse = df['classe'].to_list()
#print(coluna)
print('---------------------------------------------------------------')

# Exemplo de valores float em uma coluna
column_values = [0.123456, 0.987654, 0.543210, 0.876543]

# Calcula a entropia
entropyQPA = calculate_entropy(cQPA)
entropyPulso = calculate_entropy(cPulso)
entropyResp = calculate_entropy(cResp)
entropyGravid = calculate_entropy(cGravid)
entropyClasse = calculate_entropy(cClasse)
# Calcula a entropia2
entropyQPA2 = calculate_entropy2(cQPA, -10, 10)
entropyPulso2 = calculate_entropy2(cPulso, 0, 200)
entropyResp2 = calculate_entropy2(cResp, 0, 22)
entropyGravid2 = calculate_entropy2(cGravid, 0, 100)
entropyClasse2 = calculate_entropy2(cClasse, 1, 4)
print(f"================================================")
print(f"A entropia da cQPA e: {entropyQPA:.6f}")
print(f"A entropia2 da cQPA e: {entropyQPA2:.6f}")
print(f"================================================")
print(f"A entropia da pulso e: {entropyPulso:.6f}")
print(f"A entropia2 da pulso e: {entropyPulso2:.6f}")
print(f"================================================")
print(f"A entropia da resp e: {entropyResp:.6f}")
print(f"A entropia2 da resp e: {entropyResp2:.6f}")
print(f"================================================")
print(f"A entropia da gravid e: {entropyGravid:.6f}")
print(f"A entropia2 da gravid e: {entropyGravid2:.6f}")
print(f"================================================")
print(f"A entropia da classe e: {entropyClasse:.6f}")
print(f"A entropia2 da classe e: {entropyClasse2:.6f}")
print(f"================================================")


