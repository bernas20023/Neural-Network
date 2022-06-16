from numpy import exp
from csv import reader

# Representação da rede neuronal
class NeuralNetwork():
    def __init__(self, inputUnits, hiddenUnits):
        self.inputUnits = inputUnits
        self.hiddenUnits = hiddenUnits
        self.outputUnits = 1
        
# Função sigmóide
def sigmoid(value):
    return 1/(1+exp(-value))
    
# Derivada da sigmóide
def devsigmoid(value):
    return exp(value)/(pow(exp(value)+1, 2))

# Leitura do dataset 
def read_dataset(file):
    try:
        dataset = open(file, 'r')

    except IOError:
        print("Não foi possível abrir o ficheiro", file)
        exit()

    csvreader = reader(file, delimiter = ',')


# ---------- Código para testar e treinar a rede neuronal ---------- #

# Dataset de treino
trainSet = input('Escolha um ficheiro em formato .csv para treinar a rede neuronal\n')

# Verificar se o ficheiro tem o formato .csv
while not trainSet.endswith('.csv'):
    print('Por favor inclua um ficheiro com a extensão .csv como exemplificado: teste.csv')
    trainSet = input()

read_dataset(trainSet)

# Dataset de teste
testSet = input('Escolha um ficheiro em formato .csv para testar a rede neuronal\n')

# Verificar se o ficheiro tem o formato .csv
while not testSet.endswith('.csv'):
    print('Por favor inclua um ficheiro com a extensão .csv como exemplificado: teste.csv')
    testSet = input()

read_dataset(testSet)


