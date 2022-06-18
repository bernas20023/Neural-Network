from numpy import exp
from csv import reader
from random import uniform

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

# Inicializar os pesos com números aleatórios uniformes entre -1 e 1
def initweights(weights):
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j] = uniform(-1, 1)

# Inicializar os biases com números aleatórios uniformes entre -1 e 1
def initbiases(biases):
    for i in range(len(biases)):
        for j in range(len(biases[0])):
            biases[i][j] = uniform(-1, 1)

# Treinar a rede neuronal
def train():
    return 0

# Testar a rede neuronal
def test():
    return 0

# Algoritmo Feed-Forward
def feedforward():
    return 0

# Algoritmo Backpropagation
def backpropagation():
    return 0

# Calcular o erro absoluto
def abserr(x0, x):
    return x0-x

# Leitura do dataset 
def read_dataset(file):
    data = []

    try:
        dataset = open(file, 'r')

    except IOError:
        print("Não foi possível abrir o ficheiro", file)
        exit()

    datareader = reader(file, delimiter = ',')

    
# ---------- Código de teste ---------- #

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


