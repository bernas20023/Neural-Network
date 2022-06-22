from numpy import exp
from csv import reader
from random import uniform
from sys import maxsize

# Função sigmóide
def sigmoid(value):
    return 1/(1+exp(-value))
    
# Derivada da sigmóide
def devsigmoid(value):
    return value*(1-value)

# Inicializar os pesos com números aleatórios uniformes entre -1 e 1
def initweights(weights):
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j] = uniform(-1, 1)

# Inicializar os biases com valor 0
def initbiases(biases):
    for i in range(len(biases)):
        for j in range(len(biases[0])):
            biases[i][j] = 0

# Ligações entre as unidades
def connections(unit1, unit2):
    connects = []

    for i in range(unit1):
        connects.append([0]*unit2)

    return connects

# Classe que contém a representação da rede neuronal
class NeuralNetwork():
    # Representação da rede neuronal
    def __init__(self, inputUnits, hiddenUnits):
        # Número de nós da camada de input, da camada escondida e da camada de output
        self.inputUnits = inputUnits
        self.hiddenUnits = hiddenUnits
        self.outputUnits = 1
        # Definição das matrizes dos pesos para cada camada
        self.inputLayer = [0] * self.inputUnits
        self.hiddenLayer = [0] * self.hiddenUnits
        self.outputLayer = [0] * self.outputUnits
        # Definição das matrizes dos pesos das ligações
        self.inputWeights = connections(self.inputUnits, self.hiddenUnits)
        self.outputWeights = connections(self.hiddenUnits, self.outputUnits)
        # Inicializar os pesos
        initweights(self.inputWeights)
        initweights(self.outputWeights)
        # Matrizes de atualização dos pesos
        self.updateInLayer = connections(self.inputUnits, self.hiddenUnits)
        self.updateOutLayer = connections(self.hiddenUnits, self.outputUnits)

    # Criação da rede neuronal feed-forward
    def feedForward(self, inputs):
        for i in range(self.inputUnits):
            self.inputLayer[i] = inputs[i]

        for i in range(self.hiddenUnits):
            sum = 0.0

            for j in range(self.inputUnits):
                sum += self.inputLayer[j] * self.inputWeights[i][j]

            self.hiddenLayer[j] = sigmoid(sum)

        for i in range(self.outputUnits):
            sum = 0.0

            for j in range(self.hiddenUnits):
                sum += self.hiddenLayer[j] * self.outputWeights[j][i]

            self.outputLayer[i] = sigmoid(sum)

        return self.outputLayer
                
    # Algoritmo backpropagation
    def backpropagation(self, targets, lrate):
        # Cálculo dos deltas de output
        for i in range(self.outputUnits):
            outdeltas = devsigmoid(self.outputLayer[i]) * (targets[i] - self.outputLayer[i])

        for j in range(self.hiddenUnits):
            err = 0.0

            for k in range(self.outputUnits):
                err += outdeltas * self.outputWeights[j][k]

            hiddeltas[j] = devsigmoid(self.hiddenLayer[j]) * err

        for j in range(self.hiddenUnits):
            for k in range(self.outputUnits):
                updweights = outdeltas[k] * self.hiddenLayer[j]
                self.updateOutLayer[j][k] = updweights
                self.outputWeights[j][k] += (lrate * updweights)

        for i in range(self.inputUnits):
            for j in range(self.hiddenUnits):
                updweights = hiddeltas[j] * self.inputLayer[i]
                self.updateInWeights[i][j] = updweights
                self.inputWeights[i][j] += (lrate * updweights)

        maxerr = 0.0

        for k in range(len(targets)):
            if(maxerr < targets[k] - self.outputLayer[k]):
                maxerr = targets[k] - self.outputLayer[k]

        return maxerr

    # Treinar a rede neuronal
    def train(self, dataset, epochs, lrate):
        flag = 0
        epoch = 0

        while flag == 0:
            epoch += 1

            for line in dataset:
                inputs = line[0]
                targets = line[1]
                self.feedForward(inputs)
                loss = self.backPropagation(targets, lrate)

            if loss < 0.05:
                flag = 1

    # Testar a rede neuronal
    def test(self, dataset):
        output = self.feedForward(dataset)

        if(output[0] >= 0.5):
            print('1')

        else:
            print('0')

# Leitura do dataset de treino 
def readTrainDataset(file):
    data = []

    try:
        dataset = open(file, 'r')

    except IOError:
        print("Não foi possível abrir o ficheiro", file)
        exit()

    datareader = reader(file, delimiter = ',')

    for line in datareader:
        data.append([[int(line[0]), int(line[1]), int(line[2]), int(line[3])], int(line[4])])

    return data

# Leitura do dataset de teste
def readTestDataset(file):
    data = []

    try:
        dataset = open(file, 'r')

    except IOError:
        print("Não foi possível abrir o ficheiro" , file)
        exit()

    datareader = reader(file, delimiter = ',')

    for line in datareader:
        data.append([int(line[0]), int(line[1]), int(line[2]), int(line[3])])

    return data
    
# ---------- Código de teste ---------- #

print('Esta rede neuronal tem como objetivo testar a paridade de um número em binário. Para isso, terá de incluir um dataset para treinar a rede neuronal e um dataset para testar. Para cada linha do dataset de teste, será impressa uma linha com 0 ou 1, consoante o número dado na linha seja par ou não\n')

# Dataset de treino
trainSet = input('Escolha um ficheiro em formato .csv para treinar a rede neuronal\n')

# Verificar se o ficheiro tem o formato .csv
while not trainSet.endswith('.csv'):
    print('Por favor inclua um ficheiro com a extensão .csv como exemplificado: teste.csv')
    trainSet = input()

NN = NeuralNetwork(4, 4)
traindata = readTrainDataset(trainSet)
NN.train(trainData, maxsize, 0.05)

# Dataset de teste
testSet = input('Agora escolha um ficheiro em formato .csv para testar a rede neuronal\n')

# Verificar se o ficheiro tem o formato .csv
while not testSet.endswith('.csv'):
    print('Por favor inclua um ficheiro com a extensão .csv como exemplificado: teste.csv')
    testSet = input()

testData = readTestDataset(testSet)

for line in testData:
    NN.test(testData)


