from numpy import exp
from random import uniform
from csv import reader
from sys import maxsize

# Sigmóide
def sigmoid(v):
  return 1 / (1 + exp(-v))

# Derivada da sigmóide
def devsigmoid(v):
  return v * (1 - v)

# Ligações
def connect(unit1,unit2):
  connects = []
  
  for i in range(unit1):
    connects.append([0]*unit2)

  return connects

# Inicializar os pesos entre -1 e 1 
def initWeights(weights):
  for i in range(len(weights)):
    for j in range(len(weights[0])): 
      weights[i][j] = uniform(-1, 1)

"""
# Inicializar os biases com valor 0
def initbiases(biases):
    for i in range(len(biases)):
        for j in range(len(biases[0])):
            biases[i][j] = 0
"""

# Classe da rede neuronal
class NeuralNetwork:
  def __init__(self, inputUnits, hiddenUnits):
    self.numberInput = inputUnits +1 
    self.numberHidden = hiddenUnits 
    self.numberOutput = 1
    self.inputLayer = [0] * self.numberInput 
    self.hiddenLayer = [0] * self.numberHidden 
    self.outputLayer = [0] * self.numberOutput  
    self.inputWeights = connect(self.numberInput, self.numberHidden)
    self.outputWeights = connect(self.numberHidden, self.numberOutput)
    initWeights(self.inputWeights)
    initWeights(self.outputWeights)
    self.inputUpdate = connect(self.numberInput, self.numberHidden)
    self.outputUpdate = connect(self.numberHidden, self.numberOutput)

  # Construção da rede feed-forward
  def feedForward(self, inputs):
    for i in range(self.numberInput-1):
      self.inputLayer[i] = inputs[i]
      
    for j in range(self.numberHidden):
      sum = 0.0
      
      for i in range(self.numberInput): 
        sum += self.inputLayer[i] * self.inputWeights[i][j]
        
      self.hiddenLayer[j] = sigmoid(sum)
      
    for k in range(self.numberOutput):
      sum = 0.0
      
      for j in range(self.numberHidden): 
        sum += self.hiddenLayer[j] * self.outputWeights[j][k]
        
      self.outputLayer[k] = sigmoid(sum)
  
    return self.outputLayer

  # Algoritmo backpropagation
  def backPropagation(self, targets, lrate):
    output_deltas = [0] * self.numberOutput
  
    for k in range(self.numberOutput):
      output_deltas[k] = (targets[k] - self.outputLayer[k]) * devsigmoid(self.outputLayer[k])
    
    hidden_deltas = [0] * self.numberHidden
    
    for j in range(self.numberHidden):
      err = 0.0
      
      for k in range(self.numberOutput):
        err += output_deltas[k] * self.outputWeights[j][k]
        
      hidden_deltas[j] = devsigmoid(self.hiddenLayer[j]) * err 
  
    for j in range(self.numberHidden):
      for k in range(self.numberOutput):
        change = output_deltas[k] * self.hiddenLayer[j]
        self.outputUpdate[j][k] = change
        self.outputWeights[j][k] += (lrate * change)
     
    for i in range(self.numberInput):
      for j in range(self.numberHidden):
        change = hidden_deltas[j] * self.inputLayer[i]
        self.inputUpdate[i][j] = change
        self.inputWeights[i][j] += (lrate * change)

    maxerr = 0.0
    
    for k in range(len(targets)):
      if(maxerr < targets[k] - self.outputLayer[k]):
        maxerr = targets[k] - self.outputLayer[k]

    return maxerr

  # Treinar a rede neuronal
  def train(self, dataset, epochs, lrate):
    flag = 0;
    epoch = 0;

    while flag == 0:
      epoch += 1
      
      for d in dataset:
        inputs = d[0]
        targets = d[1]
        self.feedForward(inputs)
        loss = self.backPropagation(targets, lrate)

      if loss < 0.05:
        flag = 1
    
    #print('Épocas de treino necessárias: ' + str(epoch))
    #print('Perda: ' + str(loss))

  # Testar a rede neuronal
  def test(self, data):
    inputs = data
    output = self.feedForward(inputs)

    if (output[0] >= 0.5):
      print('1')
    else:
      print('0')

"""
# Leitura do dataset de treino 
def readTrainDataset(file):
    data = []

    try:
        dataset = open(file, 'r')

    except IOError:
        print("Não foi possível abrir o ficheiro", file)
        exit()

    datareader = reader(dataset, delimiter = ',')

    for line in datareader:
        data.append([[int(line[0]), int(line[1]), int(line[2]), int(line[3])], int(line[4])])

    return data
"""

"""

Não necessário

# Leitura do dataset de teste
def readTestDataset(file):
    data = []

    try:
        dataset = open(file, 'r')

    except IOError:
        print("Não foi possível abrir o ficheiro" , file)
        exit()

    datareader = reader(dataset, delimiter = ',')

    for line in datareader:
        data.append([int(line[0]), int(line[1]), int(line[2]), int(line[3])])

    return data
"""

"""
# ------ Código de teste ------ #

print('Esta rede neuronal tem como objetivo testar a paridade de um número em binário. Para isso, terá de incluir um dataset para treinar a rede neuronal.\n')

# Dataset de treino
trainSet = input('Escolha um ficheiro em formato .csv para treinar a rede neuronal\n')

# Verificar se o ficheiro tem o formato .csv
while not trainSet.endswith('.csv'):
    print('Por favor inclua um ficheiro com a extensão .csv como exemplificado: teste.csv')
    trainSet = input()

print('A treinar a rede neuronal...\n')

NN = NeuralNetwork(4, 4)
trainData = readTrainDataset(trainSet)

NN.train(trainData, maxsize, 0.05)

num = input('Insira um número em binário com 4 dígitos: ')
numlist = []

for i in range(len(num)):
    ntest = int(test[i])
    numlist.append(ntest)

NN.test(numlist)
"""

"""
Não é ncessário, é só para introduzir um número

# Dataset de teste
testSet = input('Agora escolha um ficheiro em formato .csv para testar a rede neuronal\n')

# Verificar se o ficheiro tem o formato .csv
while not testSet.endswith('.csv'):
    print('Por favor inclua um ficheiro com a extensão .csv como exemplificado: teste.csv\n')
    testSet = input()

testData = readTestDataset(testSet)

for line in testData:
    print('Para o número ' + line[0] + '' + line[1] + '' + line[2] + '' + line[3] + ', o output foi: ')
    NN.test(testData)
"""

# Função principal
def exec():
  print('Esta rede neuronal tem como objetivo testar se um número binário tem um número par de 1s. O output será 1 se o número dado tiver número par de 1s, ou 0, caso contrário. Para isso, terá de incluir um dataset para treinar a rede neuronal\n')
  trainSet = input('Escolha um ficheiro em formato .csv para treinar a rede neuronal\n')

  while not trainSet.endswith('.csv'):
    print('Por favor inclua um ficheiro com a extensão .csv')
    trainSet = input()

  trainData = []

  try:
    with open(trainSet, 'r') as trainfile:
      datareader = reader(trainfile, delimiter=',')
    
      for row in datareader:
        trainData.append([[int(row[0]),int(row[1]),int(row[2]),int(row[3])], [int(row[4])]])

  except IOError:
    print('Não foi possível abrir o ficheiro', trainSet, ': ficheiro inexistente')
    exit()

  print('A treinar a rede neuronal...')
    
  NN = NeuralNetwork(4, 4)
  NN.train(trainData, maxsize, 0.05)
  test = input('Insira um número binário com 4 dígitos: ')
  numtest = [int(test[0]),int(test[1]),int(test[2]),int(test[3])]
  NN.test(numtest) 

exec()

