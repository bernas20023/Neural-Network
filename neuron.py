import math
import random
import csv
import sys

#sigmoidal function defined nicely 
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# desigmoid: used to calculate the derivative of the sigmoid function for backpropagation
def desigmoid(x):
  return x * (1 - x)

# Function to make a matrix of zeros
def makeMatrix(x,y):
  matrix = []
  for i in range(x):
    matrix.append([0]*y)
  return matrix

# randomize the weights of the  
def matrixRandomizer(matrix, a, b):
  for i in range(len(matrix)):
    for j in range(len(matrix[0])):
      # use a uniform distribution to generate random numbers 
      matrix[i][j] = random.uniform(a,b)


# defenition of NN class
class Network:
  #initialize the neural net
  def __init__(self, inputNodes, hiddenNodes):
    # define the number of nodes in each layer from what we input when we call the functionm
    self.numberInput = inputNodes +1 # inputs
    self.numberHidden = hiddenNodes # hidden 
    self.numberOutput = 1
    # define the matrices for the weights with the number of nodes in each layer
    self.matrixInput = [0] * self.numberInput # input layer
    self.matrixHidden = [0] * self.numberHidden # hidden layer
    self.matrixOutput = [0] * self.numberOutput # output layer
    # define the weight matrixes and biases for the neural net 
    self.inputWeights = makeMatrix(self.numberInput, self.numberHidden)
    self.outputWeights = makeMatrix(self.numberHidden, self.numberOutput)
    # starting weights for the neural net 
    matrixRandomizer(self.inputWeights, -1, 1)
    matrixRandomizer(self.outputWeights, -1, 1)
    # define the change matrixes for the neural net.
    # the change matrixes are used to keep track of the changes in the weights
    self.inputChange = makeMatrix (self.numberInput, self.numberHidden)
    self.outputChange = makeMatrix (self.numberHidden, self.numberOutput)
    
  # Run the network with the provided input
  def run(self, inputs):
    # add the inputs to the input layer
    for i in range(self.numberInput-1):
      self.matrixInput[i] = inputs[i]
    # calculate the hidden layer
    for j in range(self.numberHidden): # for each hidden node
      sum = 0.0 # sum of the weights * inputs
      for i in range(self.numberInput): # for each input for each hidden node
        sum += self.matrixInput[i] * self.inputWeights[i][j] 
      self.matrixHidden[j] = sigmoid(sum)
    # calculate the output layer
    for k in range(self.numberOutput): # for each output node
      sum = 0.0
      for j in range(self.numberHidden): # for each hidden node for each output node
        sum += self.matrixHidden[j] * self.outputWeights[j][k]
      self.matrixOutput[k] = sigmoid(sum)
    # return the output layer
    return self.matrixOutput

  def backPropagate (self, targets, learning_rate):
    # inicializar output deltas
    output_deltas = [0] * self.numberOutput
    # calcular os output deltas
    for k in range(self.numberOutput):
      output_deltas[k] = (targets[k] - self.matrixOutput[k]) * desigmoid(self.matrixOutput[k])
    # inicializar hidden deltas
    hidden_deltas = [0] * self.numberHidden
    # calcular hidden deltas
    for j in range(self.numberHidden):
      error = 0.0
      for k in range(self.numberOutput):
        error += output_deltas[k] * self.outputWeights[j][k]
      hidden_deltas[j] = desigmoid(self.matrixHidden[j]) * error 
    # hidden to output weight changes
    for j in range(self.numberHidden):
      for k in range(self.numberOutput):
        change = output_deltas[k] * self.matrixHidden[j]
        self.outputChange[j][k] = change
        self.outputWeights[j][k] += (learning_rate * change)
    # input to hidden weight changes    
    for i in range(self.numberInput):
      for j in range(self.numberHidden):
        change = hidden_deltas[j] * self.matrixInput[i]
        self.inputChange[i][j] = change
        self.inputWeights[i][j] += (learning_rate * change)
    # calculate the error
    sum = 0.0
    for k in range(len(targets)):
      sum += (targets[k] - self.matrixOutput[k])**2
    return sum/len(targets)

  # function to save the weights of the neural net for documentation
  def saveWeights(self):
    # open the file to save the weights
    with open('weights.csv', 'w', newline='') as csvfile:
      # create a csv writer
      writer = csv.writer(csvfile, delimiter=',')
      # write the weights to the file
      writer.writerow(self.inputWeights)
      writer.writerow(self.outputWeights)
        
  # Function to print the weights of the neural net. Print so that it is easy to read
  def printWeights(self):
    print('Input weights:')
    for i in range(self.numberInput):
      print(self.inputWeights[i])
    print('Output weights:')
    for i in range(self.numberHidden):
      print(self.outputWeights[i])

  # also print the learning rate, final mean square error, and amount of hidden nodes
  def printInfo(self):
    print('Number of input nodes: ' + str(self.numberInput - 1))
    print('Number of hidden nodes: ' + str(self.numberHidden))
    print('Number of output nodes: ' + str(self.numberOutput))
  
  # test the network with provided input
  def test(self, data):
    for d in data:
      inputs = d[0]
      # print input, run the network, and print the output, then print the target
      print('Input:', inputs, '->', self.run(inputs), 'Target:', d[1], "Loss:", self.backPropagate(d[1], 0))
      
  # Train the network with the provided data and the number of iterations and the learning rate 
  def train(self, data, epochs, learning_rate):
    for gen in range(epochs+1):
      for d in data:
        inputs = d[0]
        targets = d[1]
        self.run(inputs)
        loss = self.backPropagate(targets, learning_rate)
      if loss<0.05 :
        print('Generation: ' + str(gen) + ' Loss: ' + str(loss))
        break
#Build Neural Net and Run Some tests
def run():
  # import data from csv file. 
  trainingData = []
  with open('data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
      # the first row is the input, the second is the target.
      trainingData.append([[int(row[0][i]) for i in range(len(row[0]))], [int(row[1])]])
  # input nodes, output nodes. Input should be <= number of bits because each bit is a node
  nn = Network(4, 4)
  nn.train(trainingData, sys.maxsize, 0.05)
  #nn.printWeights()
  # test some data and print the results
  print('Test data:')
  nn.test(trainingData)
  nn.printInfo()

run() # main function
