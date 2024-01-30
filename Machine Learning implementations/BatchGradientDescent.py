import numpy as np
import matplotlib.pyplot as plt

class BatchGradientDescent:

  def __init__(self,train_x,train_y,learning_rate=0.01,epochs=100):
    self.epochs = epochs
    self.train_x = np.array(train_x)
    self.train_y = np.array(train_y)
    self.preprocess_the_data()
    self.learning_rate = learning_rate
    self.parameters = []
    for i in range(0,self.number_of_features+1):
      self.parameters.append(0)
    self.parameters = np.array(self.parameters)
    self.parameters = self.parameters.astype(np.float32)
    self.parameters = self.parameters.T


  def preprocess_the_data(self):
    self.train_x = self.train_x.astype(np.float32)
    self.train_y = self.train_y.astype(np.float32)
    self.number_of_features = 1 if len(self.train_x.shape) == 1 else self.train_x.shape[0]
    self.number_of_train_examples = len(self.train_y)
    ones = np.ones(self.number_of_train_examples)
    ones = ones.astype(np.float32)
    self.train_x = np.vstack((ones, self.train_x))
    self.train_x = self.train_x.T
    self.train_y = self.train_y.T



  def __str__(self):
    return f"Number of features : {self.number_of_features}\n"+f"Number of examples : {self.number_of_train_examples}\n"+f"Learning rate : {self.learning_rate}\n"+f"Prameters : {self.parameters}"

  def hypothesis_function(self,x):
    return sum(self.parameters * x)


  def cost_functionJ(self):
    sum = 0
    for i in range(0,self.number_of_train_examples):
      val = (self.hypothesis_function(self.train_x[i]) - self.train_y[i])
      sum = sum + val*val

    return sum / 2

  def find_optimal_parameters(self):
    X = self.train_x
    self.parameters = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),self.train_y)

  def train_the_model(self,epoch_flag = False):
    if(epoch_flag):
      print("Training started")
    epoch = 0
    while(epoch != self.epochs):
      if epoch_flag:
        print(f"Epoch : {epoch + 1}")
      for j in range(0,self.number_of_features+1):
        extracted_value = 0
        for i in range(0,self.number_of_train_examples):
          extracted_value = extracted_value + (self.hypothesis_function(self.train_x[i]) - self.train_y[i]) * self.train_x[i][j]

        self.parameters[j] = self.parameters[j] - self.learning_rate * extracted_value
      epoch = epoch + 1
      if epoch_flag:
        print(self.parameters)

    if(epoch_flag):
      print(f"Training ended with {self.epochs} epochs")


  def estimate_value(self,x):
    x = np.array(x)
    x = np.insert(x,0,1)
    return self.hypothesis_function(x)


  def evaluate_the_model(self):
    return self.cost_functionJ()