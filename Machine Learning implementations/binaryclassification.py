# -*- coding: utf-8 -*-
"""BinaryClassification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CUpe2XKyEzTBspZUsnpj70-Pb0CS8RwX
"""

import numpy as np
import math

class BinaryClassification:
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


  def sigmoid_function(self,z):
    return 1 / (1 + math.exp(-z))


  def hypothesis_function(self,x):
    exponent = sum(self.parameters * x)
    return self.sigmoid_function(exponent)


  def estimate_value(self,x):
    x = np.array(x)
    x = np.insert(x,0,1)
    return round(self.hypothesis_function(x))
  

  def load_the_model(self,file_path):

    with open(file_path,'r') as file:
      i = 0
      for line in file:
        self.parameters[i] = float(line.strip())
        i=i+1


  def save_the_model(self,file_path):

    with open(file_path,'w') as file:

      for q in self.parameters:
        record = str(q) + '\n'
        file.write(record)

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
          extracted_value = extracted_value + (self.train_y[i] - self.hypothesis_function(self.train_x[i])) * self.train_x[i][j]

        self.parameters[j] = self.parameters[j] + self.learning_rate * extracted_value
      epoch = epoch + 1
      if epoch_flag:
        print(self.parameters)

    if(epoch_flag):
      print(f"Training ended with {self.epochs} epochs")
