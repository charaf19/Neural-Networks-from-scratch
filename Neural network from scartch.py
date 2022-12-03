import numpy as np
np.random.seed(0)
X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,0.8]
     ]
#inputs= [0,2,-1,3.3,-2.7,1.1,2.2,-100 ]
#output = []
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights= 0.10*np.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)