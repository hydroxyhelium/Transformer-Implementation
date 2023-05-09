import tensorflow as tf
import numpy as np

## implement multihead attention mechanism

class Transformer:
    ## X refers to the input sequence of tensor embeddings (batch, row, columns) in this format
    def __init__(self, X , heads=1):    
        self.X = X
        self.heads = heads
        t,k = X.shape

        assert (k%heads)==0 

        self.get_query = tf.random.normal([k,k], 0 , 1)
        self.get_keys = tf.random.normal([k,k], 0 , 1)
        self.get_values = tf.random.normal([k,k], 0 , 1) 
    
    def get_matrix(self):
        self.query = tf.linalg.matmul(self.get_query, self.X)
        self.keys = tf.linalg.matmul(self.get_keys, self.X)
        self.values = tf.linalg.matmul(self.get_values, self.X) 

    def forward(self):
        t, k = self.X.shape
        self.get_matrix()
        temp = tf.linalg.matmul(self.query, self.keys)
        self.outputs = tf.linalg.matmul(temp, self.values)
        
        return self.outputs

