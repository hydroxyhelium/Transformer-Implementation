import tensorflow as tf
import numpy as np

## implement multihead attention mechanism

class Transformer:
    ## X refers to the input sequence of tensor embeddings (batch, row, columns) in this format
    def __init__(self, X):    
        self.X = X
        b,t,k = X.shape
        self.get_query = tf.random.normal([k,k], 0 , 1)
        self.get_keys = tf.random.normal([k,k], 0 , 1)
        self.get_values = tf.random.normal([k,k], 0 , 1) 
    
    def get_matrix(self):
        self.query = tf.linalg.matmul(self.get_query, self.X)
        self.keys = tf.linalg.matmul(self.get_keys, self.X)
        self.values = tf.linalg.matmul(self.get_values, self.X) 

    def forward(self):
        b, t, k = self.X.shape
        self.output = tf.zeros([b,t,k])
        self.keys = tf.zeros([])

        for i in range(t):
            temp_q = self.query[:,i,:]            

            for j in range(b):
                key_vectors = self.keys[j, :, :]
                tf.linalg.matmul(key_vectors, temp_q)
                
        return
    
    def get_output(self):
        self.output = tf.math.multiply(self.XXt, self.X)
        return self.output 

