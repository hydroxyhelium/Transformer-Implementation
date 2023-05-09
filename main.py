import tensorflow as tf
import numpy as np

## implement multihead attention mechanism


## An adt for each attention head to store it's set of queries, keys, and values 

class Matrices:
    def __init__(self, dim, k):
        self.dim = dim
        self.k = k
        print(dim)
        self.get_query = tf.random.normal([k,dim], 0, 1)
        self.get_keys = tf.random.normal([k,dim], 0, 1)
        self.get_values = tf.random.normal([k, dim], 0, 1)
        self.query = None
        self.keys = None 
        self.value = None

    ## setting these up for backward propagation
    def set_query(self, new_query):
        self.get_query = new_query
    
    def set_keys(self, new_keys):
        self.get_keys = new_keys

    def set_values(self, new_values):
        self.get_values = new_values

    ## calulating queries, keys, values 
    def caclulate(self, X):
        self.query = tf.linalg.matmul(X, self.get_query)
        self.query = tf.nn.softmax(self.query, 1) / ((self.k)**(0.5))
        self.keys = tf.linalg.matmul(X, self.get_keys)
        self.value = tf.linalg.matmul(X, self.get_values)

    
    def forward(self):
        temp = tf.linalg.matmul(self.query, tf.transpose(self.keys) )
        return tf.linalg.matmul(temp, self.value)

class SelfAttention:
    ## X refers to the input sequence of tensor embeddings (batch, row, columns) in this format
    def __init__(self, X , heads=1):    
        self.X = X
        self.heads = heads
        t,k = X.shape

        assert (k%heads)==0

        dim = k//heads

        array = [Matrices(dim, k) for i in range(heads)]

        self.array = array
    
    def get_matrix(self):
        for i in range(self.heads):
            self.array[i].caclulate(self.X) 

    def forward(self):
        self.outputs = None
        self.get_matrix()
        for i in range(self.heads):
            if(self.outputs is None):
                self.outputs = self.array[i].forward()
            else:
                self.outputs = tf.concat([self.outputs, self.array[i].forward()], 1)

        return self.outputs
    
def main():
    
    ## Testing Single Head Attention
    a = np.array([[1, 2],[4, 5],[6, 7]])
    a = tf.convert_to_tesnor(a)

    tran = SelfAttention(a, 1)
