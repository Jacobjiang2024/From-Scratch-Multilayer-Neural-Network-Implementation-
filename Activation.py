import numpy as np
'''
    There are several activations for constructing Neural Networks:
    
    - ReLu activation function
    
    - Softmax activation function 
    
    - ReLU_derivative: Used for ReLU derivative calculation in backpropagation.
'''


    # ReLU activation function
def ReLU(inputs):
    # Values less than or equal to 0 become 0, and values greater than 0 remain unchanged
    return np.maximum(0, inputs)

def ReLU_derivative(x):
    # The derivative is 1 when x > 0 and 0 otherwise.
    return (x > 0).astype(float)

    # Tanh actiavtion function
def Tanh(x):
    return np.tanh(x)

    # Tanh derivative function (based on tanh(x) itself)
def Tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t ** 2
def Softmax(inputs):
    # Subtract the largest number in this row to prevent exponential explosion
    x_shifted = inputs - np.max(inputs, axis=1, keepdims=True)
    exp_vals = np.exp(x_shifted) #Perform exponential operations
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True) # Calculating Probability

