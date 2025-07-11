import numpy as np
from Activation import *
from Data import *
'''
    Initialization parameters: weight and bias are randomly generated
    
    Forward propagation calculates output

    Backward propagation calculates gradients

    Update parameters
    
    batchnorm_forward: Forward propagation calculates output with Normalization
    
    batchnorm_backward：Backward propagation calculates gradients
'''
class Layer:
    def __init__(self,n_inputs, n_neurons, dropout_rate=0.0, layer_id=None, optimizer=None, use_batchnorm=True, epsilon=1e-5):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs) # After initialization to standard normal distribution
        self.biases = np.zeros(n_neurons) # Initialize to all 0 vector
        self.dropout_rate = dropout_rate # Randomly "drop" neurons to increase the generalization ability of the model.
        self.dropout_mask = None # A mask of which neurons are "dropped"
        self.optimizer = optimizer #If the optimiser is set, use it to update the parameters
        self.layer_id = layer_id #Used to index the layer in the optimiser
        self.use_batchnorm = use_batchnorm # Whether to use BatchNorm
        self.epsilon = epsilon # Small constants that prevent division by 0

        # Determine whether to use normalization
        if use_batchnorm:
            self.gamma = np.ones(n_neurons)
            self.beta = np.zeros(n_neurons)
            self.running_mean = np.zeros(n_neurons)
            self.running_var = np.ones(n_neurons)

    def layer_forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases #

        # Determine whether to use normalization
        if self.use_batchnorm:
            self.inputs_bn = self.z
            self.z = self.batchnorm_forward(self.z)
        self.output = self.z
        if self.dropout_rate > 0: # Randomly "dropping" neurons
            # Generate a random retention mask
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.dropout_rate).astype(float)
            # Actually masking neuron output
            self.output *= self.dropout_mask
        return self.output

    '''
    Parameters：
        z: pre-activation value in the forward propagation of this layer
        
        afterWeights_demands: error signals from next layer
        
        BATCH_SIZE: number of samples in the current batch
        
        use_relu: Whether this layer uses ReLU or not.
    '''
    def layer_backward(self, z, afterWeights_demands, BATCH_SIZE,activation='ReLU'):
        # Determine if ReLu is used
        if activation == 'ReLU':
            value_derivatives = ReLU_derivative(z)  # Calculates derivative value
        elif activation == 'Tanh':
            value_derivatives = Tanh_derivative(z) # Calculates derivative value
        else:
            value_derivatives = np.ones_like(z) # No effect on softmax / linear layers by default.
            # Determine if dropout is enabled
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            value_derivatives *= self.dropout_mask
        # Chain rule propagation error
        preAct_demands = value_derivatives * afterWeights_demands
        # Determine if Normalization's backpropagation is enabled.
        if self.use_batchnorm:
            preAct_demands = self.batchnorm_backward(preAct_demands)

        # Calculating weights and bias gradients
        self.grad_weights = np.dot(self.inputs.T, preAct_demands) / BATCH_SIZE
        self.grad_biases = np.mean(preAct_demands, axis=0)

        # The error signal is passed to the previous layer
        preWeights_demands = np.dot(preAct_demands, self.weights.T)
        return preWeights_demands, self.grad_weights

    # use optimizer to update weights and biases
    def update(self):
        if self.optimizer and self.layer_id is not None:
            self.weights, self.biases = self.optimizer.update(
                self.layer_id, self.weights, self.biases, self.grad_weights, self.grad_biases
            )
    # Do normalization
    def batchnorm_forward(self, x):
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        self.batch_mu = mu
        self.batch_var = var
        self.x_norm = (x - mu) / np.sqrt(var + self.epsilon) # Standardised input
        out = self.gamma * self.x_norm + self.beta # Restoring the expressive capacity of the network
        # Update runtime mean and variance
        self.running_mean = 0.9 * self.running_mean + 0.1 * mu
        self.running_var = 0.9 * self.running_var + 0.1 * var
        return out

    def batchnorm_backward(self, dout):
        N, D = dout.shape # Getting network shape
        x_mu = self.inputs_bn - self.batch_mu # calculate offset of the mean
        std_inv = 1. / np.sqrt(self.batch_var + self.epsilon) # Calculate the inverse of the standard deviation

        dx_norm = dout * self.gamma # Calculate x deviation
        dvar = np.sum(dx_norm * x_mu, axis=0) * -0.5 * std_inv**3 # Getting derivative of the variance
        dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mu, axis=0) # Getting derivative of the mean

        dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N) # Getting the gradient of x
        dgamma = np.sum(dout * self.x_norm, axis=0) # Calculate learning parameters
        dbeta = np.sum(dout, axis=0)

        self.grad_gamma = dgamma
        self.grad_beta = dbeta
        return dx