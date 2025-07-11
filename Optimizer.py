import numpy as np
'''
    Define a class for optimization problem. It supports SGD, Adam and weight decay
    
    Parameters:
        mode: ‘sgd’ or ‘adam’

        learning rate: learning rate

        momentum: momentum term of SGD

        weight_decay: weight decay coefficient (L2 regularisation)

        beta1, beta2: Adam's first and second order momentum decay coefficients.

        epsilon: small constant to prevent division by zero (used in Adam)
'''
class Optimizer:
    def __init__(self, mode='sgd', learning_rate=0.01, momentum=0.9, weight_decay=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.mode = mode
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # For momentum
        self.velocity_weights = {}
        self.velocity_biases = {}

        # For Adam
        self.m = {}
        self.v = {}
        self.t = {}
    # Determining what optimisation class to use
    def update(self, layer_id, weights, biases, grad_weights, grad_biases):
        # Apply weight decay (L2 regularization)
        grad_weights += self.weight_decay * weights

        if self.mode == 'sgd':
            return self._sgd(layer_id, weights, biases, grad_weights, grad_biases)
        elif self.mode == 'adam':
            return self._adam(layer_id, weights, biases, grad_weights, grad_biases)
        else:
            raise ValueError(f"Unsupported optimizer mode: {self.mode}")

    # SGD
    def _sgd(self, layer_id, weights, biases, grad_weights, grad_biases):
        # SGD
        if self.momentum == 0:
            weights -= self.lr * grad_weights
            biases -= self.lr * grad_biases
        else:
            # SGD + Momentum
            if layer_id not in self.velocity_weights:
                self.velocity_weights[layer_id] = np.zeros_like(weights)
                self.velocity_biases[layer_id] = np.zeros_like(biases)

            # Accelerate the gradient to help jump out of local minima or saddle points
            self.velocity_weights[layer_id] = self.momentum * self.velocity_weights[layer_id] - self.lr * grad_weights
            self.velocity_biases[layer_id] = self.momentum * self.velocity_biases[layer_id] - self.lr * grad_biases

            weights += self.velocity_weights[layer_id]
            biases += self.velocity_biases[layer_id]

        return weights, biases

    # Adam
    def _adam(self, layer_id, weights, biases, grad_weights, grad_biases):
        if layer_id not in self.m:
            self.m[layer_id] = {'w': np.zeros_like(weights), 'b': np.zeros_like(biases)}
            self.v[layer_id] = {'w': np.zeros_like(weights), 'b': np.zeros_like(biases)}
            self.t[layer_id] = 0

        self.t[layer_id] += 1
        # First order momentum update
        m_w = self.m[layer_id]['w'] = self.beta1 * self.m[layer_id]['w'] + (1 - self.beta1) * grad_weights
        m_b = self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * grad_biases
        # Second-order momentum update
        v_w = self.v[layer_id]['w'] = self.beta2 * self.v[layer_id]['w'] + (1 - self.beta2) * (grad_weights ** 2)
        v_b = self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * (grad_biases ** 2)
        # Bias correction
        m_w_hat = m_w / (1 - self.beta1 ** self.t[layer_id])
        m_b_hat = m_b / (1 - self.beta1 ** self.t[layer_id])
        v_w_hat = v_w / (1 - self.beta2 ** self.t[layer_id])
        v_b_hat = v_b / (1 - self.beta2 ** self.t[layer_id])
        # Parameter update
        weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        biases -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return weights, biases
