import numpy as np

'''
predicted：The softmax probability matrix
real:The true label, which is an integer vector of shape (N,)
'''

def cross_entropy_loss(predicted, real, epsilon=0.1):
    eps = 1e-12 # Defines a very small value to prevent log(0) from evaluating to -inf (negative infinity).
    predicted = np.clip(predicted, eps, 1. - eps) # All values are clamped to the range [eps, 1 - eps]
    return -np.mean(np.log(predicted[np.arange(len(real)), real])) # Calculate the cross entropy loss for each sample

#Loss function 1
def precise_loss_function(predicted,real):
    eps = 1e-12 # Defines a very small value to prevent log(0) from evaluating to -inf (negative infinity).
    N = predicted.shape[0] # Get the number of samples
    predicted = np.clip(predicted, eps, 1. - eps) # All values are clamped to the range [eps, 1 - eps]
    return -np.mean(np.log(predicted[np.arange(N), real])) # Calculate the cross entropy loss for each sample


# Return prediction error for each sample
def loss_function(predicted, real):
    predicted_labels = np.argmax(predicted, axis=1)  # Take the category corresponding to the maximum probability in each row
    return (predicted_labels != real).astype(np.float32)
