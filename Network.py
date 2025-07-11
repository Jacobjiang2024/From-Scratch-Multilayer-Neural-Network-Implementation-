from Activation import *
from Lossfunction import *
from Utility import *
import numpy as np
import Layer as Layer
import time
'''
Function：
    Network: Build network construction with inputting value of network_shape
    
    Network_forward: Complete the forward propagation and calculate the activation outputs for each layer
    
    Network_backward: Complete the backward propagation
    
    get_final_layer_preAct_demands: Calculate Softmax error of output layer
    
    one_batch_train: Doing train with a little batch data 
    
    train: Train the whole model (multiple batches + multiple rounds)
'''
class Network:

    def __init__(self, network_shape, dropout_rate=0.3, optimizer=None, activation='ReLU',use_batchnorm=True):
        self.shape = network_shape # Define the number of neurons for each layer
        self.optimizer = optimizer
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        # Build each layer of the neural network with Dropout for the hidden layer and no Dropout for the output layer.
        self.layers = [
            Layer.Layer(network_shape[i], network_shape[i + 1],
                        dropout_rate=dropout_rate if i < len(network_shape) - 2 else 0.0,
                        layer_id=i, optimizer=self.optimizer,use_batchnorm=self.use_batchnorm)
            for i in range(len(network_shape) - 1)
        ]
        print(f" Network initialized with BatchNorm: {getattr(self.layers[0], 'use_batchnorm', self.use_batchnorm)}")
    # Forward Propagation
    def network_forward(self, inputs):
        self.z_values = [] # Record each layer output
        outputs = [inputs] # Input to upper network layer output
        # Loopy all network layers
        for i in range(len(self.layers)):
            z = self.layers[i].layer_forward(outputs[i]) # calculate linear result
            self.z_values.append(z) # Record linear result
            # Use ReLU for all layers except the last one; use Softmax for the final layer
            if i < len(self.layers) - 1:
                output = {'ReLU': ReLU, 'Tanh': Tanh}.get(self.activation, ReLU)(z)
            else:
                output = Softmax(z)
            outputs.append(output) #Record network layer output
        return outputs

    # Backward propagation
    '''
    Parameters：
        layer_outputs: outputs of all layers obtained by forward propagation (both input and output layers).

        target_vector: target labels (real values).

        batch_size: amount of data processed in each round.

        learning_rate: learning rate, used to update the weights.
    '''
    def network_backward(self, layer_outputs, target_vector, batch_size):
        # Calculate the error signal of the output layer
        preAct_demands = self.get_final_layer_preAct_demands(layer_outputs[-1], target_vector)
        # Traverse each layer of the network in reverse order (starting with the output layer)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]  # return the front layer
            z = self.z_values[i] # return result of z = Wx + b for the front layer
            activation_fn = self.activation if i < len(self.layers) - 1 else 'softmax' # Determining whether to use ReLU or Tanh derivatives
            # Backpropagating the gradient of each layer
            preAct_demands, _= layer.layer_backward(z, preAct_demands, batch_size, activation=activation_fn)
            # Update the weights using the gradients just calculated.
            layer.update()

    # Calculate error between predict value and true value
    def get_final_layer_preAct_demands(self, predicted, real):
        N, C = predicted.shape # Getting the output shape
        one_hot = np.zeros_like(predicted) # Construct a 0 matrix of the same shape
        # Assign a value of 1 to the position of the correct category on each line
        one_hot[np.arange(N), real] = 1
        # Return Error
        return predicted - one_hot


    # Define mini-batch for train
    '''
    Parameters：
        batch: Input data for a mini-batch
        
        target: The real label corresponding to the batch
        
        batch_size: Number of samples in the batch
        
        learning_rate: learning rate, used to update the weights.
    '''
    def one_batch_train(self, batch, target, batch_size):
        outputs = self.network_forward(batch)  # get the output of all layers
        loss = cross_entropy_loss(outputs[-1], target) #  Calculation of cross-entropy loss
        self.network_backward(outputs, target, batch_size) # backward propagation and update weights
        preds = classify(outputs[-1]) # Obtaining categorical predictions
        acc = accuracy(preds, target) #Calculation accuracy
        return loss, acc, preds


    # Multi-batch training
    '''
        Parameters：
        
        data, target: Training data and labelling
        
        batch_size: Number of samples in the batch
        
        learning_rate: learning rate, used to update the weights.
        
        epochs: Number of epochs to train the network
        
        valid_data, valid_target: Validation set (optional)
        
        patience: Number of tolerances for early stopping
    '''
    def train(self, data, target, batch_size , epochs=30, valid_data=None, valid_target=None, patience=8):
        history = {'loss': [], 'acc': [],'f1': [], 'val_acc': [],'val_loss':[], 'val_f1':[],'epoch_time': []} # Used to record loss, acc, val_acc for each round (can be used to draw graphs)
        best_val_acc = 0 # Record the highest accuracy of the validation set
        epochs_no_improve = 0 # Record how many consecutive rounds of the validation set did not lift
        best_params = None # Record the current best model parameters
        # Loopy all data for training
        for epoch in range(epochs):
            start_time = time.time() # recording time
            # Disrupt the training set before the start of each round
            perm = np.random.permutation(len(data))
            data = data[perm]
            target = target[perm]

            batch_preds_all, batch_targets_all = [], [] #recording result for getting F1 score
            batch_losses, batch_accs= [], []# recording the loss, accuracy
            for i in range(0, len(data), batch_size): # Do mini-batch train
                batch = data[i:i + batch_size] # Getting data
                batch_target = target[i:i + batch_size] # Getting labels
                loss, acc, preds = self.one_batch_train(batch, batch_target, batch_size)# Getting the loss and accuracy
                batch_losses.append(loss) # Recording loss
                batch_accs.append(acc) # Recording accuracy
                batch_preds_all.extend(preds)
                batch_targets_all.extend(batch_target)
            epoch_loss = np.mean(batch_losses) # Recording the mean of loss
            epoch_acc = np.mean(batch_accs) # Recording the mean of accuracy
            epoch_f1 = micro_f1_score(np.array(batch_preds_all),np.array(batch_targets_all), num_classes=10)
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)
            history['f1'].append(epoch_f1)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, F1: {epoch_f1:.4f}")
            end_time = time.time()  # recording the ending time
            epoch_duration = end_time - start_time
            history['epoch_time'].append(epoch_duration)  # recording time
            print(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_duration:.2f}s")
            # Early stopping
            # Using validation data to determine if overfitting
            if valid_data is not None and valid_target is not None:
                val_result = self.network_forward(valid_data)
                # calculate f1, loss and acc
                val_f1,val_acc,val_loss =calculate_result(val_result, valid_target)
                history['val_acc'].append(val_acc) # Recording the accuracy of validation dataset
                history['val_loss'].append(val_loss)
                history['val_f1'].append(val_f1)
                print(f"	Validation Loss {val_loss:.4f} , Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
                # Determine whether the model overfitting
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # copy best weights, biases
                    best_params = [(layer.weights.copy(), layer.biases.copy()) for layer in self.layers]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(" Early stopping triggered.")
                        break
            else:
                history['val_acc'].append(None)
                history['val_loss'].append(None)
                history['val_f1'].append(None)
        if best_params:
            for i, layer in enumerate(self.layers): \
                    layer.weights, layer.biases = best_params[i]  # recover the best weights and biases

        return history