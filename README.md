# From-Scratch Multilayer Neural Network Implementation

Built a fully functional multi-layer neural network (MLP) from scratch using only Numpy, manually implementing forward propagation, backpropagation, gradient descent, and weight initialization.

---

## Project Structure
```
.
├── Activation.py           # Activation functions (ReLU, Tanh, Softmax, etc.)
├── Data.py                 # Data loading and preprocessing
├── Layer.py                # Definition of individual network layers
├── Lossfunction.py         # Cross entropy with label smoothing
├── Network.py              # Neural network class: forward, backward, training logic
├── Optimizer.py            # SGD, Adam, momentum, weight decay, gradient clipping
├── Utility.py              # Helper functions (metrics, plotting, etc.)
├── Plot.py                 # Visualization utilities
├── Ablation_studies.py     # Ablation study (different learning rates and batch size, etc.)
├── main.py                 # Entry point for final training and evaluation
└── README.md               # Project documentation
```

---

## How to Run

### Step 1: Install Dependencies
This project requires only NumPy and Matplotlib:
```bash
pip install numpy matplotlib
```

### Step 2: Prepare Data
Ensure the dataset is available in `.npy` format and loaded correctly in `Data.py`:
```python
train_data.npy, train_label.npy
test_data.npy,  test_label.npy
```

### Step 3: Train and Evaluate the Model
You can run:
```bash
python main.py
```
Inside `main.py`, the script will:
1. Train the model using early stopping according to the accuracy of the validation set to judge.
2. Save the best weights and biases.
3. Evaluate on the test set.
4. Print out final test accuracy, F1 score, and loss and train time.

---

## Model Configuration
The model can be configured in `main.py`:
```python
NETWORK_SHAPE = [128, 512, 512, 256, 128, 64, 10]
DROPOUT_RATE = 0.2
BATCH_SIZE = 128
MAX_EPOCH = 60
PATIENCE = 10
ACTIVATION = 'ReLU'
opt = {
    'SGD+Momentum': Optimizer(mode='sgd', learning_rate=0.01, momentum=0.95),
}
```

---
