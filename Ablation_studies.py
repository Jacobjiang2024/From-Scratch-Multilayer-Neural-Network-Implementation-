from Data import *
from Network import *
from Utility import *
from Plot import *
from Optimizer import Optimizer

'''
This script contains various ablation experiments
for a custom neural network built using NumPy.
The experiments test different aspects such as:
   - Network shape
   - Optimizers
   - Dropout usage
   - Batch Normalization
   - Activation functions
   - Learning rate & batch size
   - Weight decay

HOW TO USE:
   - Uncomment one of the function calls in the
     main section to run a specific ablation study.
   - Only ONE study should be run at a time.
'''
data = DataProcessor()
X_train, y_train = data.X_train, data.y_train
X_valid, y_valid = data.X_valid, data.y_valid
X_test, y_test = data.X_test, data.y_test

INPUT_DIM = X_train.shape[1]
DROPOUT_RATE = 0.1
BATCH_SIZE = 128
MAX_EPOCH = 50
PATIENCE = 5
NETWORK_SHAPE = [
        build_shape(INPUT_DIM, [512, 256, 64, 32]),
        build_shape(INPUT_DIM, [512, 256, 128, 64, 32]),
        build_shape(INPUT_DIM, [512, 512, 256, 128, 64]),
        build_shape(INPUT_DIM, [512, 512, 256, 256,128, 64]),
        build_shape(INPUT_DIM, [512, 512, 256, 256,128, 64,32]),
]
optimizers = {
        'SGD': Optimizer(mode='sgd', learning_rate=0.01, momentum=0,weight_decay=0),
        'SGD+Momentum': Optimizer(mode='sgd', learning_rate=0.01, momentum=0.95, weight_decay=0),
        'Adam': Optimizer(mode='adam', learning_rate=0.01, weight_decay=0),
        'Adam+Decay': Optimizer(mode='adam', learning_rate=0.001, weight_decay=0.0001),
}

# Training function for reuse
def run_training(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid , network_shape=NETWORK_SHAPE[2], dropout=0.2,opt=optimizers['SGD'],act = 'ReLU', bn=True,bs =BATCH_SIZE):
    # Runs a full training loop on a given model configuration
    network = Network(network_shape, dropout_rate=dropout, optimizer=opt,activation=act, use_batchnorm=bn)
    history = network.train(
        X_train, y_train, bs,
        epochs=MAX_EPOCH,
        valid_data=X_valid, valid_target=y_valid,
        patience=PATIENCE
    )
    return history
# Ablation: Different Network Architectures
def Ablation_Network_Shape(X_train, y_train, X_valid, y_valid, network_shape,opt = 'SGD',dropout_rate = 0.3):
    optimizer = optimizers[opt]
    histories = {}
    metrics = {}
    print(f"\n Ablation Network Shape : {network_shape}")
    for i, shape in enumerate(network_shape):
        histories[f'\n Shape {i+1}'] = run_training(X_train, y_train, X_valid, y_valid, network_shape[i], dropout_rate,
                                            optimizer)
        name, result = extract_metrics_from_history(f'\n Shape {i+1}', histories[f'\n Shape {i+1}'])
        metrics[name] = result

    # print result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    # draw result
    plot_all_histories(histories)

#  Ablation: Different Optimizers
def Ablation_optimizers(X_train, y_train, X_valid, y_valid,network_shape):
    histories = {}
    metrics = {}
    print(f"\n Ablation Network Shape : {network_shape}")
    for name, opt in optimizers.items():
        print(f"\n Ablation Optimizers : {name}")
        history = run_training(X_train, y_train, X_valid, y_valid, network_shape, 0.2, opt)
        histories[name] = history

        name, result = extract_metrics_from_history(name, history)
        metrics[name] = result
    # print result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    plot_all_histories(histories)

#  Ablation: With vs Without Dropout
def Ablation_Dropout(X_train, y_train, X_valid, y_valid,network_shape=NETWORK_SHAPE[2], opt = 'SGD',dropout_rate = 0.2):
    optimizer = optimizers[opt]
    histories = {}
    metrics = {}
    print(f"\n Ablation Dropout Rate : {dropout_rate} or 0")

    # train model with Dropout
    histories['Dropout rate'] = run_training(X_train, y_train, X_valid, y_valid,network_shape,dropout_rate,optimizer)
    name, result = extract_metrics_from_history('Dropout rate', histories['Dropout rate'])
    metrics[name] = result
    # train model without Dropout
    histories['None Dropout rate'] = run_training(X_train, y_train, X_valid, y_valid,network_shape,0,optimizer)
    name, result = extract_metrics_from_history('None Dropout rate', histories['None Dropout rate'])
    metrics[name] = result
    # print result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    # draw result
    plot_all_histories(histories)

# Ablation: With vs Without BatchNorm
def Ablation_BN(network_shape,dropout,opt,act):
    optimizer = optimizers[opt]
    histories = {}
    metrics = {}
    print(f"\n Ablation Batch normalization : Used or No")
    # train model with BN
    histories['BN']= run_training(network_shape=network_shape,dropout=dropout,opt=optimizer,act=act,bn=True)
    name, result = extract_metrics_from_history('BN', histories['BN'])
    metrics[name] = result
    # train model without BN
    histories['None BN'] = run_training(network_shape=network_shape, dropout=dropout, opt=optimizer, act=act, bn=False)
    name, result = extract_metrics_from_history('None BN', histories['None BN'])
    metrics[name] = result
    # print result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    plot_all_histories(histories)
# Ablation: Activation Function
def Ablation_ACT(network_shape,dropout,opt,bn = True):
    optimizer = optimizers[opt]
    histories = {}
    metrics = {}
    # train model with ReLu
    print(f"\n Ablation Activation : ReLU")
    histories['ReLU'] = run_training(network_shape=network_shape, dropout=dropout, opt=optimizer, act='ReLU', bn=bn)
    name, result = extract_metrics_from_history('ReLU', histories['ReLU'])
    metrics[name] = result
    # train model with Tanh
    print(f"\n Ablation Activation : Tanh")
    histories['Tanh'] = run_training(network_shape=network_shape, dropout=dropout, opt=optimizer, act='Tanh', bn=False)
    name, result = extract_metrics_from_history('Tanh', histories['Tanh'])
    metrics[name] = result
    # print result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    plot_all_histories(histories)

# Ablation: Learning Rate & Batch Size
def Ablation_lr_and_bs(network_shape,dropout,opt,bn = True):
    optimizer = optimizers[opt]
    histories = {}
    metrics = {}
    learning_rates = [0.01, 0.005, 0.001]
    batch_sizes = [64, 128,256]
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n Ablation Learning Rate: {lr} , Batch Size: {bs}")
            optimizer.learning_rate = lr
            key = f"lr={lr}_bs={bs}"
            histories[key] = run_training(network_shape=network_shape, dropout=dropout, opt=optimizer, act='ReLU', bn=bn,bs=bs)
            name, result = extract_metrics_from_history(key, histories[key])
            metrics[name] = result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    plot_heatmap_from_metrics(metrics,'lr', 'bs','F1','Micro F1 Scores',sw_float=False)

# Ablation: Weight Decay
def Ablation_WD(network_shape,dropout,opt='sgd',bn = True):
    optimizer = optimizers[opt]
    histories = {}
    metrics = {}
    weight_decay= [0, 0.01, 0.005, 0.001]
    for wd in weight_decay:
        print(f"\n Ablation Weight Decay: {wd}")
        optimizer.weight_decay = wd
        key = f"Weight Decay={wd}"
        histories[key] = run_training(network_shape=network_shape, dropout=dropout, opt=optimizer, act='ReLU', bn=bn)
        name, result = extract_metrics_from_history(key, histories[key])
        metrics[name] = result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    plot_all_histories(histories)

# Ablation study about different Dropout
def Ablation_different_Droput(network_shape,opt='sgd',bn = True):
    optimizer = optimizers[opt]
    dropouts = [0.1, 0.15, 0.2, 0.3, 0.5]
    histories = {}
    metrics = {}

    # print result for viewing
    for dropout in dropouts:
        key = f"dropout={dropout}"
        histories[key] = run_training(network_shape=network_shape, dropout=dropout, opt=optimizer, act='ReLU', bn=bn)
        name, result = extract_metrics_from_history(key, histories[key])
        metrics[name] = result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    plot_all_histories(histories)

# Ablation study about the impact of different of learning rate and  weight decay
def Ablation_adam_decay(network_shape, dropout=0.2, act='ReLU', bn=True):
    learning_rates = [0.001, 0.005, 0.01]
    weight_decays = [0.0, 0.0001, 0.0005, 0.005]
    histories = {}
    metrics = {}

    # Doing girl search to find the best optimal model
    for lr in learning_rates:
        for decay in weight_decays:
            key = f"lr={lr}_wd={decay}"
            print(f"\n Running: {key}")

            opt = Optimizer(mode='adam', learning_rate=lr, weight_decay=decay)

            histories[key] = run_training(
                X_train, y_train, X_valid, y_valid,
                network_shape=network_shape,
                dropout=dropout,
                opt=opt,
                act=act,
                bn=bn
            )
            name, result = extract_metrics_from_history(key, histories[key])
            metrics[name] = result

    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    plot_heatmap_from_metrics(metrics,'lr','wd' ,'F1', 'Micro F1 Scores','Learning Rate','Weight Decay')

'''
    Ablation experiments can only be run one at a time.
'''
if __name__ == "__main__":
    # Setting seed
    np.random.seed(30)
    # Ablation_Network_Shape(X_train, y_train, X_valid, y_valid, NETWORK_SHAPE, opt='SGD',dropout_rate = 0.2)
    # Ablation_optimizers(X_train, y_train, X_valid, y_valid, NETWORK_SHAPE[2])
    #Ablation_adam_decay(NETWORK_SHAPE[2])
    #Ablation_BN(NETWORK_SHAPE[1],0.2,opt='Adam+Decay',act='ReLU')
    #Ablation_ACT(NETWORK_SHAPE[2],0.2,opt='SGD',bn=True)
    #Ablation_Dropout(X_train, y_train, X_valid, y_valid, NETWORK_SHAPE[1],opt = 'Adam+Decay', dropout_rate = 0.2)
    #Ablation_lr_and_bs(NETWORK_SHAPE[1],0.2,opt='Adam+Decay',bn=True)
    Ablation_different_Droput(NETWORK_SHAPE[2],opt='Adam+Decay')
