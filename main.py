from Data import *
from Network import *
from Utility import *
from Optimizer import Optimizer
import numpy as np
import time

data = DataProcessor()
X_train, y_train = data.X_train, data.y_train
X_valid, y_valid = data.X_valid, data.y_valid
X_test, y_test = data.X_test, data.y_test

INPUT_DIM = X_train.shape[1]
DROPOUT_RATE = 0.25
BATCH_SIZE = 64
MAX_EPOCH = 50
PATIENCE = 5
NETWORK_SHAPE = build_shape(INPUT_DIM, [512, 512, 256, 128, 64])
opt = {
        'Adam+Decay': Optimizer(mode='adam', learning_rate=0.001, weight_decay=0.0005),
}
ACTIVATION = 'ReLU'

def final_training():
    metrics = {}
    start_time = time.time() # recording time
    network = Network(NETWORK_SHAPE, dropout_rate=DROPOUT_RATE, optimizer=opt['Adam+Decay'], activation=ACTIVATION,
                      use_batchnorm=True)
    history = network.train(
        X_train, y_train, BATCH_SIZE,
        epochs=MAX_EPOCH,
        valid_data=X_valid, valid_target=y_valid,
        patience=PATIENCE
    )
    name, result = extract_metrics_from_history('Final training results', history)
    metrics[name] = result
    # print result
    print("\n Summary:")
    for name, stats in metrics.items():
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    test_preds = network.network_forward(X_test)
    end_time = time.time()  # recording the ending time
    duration = end_time - start_time
    test_f1, test_acc,test_loss, =calculate_result(test_preds, y_test)
    print('  Test f1', round(test_f1, 4))
    print('  Test loss', round(test_loss, 4))
    print('  Test acc', round(test_acc, 4))
    print('  Training time:',round(duration, 4))



if __name__ == '__main__':
    # Setting seed
    np.random.seed(5329)
    final_training()