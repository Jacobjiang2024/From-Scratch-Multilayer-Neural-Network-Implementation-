# import library
from sklearn.model_selection import train_test_split
import pandas as pd
from Plot import *


class DataProcessor:
    # This class simply reads data in .npy format and processes it.
    def __init__(self, split=0.2):
        # read train and test data
        self.train_data = np.load('train_data.npy')
        # Converting data to one-dimensional
        self.train_label = np.load('train_label.npy').flatten()
        self.test_data = np.load('test_data.npy')
        # Converting data to one-dimensional
        self.test_label = np.load('test_label.npy').flatten()
        # check mean,standard, minmum and maxmum value of train data
        self.stats_subset = self.calculated_data_features(self.train_data)
        self.stats_y_subset = self.calculated_data_labels(self.train_label)
        # Gain distribution of X_train with standardization
        self.sta_x = self.Standardization(self.train_data)
        # Split the training data into validation and training sets.
        X_train, X_valid, y_train, y_valid = train_test_split(self.train_data, self.train_label, test_size=split,
                                                              stratify=self.train_label, random_state=0)
        #
        self.X_train = self.Standardization(X_train)
        self.y_train = y_train
        self.X_valid = self.Standardization(X_valid)
        self.y_valid = y_valid
        self.X_test = self.Standardization(self.test_data)
        self.y_test = self.test_label

    def calculated_data_features(self, X):
        # Calculate mean, standard, minimum and maximum values of data
        # Put data into DataFrame, column names can be used with Feature_1
        feature_names = [f"Feature_{i + 1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        # get result of mean, standard, minimum and maximum values of data
        return df.agg(['mean', 'std', 'min', 'max'])

    def calculated_data_labels(self, y):
        # Calculate mean, standard, minimum and maximum values of data
        # Put data into DataFrame, column names can be used with Feature_1
        y_2d = y.reshape(-1, 1)
        df = pd.DataFrame(y_2d, columns=['Label'])
        # get result of mean, standard, minimum and maximum values of data
        return df.agg(['mean', 'std', 'min', 'max'])

    def Standardization(self, data):
        # Standardization
        # Change data value into mean equal 0 and standard equal 1
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # std_scaler = StandardScaler()
        # Mean and standard deviation
        mean = data.mean(axis=0)
        std = data.std(axis=0)

        std[std == 0] = 1
        return (data - mean) / std


def normalize(array):
    # change data value from -1 to 1
    max_number = np.max(np.absolute(array), axis=1, keepdims=True)
    scale_rete = np.where(max_number == 0, 0, 1 / max_number)
    norm = array * scale_rete
    return norm

    # Vector normalization function
def vector_normalize(array):
    max_value = np.max(np.absolute(array))
    scale = np.ones_like(max_value, dtype=np.float64)
    nonzero = max_value != 0
    scale[nonzero] = 1.0 / max_value[nonzero]
    return array * scale

if __name__ == '__main__':
    data = DataProcessor()

    # draw distribution of features
    plot_feature_distributions(data.train_data,data.sta_x,[0,1],'pink')