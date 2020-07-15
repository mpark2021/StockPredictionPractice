import sys
import os
import math
from src.preprocess import *
from src.model_hw import *

warnings.simplefilter(action='ignore', category=FutureWarning)


# target
if len(sys.argv) < 2:
    path = './data/fa_data_2012_2019.npy'
else:
    if 'p' in sys.argv[2].lower():
        path = './Data/fa_data_2012_2019_[5.0, 3.0, 2.0]_pit.npy'
    else:
        path = './Data/fa_data_2012_2019_[5.0, 3.0, 2.0]_bat.npy'


# model size
max_size = int(sys.argv[1])
layers = [2 ** x for x in range(int(math.log2(max_size)), 6, -1)]


# data configuration
train, cv, test, raw = load(path)

(X_train, y_train), (X_cv, y_cv), (X_test, y_test), scaler = process(train, cv, test)


#model
network, error, output = model(layers, X_train.shape[1])
pred = run(network, error, output, X_train, y_train, X_cv, y_cv, X_test, y_test)
pred = scaler.inverse_transform(np.concatenate((X_test, pred), axis=1))[:, -1]