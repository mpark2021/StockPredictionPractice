import sys
import os
from src.preprocess import *
from src.model_hw import *

warnings.simplefilter(action='ignore', category=FutureWarning)


# target
if len(sys.argv) == 1:
    path = './data/fa_data_2012_2019.npy'
else:
    if 'p' in sys.argv[1].lower():
        path = './data/fa_data_2012_2019_pit.npy'
    if 'b' in sys.argv[1].lower():
        path = './data/fa_data_2012_2019_bat.npy'


# data configuration
train, test, raw = load(path)

(X_train, y_train), (X_test, y_test), scaler = process(train, test)


#model
network, error, output = model([1024, 512, 256, 128], X_train.shape[1])
pred = run(network, error, output, X_train, y_train, X_test, y_test)
pred = scaler.inverse_transform(np.concatenate((X_test, pred), axis=1))[:, -1]