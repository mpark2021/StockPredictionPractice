import sys
import os
import math
from src.preprocess import *
from src.model_hw import *

warnings.simplefilter(action='ignore', category=FutureWarning)

# Path => Size => Path => Name => Lambda


# target
path = sys.argv[2]


# model size
max_size = int(sys.argv[1])
layers = [2 ** x for x in range(int(math.log2(max_size)), 6, -1)]
epoch = int(sys.argv[3])


name = sys.argv[4]
l = float(sys.argv[5])
use_reg = sys.argv[5] !='0'


# data configuration
train, cv, test, raw = load(path)

(X_train, y_train), (X_cv, y_cv), (X_test, y_test), scaler = process(train, cv, test)


#model
create_checkpoint_dir(str(layers[0]))
network, error, output, save = model(layers, X_train.shape[1], name, l, use_reg)
pred = run(network, error, output, save, X_train, y_train, X_cv, y_cv, X_test, y_test, num_epoch=epoch)
pred = scaler.inverse_transform(np.concatenate((X_test, pred), axis=1))[:, -1]