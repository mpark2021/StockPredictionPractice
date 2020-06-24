import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


data = pd.read_csv('../data/data_stocks.csv')
data = data.drop(['DATE'], 1)

# m = number of data / f = feature dimension (Y+X)

(m, f) = data.shape
data = data.values

# training / test index

train_start = 0
train_end = int(np.floor(m*0.8))
test_start = train_end
test_end = m
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# scaler

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# training set & test set

X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

n_stocks = f - 1

layer = [2048, 1024, 512, 256, 128, 64]

# session

net = tf.InteractiveSession()

X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
y = tf.placeholder(dtype=tf.float32, shape=[None])

# theta initialization

sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=sigma)
bias_initializer = tf.zeros_initializer()

W_hidden = []
bias_hidden = []

for i in range(len(layer)):
    if i==0:
        _in = n_stocks
    else:
        _in = layer[i-1]
    _out = layer[i]

    W_hidden.append(tf.Variable(weight_initializer([_in, _out])))
    bias_hidden.append(tf.Variable(bias_initializer([_out])))

W_out = tf.Variable(weight_initializer([layer[-1], 1]))
bias_out = tf.Variable(bias_initializer([1]))


# layers

hidden = []


for i in range(len(layer)):
    if i==0:
        _in = X
        _in_W = W_hidden[0]
    else:
        _in = hidden[i - 1]
        _in_W = W_hidden[i]

    _out = bias_hidden[i]

    hidden.append(tf.nn.relu(tf.add(tf.matmul(_in, _in_W), _out)))

out = tf.transpose(tf.add(tf.matmul(hidden[-1], W_out), bias_out))


# cost function

mse = tf.reduce_mean(tf.squared_difference(out, y))

# optimizer

adam = tf.train.AdamOptimizer().minimize(mse)

net.run(fetches=tf.global_variables_initializer())


plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# fit

batch_size = 256
mse_train = []
mse_test = []
num_epoch = 16

for epoch in range(num_epoch):
    shuffle = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle]
    y_train = y_train[shuffle]

    for i in range(0, m // batch_size):
        start = i * batch_size
        batch_X = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]

        net.run(adam, feed_dict={X: batch_X, y: batch_y})

        if np.mod(i, 50) == 0:
            mse_train.append(net.run(mse, feed_dict={X: X_train, y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, y: y_test}))
            print(f'Train Error: {mse_train[-1]} / Test Error: {mse_test[-1]}')

            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title(f'Epoch: {epoch} / Batch: {i}')
            plt.pause(0.001)


if __name__ == "__main__":
    import os
    os.chdir('..')