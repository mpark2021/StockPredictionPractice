import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load(path, cv_year = 18.0, test_year=19.0): # test만 19.0 그대로 & cv랑 train은 random (10%만 cv로, 나머지는 train)
    data_wn = np.load(path, allow_pickle=True)
    data_wn = data_wn[data_wn[:, 0].argsort()]

    idx = [x for x in range(data_wn.shape[1])]
    idx.remove(1)

    data = data_wn[:, idx]
    data = data.astype(float)

    # data = pd.read_csv('../data/data_stocks.csv')
    # data = data.drop(['DATE'], 1)

    # m = number of data / f = feature dimension (Y+X)

    (m, f) = data.shape
    # data = data.values

    # training / test index


    cv_cut = list(data[:, 0]).index(cv_year)

    test_cut = list(data[:, 0]).index(test_year)

    train_start = 0
    train_end = cv_cut
    cv_start = cv_cut
    cv_end = test_cut
    test_start = test_cut
    test_end = m
    data_train = data[np.arange(train_start, train_end), :]
    data_cv = data[np.arange(cv_start, cv_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    data_raw_test = data_wn[np.arange(test_start, test_end), :]

    return data_train, data_cv, data_test, data_raw_test


def process(data_train, data_cv, data_test):
    # scaler

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_cv = scaler.transform(data_cv)
    data_test = scaler.transform(data_test)

    # training set & test set

    X_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    X_cv = data_cv[:, :-1]
    y_cv = data_cv[:, -1]
    X_test = data_test[:, :-1]
    y_test = data_test[:, -1]

    return (X_train, y_train), (X_cv, y_cv), (X_test, y_test), scaler

