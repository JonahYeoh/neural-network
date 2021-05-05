from model import ML_Agent
from pso import PSO
from ga import GA
from parallel_bees import BEES
import pandas as pd
import numpy as np
from utility import split_data, one_hot_encoding, split_data_2  # (dataframe, label, do_shuffle=True, test_size=0.1):
from metrics import MSE
from scalers import StandardScaler, MinMaxScaler
from sklearn import datasets

"""
for name, met in board.items():
    if type(met) == dict:
        print(name)
        for key, value in met.items():
            print(key, value)
    else:
        print(name, met)

for pred, true in zip(y_pred, y_test):
    print(pred, 'VS', true)

for idx, particle in enumerate(bees.swams):
    print(idx, particle.fitness)
"""

if __name__ == "__main__":
    df = pd.read_csv("thyroid.csv")
    # thyroid.csv
    df = df.sample(frac=1)
    # df.head(5)
    cls_col = 'CLASS'
    '''
    iris = datasets.load_iris()
    feature = iris['data']
    label = iris[cls_col]
    df = pd.DataFrame(feature, columns=iris['feature_names'])
    df[cls_col] = label
    print(type(df))
    '''
    #train, validation, test = split_data(df, "CLASS")
    train, test = split_data_2(df, cls_col)
    # print(train[:5], train.shape)
    # print(validation[:5], validation.shape)
    # print(test[:5], test.shape)

    x_train, y_train = train.drop(cls_col, axis=1), train[cls_col]
    #x_val, y_val = validation.drop("CLASS", axis=1), validation["CLASS"]
    x_test, y_test = test.drop(cls_col, axis=1), test[cls_col]

    n_class = len(np.unique(y_train))
    '''
    x_train, y_train = StandardScaler(x_train), one_hot_encoding(y_train, n_class)
    #x_val, y_val = StandardScaler(x_val), one_hot_encoding(y_val, n_class)
    x_test, y_test = StandardScaler(x_test), one_hot_encoding(y_test, n_class)
    '''
    x_train, y_train = MinMaxScaler(x_train), one_hot_encoding(y_train, n_class)
    #x_val, y_val = MinMaxScaler(x_val), one_hot_encoding(y_val, n_class)
    x_test, y_test = MinMaxScaler(x_test), one_hot_encoding(y_test, n_class)

    bees = BEES(
        dict(
            name="bees_agent",
            level="weak ai",
            m=200,
            weight_constraint=[-3.0, 3.0],
            c1=2.0,
            c2=2.0,
            c3=1,
            max_c1=2.0,
            min_c1=0.8,
            max_c2=2.0,
            min_c2=0.9,
            max_c3=1.0,
            min_c3=0.05,
            velocity_constraint=[-0.5, 0.5],
            w=1.6,
            w_decay=0.9,
            goal=0.0001,
            initializer="glorot_uniform",
        )
    )
    ga = GA(
        dict(
            name="ga_agent2",
            level="weak weak ai",
            m=30,
            x_rate=[0.1, 0.95],
            m_rate=[0.05, 0.2],
            weight_constraint=[-1.0, 1.0],
            radioactive_rating=[0.1, 1.0],
            grow_factor=[0.1, 1.0],
        )
    )
    model = ML_Agent(n_feature=5, target_size=3)

    # add_layer(self, units, activation='relu', regularizer='l2', use_bias=True):
    model.add_layer(8, "sigmoid", True, 'glorot_uniform')
    model.add_layer(3, "sigmoid", True, 'glorot_uniform')

    model.compile_configuration(
        bees,
        loss="categorical_crossentropy",
        monitor=["accuracy", "precision", "recall", "essential_metrics"],
        regularizer=None,
    )  #

    # x, y, max_iter=100, goal=0.1, batch_size=5, validation=False, x_val=[], y_val=[], validation_batch_size=5
    minimum_loss = model.fit(
        x_train,
        y_train,
        max_iter=500,
        goal=0.001,
        batch_size=len(y_train),
        validation=False,
        x_val=[],
        y_val=[],
        validation_batch_size=None
    )
    print("minimum loss:", minimum_loss)

    loss, board = model._evaluate(x_test, y_test, len(y_test))
    print("loss", loss)
    print(board)
    y_pred = model._predict(x_test)
    y_test = np.array(y_test)
    loss = model.__loss__(y_pred, y_test)
    print(loss)

    mse_loss = MSE(y_pred, y_test)
    print(mse_loss)
    print('trace')
    for pred, true in zip(y_pred, y_test):
        print(pred, 'VS', true)

    for name, met in board.items():
        if type(met) == dict:
            print(name)
            for key, value in met.items():
                print(key, value)
        else:
            print(name, met)

    for wt in model.flat_weight:
        print(wt)