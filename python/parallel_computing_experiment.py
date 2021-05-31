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
import os
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
def get_data(root, dir, columns=None):
    with open(os.path.join(root, dir), 'r') as freader:
        lines = freader.readlines()
    records = list(map(lambda ele: list(map(lambda e: float(e), ele.split('\t'))), lines))
    df = pd.DataFrame(records)
    if columns is not None:
        df.columns = columns
    return df


if __name__ == "__main__":
    print('ROOT: ', os.getcwd())
    DATA = os.path.join(os.getcwd(), 'dataset')
    GROUP = 1
    N_COLUMN = 5
    N_CLASS = 3
    COLUMNS = ['f1', 'f2', 'f3', 'f4', 'f5', 'label']
    EPOCHS = 100
    BATCH_SIZE = 100
    GOAL = 0.001
    train_dir, test_dir = "..//dataset//training_data{}.txt".format(GROUP), "..//dataset//testing_data{}.txt".format(GROUP)
    train, test = get_data(DATA, train_dir, COLUMNS), get_data(DATA, test_dir, COLUMNS)
    # shuffle
    train, test = train.sample(frac=1), test.sample(frac=1)
    print('TRAIN DIM:', train.shape)
    print('TEST DIM:', test.shape)
    #
    x_train, y_train = train.drop('label', axis=1), train['label']
    x_test, y_test = test.drop('label', axis=1), test['label']
    scaler = MinMaxScaler(['f1', 'f2', 'f3', 'f4', 'f5'])
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = one_hot_encoding(y_train, N_CLASS)
    y_test = one_hot_encoding(y_test, N_CLASS)

    bees = BEES(
        dict(
            name="bees_agent",
            level="weak ai",
            m = 100,
            weight_constraint=[-3.0, 3.0],
            c1 = 2.0,
            c2 = 0.05,
            max_c1=2.0,
            min_c1=0.5,
            max_c2=2.0,
            min_c2=0.5,
            velocity_constraint=[-0.2, 0.2],
            w=1.2,
            decay = 0.95,
            goal=0.0001,
            initializer="glorot_uniform",
            n_workers=3
        )
    )
    model = ML_Agent(n_feature=N_COLUMN, target_size=N_CLASS)

    # add_layer(self, units, activation='relu', regularizer='l2', use_bias=True):
    model.add_layer(8, 'sigmoid', True, 'uniform')
    model.add_layer(N_CLASS, "sigmoid", True, 'uniform')

    model.compile_configuration(
        bees,
        loss="categorical_crossentropy",
        monitor=["accuracy", "precision", "recall", "essential_metrics"],
        regularizer='l1',
    )  #

    # x, y, max_iter=100, goal=0.1, batch_size=5, validation=False, x_val=[], y_val=[], validation_batch_size=5
    minimum_loss = model.fit(
        x_train,
        y_train,
        max_iter = EPOCHS,
        goal = GOAL,
        batch_size = len(x_train) #BATCH_SIZE
    )
    print("minimum loss:", minimum_loss)
    
    loss, board = model._evaluate(x_test, y_test, len(y_test))
    #print("loss", loss)
    #print(board)
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
