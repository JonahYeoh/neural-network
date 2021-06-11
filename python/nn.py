import numpy as np
import copy
from metrics import ACC, CCE, MSE, PRECISION, RECALL, essential_metrics, F1

metrics_fn_dict = dict()
metrics_fn_dict['accuracy'] = ACC
metrics_fn_dict['categorical_crossentropy'] = CCE
metrics_fn_dict['mean_square_error'] = MSE
metrics_fn_dict['precision'] = PRECISION
metrics_fn_dict['recall'] = RECALL
metrics_fn_dict['essential_metrics'] = essential_metrics
metrics_fn_dict['f1'] = F1

# n_layer * owns unit count
class Network:
    def __init__(self, nin, lr):
        #print('init')
        self.nin = nin
        self.layers = []
        self.learning_rate = lr

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss_fn, optimizer, metrics=[]):
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.layers[0].__build__(self.nin)
        prev = self.layers[0].units
        for layer in self.layers[1:]:
            layer.__build__(prev)
            prev = layer.units
        self.optimizer = optimizer
        aim = True if loss_fn == 'accuracy' else False
        self.optimizer.__build__(self.get_nparams(), aim)

    def fit(self, x, y, epochs=100, verbose = 1):
        history = list()
        X, Y = x.values, y.values
        if self.optimizer == 'sgd':
            for itr in range(epochs):
                #print('itr', itr)
                yhat = self.predict(X)
                #print('backprop')
                self.__backprop__(Y)
                self.__update__()
                #print('update')
                score = self.evaluate(X, Y, training=True, verbose = 0)
                #print('Epoch {:03d}: {}'.format(itr, score))
                #print('done', itr)
                if itr % 50 == 0:
                    self.learning_rate *= 0.95
        else:
            fitness, wmatrix, history = self.optimizer.fit(self, X, Y, epochs, \
                loss = self.loss_fn, verbose = verbose)
            self.__load__(wmatrix)
            self.best_weight = copy.deepcopy(wmatrix)
        #print('finish training')
        return history

    def predict(self, x):
        return self.__feedforward__(x)

    def evaluate(self, x, y, training=True, verbose = 1):
        if not training:
            x, y = x.values, y.values
        yhat = self.__feedforward__(x)
        if verbose == 1:
            for t, p in zip(y, yhat):
                print(t, '<->', p)
        return self.monitor(y, yhat)
    
    def loss(self, y, yhat):
        return 0
    
    def monitor(self, y, yhat):
        essential_metrics = metrics_fn_dict['essential_metrics'](yhat, y)
        metric_board = dict()
        for key in self.metrics:
            metric_board[key] = metrics_fn_dict[key](yhat, y, essential_metrics)
        metric_board['essential_metrics'] = essential_metrics
        return metric_board

    def __str__(self):
        for layer in self.layers:
            print(layer)
        return ''

    def __load__(self, weights, tmpmsg=''):
        collected = 0
        if type(weights) == tuple:
            v, weights = weights
        for lyr in self.layers:
            seg = weights[collected:collected + lyr.n_params]
            #print(idx, '->', len(seg))
            lyr.__load__(seg)
            collected += lyr.n_params

    def __feedforward__(self, inputs):
        #print('feedforward')
        x = copy.deepcopy(inputs)
        for layer in self.layers:
            #print('exec layer', layer.idx)
            x = layer(x)
        #print('end feedforward')
        return x

    def __backprop__(self, labels): # [ B, C ]
        #print('== start backprop ==')
        last_layer = self.layers[-1]
        last_layer.delta = list() # [ B, C]
        for o, l in zip(last_layer.outputs, labels):
            d = []
            for c, (oi, li) in enumerate(zip(o, l)):
                e = -(li - oi) * last_layer.d_activation(oi)
                d.append(e)
            last_layer.delta.append(d)
        # above OK
        l = len(self.layers) - 2
        while (l >= 0):
            clayer = self.layers[l]
            player = self.layers[l+1]
            bdelta = list()
            for b in range(len(clayer.outputs)):
                delta = list()
                for c, _ in enumerate(clayer.neurons):
                    acc_error = 0.0
                    for p in range(len(player.neurons)):
                        acc_error += player.delta[b][p] * player.neurons[p].weight[c]
                    ret = acc_error * clayer.d_activation(clayer.outputs[b][c])
                    delta.append(ret)
                bdelta.append(delta)
            clayer.delta = bdelta
            #print(len(bdelta), len(bdelta[0]))
            l -= 1
        #print('== end backprop ==')

    def __update__(self):
        for layer in self.layers:
            layer.__update__(self.learning_rate)
        #print('== update ==')

    def get_nparams(self):
        n_params = 0
        for layer in self.layers:
            n_params += layer.seqlen
        return n_params
