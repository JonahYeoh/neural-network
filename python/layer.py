import numpy as np
import copy
from neuron import Neuron

# n_weights = (n+1) * nin
# sgd: [-0.5, 0.5]
class DenseLayer:
    def __init__(self, idx, units, bias, weight_constraint=[-3, 3], initializer=np.random.uniform, batch_size=5, afn=None, dafn=None, rfn=None):
        self.idx = idx
        self.activation = afn
        self.d_activation = dafn
        self.regularization = rfn
        self.outputs = None
        self.delta = None
        self.bias = bias
        self.weight_constraint = weight_constraint
        self.initializer = initializer
        self.units = units

    def __build__(self, nin):
        print('building layer', self.idx, 'with', nin)
        self.nin = nin
        self.n_params = self.units * nin + self.units
        print(self.n_params)
        self.neurons = [Neuron(i, nin, self.bias, self.weight_constraint, self.initializer) for i in range(self.units)]

    @property
    def seqlen(self):
        return self.n_params

    def __load__(self, weights):

        assert len(weights) == self.n_params, '{} vs {}'.format(len(weights), self.n_params)
        for idx, neuron in enumerate(self.neurons):
            seg = weights[idx*(self.nin+1):(idx+1)*(self.nin+1)]
            neuron.weight_matrix = dict(weight=seg[:-1], bias=seg[-1])

    def __call__(self, x): # [N, C]
        by = list()
        y = list()
        for X in x:
            for neuron in self.neurons:
                w, b = neuron.weight_matrix
                a = np.sum(w * X) + b
                #print('a', a.shape)
                y.append(a)
            #print('y', y)
            by.append(np.array(y, dtype='float32'))
            y = list()
        if self.activation is not None:
            by = self.activation(by)
        #print('by', by)
        tmp = np.array(by, dtype='float32')
        #print('layer:', self.idx, tmp.shape)
        self.inputs = copy.deepcopy(x)
        self.outputs = by
        #print(len(by), len(by[0]))
        return by

    def __update__(self, learning_rate):
        momentum = 0.005
        #velocity = momentum * velocity - learning_rate * g
        for n in range(len(self.neurons)):
            for i in range(self.nin):
                batch_delta = 0.0
                tmp = list()
                for b in range(len(self.outputs)):
                    adjustment = self.delta[b][n] * self.inputs[b][i]
                    #print('adj', adjustment)
                    tmp.append(adjustment)
                    batch_delta += self.delta[b][n]
                self.neurons[n].velocity = momentum * self.neurons[n].velocity - learning_rate * np.sum(tmp)
                self.neurons[n].weight[i] += self.neurons[n].velocity # / len(self.outputs)
            self.neurons[n].bias -= learning_rate * batch_delta #/ len(self.outputs)


    def loss(self, y=[], yhat=[]):
        if len(yhat) == 0:
            if self.regularization:
                wmatrix = list()
                for neuron in self.neurons:
                    wmatrix.extend(neuron.flat_weight)
                return self.regularization(wmatrix)
            return 0
        # to be implemented
        return 0
    
    def __str__(self):
        for neuron in self.neurons:
            print(neuron)
        return '$'
