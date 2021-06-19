import numpy as np
import copy

class Neuron:
    def __init__(self, idx, units, bias, weight_constraint, initializer):
        self.idx = idx
        self.units = units
        self.w = np.array([initializer(weight_constraint[0], weight_constraint[1], 1) for _ in range(units)], dtype='float32')
        self.velocity = np.zeros(units)
        if bias:
            self.b = initializer(weight_constraint[0], weight_constraint[1], 1)
        else:
            self.b = False

    @property
    def weight(self):
        return self.w

    @weight.setter
    def weight(self, nw):
        self.w = nw

    @property
    def bias(self):
        return self.b

    @bias.setter
    def bias(self, nb):
        self.b = nb

    @property
    def weight_matrix(self):
        return self.w, self.b

    @weight_matrix.setter
    def weight_matrix(self, args):
        assert len(self.w) == len(args['weight']), '{} vs {}'.format(len(self.w), len(args['weight']))
        self.w = copy.deepcopy(args['weight'])
        self.b = copy.deepcopy(args['bias'])

    @property
    def flat_weight(self):
        lst = list()
        for w in self.w:
            lst.append(w)
        if self.b:
            lst.append(self.b)
        return lst

    def __str__(self):
        return 'Neuron {}: weights:\n{}\n bias: {}\n'.format(self.idx, self.w, self.b)
