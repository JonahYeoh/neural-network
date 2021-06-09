'''
PSO
'''
# dependency
import numpy as np
import copy
import sys
from sklearn.utils import shuffle
from utility import trim_tail

class Particle(object):
    def __init__(self, idx, seq_len, aim, weight_constraint = None):
        self._idx = idx
        if weight_constraint is not None:
            self._weight_matrix = np.random.uniform(weight_constraint[0], weight_constraint[1], seq_len)
            self._velocity_matrix = np.zeros(seq_len)
            #self._weight_matrix = trim_tail(self._weight_matrix)
            assert len(self._weight_matrix) == seq_len, 'id {}, {}'.format(idx, len(self._weight_matrix))
        else:
            self._weight_matrix = None
        if not aim:
            self._fitness = sys.float_info.max
        else:
            self._fitness = sys.float_info.min
        self._best = None # {'fitness':0, 'wmatrix':[]}
        self._velocity_matrix = np.array([np.random.uniform(0,1,1) for i in range(seq_len)], dtype='float32')
        
    @property
    def idx(self):
        return self._idx
    
    @property
    def wmatrix(self):
        return self._weight_matrix
    
    @wmatrix.setter
    def wmatrix(self, matrix):
        self._weight_matrix = copy.deepcopy(matrix)
        
    @property
    def vmatrix(self):
        return self._velocity_matrix
    
    @vmatrix.setter
    def vmatrix(self, matrix):
        self._velocity_matrix = copy.deepcopy(matrix)
        
    @property
    def best(self):
        return self._best
    
    @best.setter
    def best(self, record):
        self._best = record
        
    @property
    def fitness(self):
        return self._fitness
    
    @fitness.setter
    def fitness(self, score):
        self._fitness = score

class PSO(object):
    def __init__(self, config):
        self.M = config['m']
        self.weight_constraint = config['weight_constraint']
        self.velocity_constraint = config['velocity_constraint']
        self.C1, self.C2 = config['c1'][1], config['c2'][0] # 
        self.MAX_C1, self.MIN_C1 = config['c1'][1], config['c1'][0]
        self.MAX_C2, self.MIN_C2 = config['c2'][1], config['c2'][0]
        self.W = config['w'] # learning rate
        self.W_Decay = config['w_decay']
        self.global_best_matrix = None
        
    def __build__(self, seq_len, aim):
        self.seq_len = seq_len
        self.aim = aim
        self.swams = [Particle(m, seq_len, aim, self.weight_constraint) for m in range(self.M)]
        self.global_best_fitness = sys.float_info.min if aim else sys.float_info.max
    
    def fit(self, obj, x, y, max_iter, goal = 0, batch_size = 0, loss = 'categorical_crossentropy'):
        min_improv = 0.0001
        patience = 10
        best_val = sys.float_info.max
        arrgg = 0
        history = list()
        for itr in range(max_iter):
            for m in range(len(self.swams)):
                wmatrix = self.swams[m].wmatrix
                obj.__load__(wmatrix)
                metrics = obj.evaluate(x, y, len(x))
                self.swams[m].fitness = metrics[loss]
            best_fitness, best_wmatrix, _ = self.update_state()
            history.append(best_fitness)
            self.update_pool()
            self.update_learning_config(max_iter, itr)
            print('Iteration {}/{}: \t{}'.format(itr, max_iter, best_fitness))
        return best_fitness, best_wmatrix, history     

    def update_state(self):
        best_particle = None
        if not self.aim:
            for particle in self.swams:
                if particle.best is None:
                    particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
                elif particle.fitness <= particle.best['fitness']:
                    particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
                    print('update local,', particle.fitness)
                if particle.best['fitness'] <= self.global_best_fitness:
                    print('update global', particle.best['fitness'])
                    self.global_best_fitness = particle.best['fitness']
                    best_particle = particle.idx
        else:
            for particle in self.swams:
                if particle.best is None:
                    particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
                elif particle.fitness >= particle.best['fitness']:
                    particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
                    print('update local,', particle.fitness)
                if particle.best['fitness'] >= self.global_best_fitness:
                    print('update global', particle.best['fitness'])
                    self.global_best_fitness = particle.best['fitness']
                    best_particle = particle.idx

        swam_fitness = [p.fitness for p in self.swams]
        if best_particle is not None:
            self.global_best_matrix = copy.deepcopy(self.swams[best_particle].best['wmatrix'])
        return self.global_best_fitness, self.global_best_matrix, np.sum(swam_fitness) / len(swam_fitness)

    def update_pool(self):
        #print('Update Pool')
        for particle in self.swams:
            #print('\n\nParticle:', particle.idx)
            new_matrix = []
            new_velocity = []
            for v,p,l,g in zip(particle.vmatrix, particle.wmatrix, particle.best['wmatrix'], self.global_best_matrix):
                #print(v, p, l, g)
                _velocity = (self.W * v) + \
                (self.C1 * np.random.uniform(0,1,1) * (l-p)) + \
                (self.C2 * np.random.uniform(0,1,1) * (g-p))
                velocity = self.clip(_velocity+v, self.velocity_constraint)
                #print('velo', velocity)
                new_velocity.append(velocity)
                weight = self.clip(p+velocity + np.random.uniform(-0.1, 0.1, 1), self.weight_constraint)
                new_matrix.append(weight)
            particle.wmatrix = np.array(new_matrix, dtype='float32')
            particle.vmatrix = np.array(new_velocity, dtype='float32')
            #print(particle.idx, particle.wmatrix.shape, particle.vmatrix.shape)

    def update_learning_config(self, max_iter, itr):
        #print('Update Learning Configuration')
        #self.C1 = self.MAX_C1 + (self.MAX_C1 - self.MIN_C1) / max_iter * itr
        self.C1 = self.MAX_C1 - (self.MAX_C1 * itr / max_iter) + self.MIN_C1
        self.C2 = np.min([self.MIN_C2 + (self.MAX_C2 * itr / max_iter), self.MAX_C2])
        self.W = self.W * self.W_Decay
        pass

    def clip(self, x, bound):
        #print('clip', type(x), type(bound[0]))
        x = x if x > bound[0] else np.array([bound[0]])
        return x if x < bound[1] else np.array([bound[1]])
    
    def get_agent_wmatrix(self, idx):
        return self.swams[idx].wmatrix
    
    @property
    def m(self):
        return self.M