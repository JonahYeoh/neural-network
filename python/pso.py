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
    def __init__(self, idx, seq_len):
        self._idx = idx
        self._weight_matrix = np.array([np.random.uniform(0,0.05,1) for i in range(seq_len)], dtype='float32')
        self._weight_matrix = trim_tail(self._weight_matrix)
        self._fitness = sys.float_info.max
        self._val_fitness = sys.float_info.max
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

    @property
    def val_fitness(self):
        return self._val_fitness

    @val_fitness.setter
    def val_fitness(self, score):
        self._val_fitness = score

class PSO(object):
    def __init__(self, config):
        print('building object according to ', config)
        self.M = config['m']
        assert len(config['weight_constraint']) == 2
        self.weight_constraint = config['weight_constraint']
        self.C1, self.C2 = config['c1'], config['c2']
        self.MAX_C1, self.MIN_C1 = config['max_c1'], config['min_c1']
        self.MAX_C2, self.MIN_C2 = config['max_c2'], config['min_c2']
        self.velocity_constraint = config['velocity_constraint']
        self.W = config['w']
        self.W_Decay = config['w_decay']
        self.initializer = config['initializer'] # unused
        self.global_best_matrix = None
        self.global_best_fitness = sys.float_info.max
        
    def __build__(self, seq_len):
        self.seq_len = seq_len
        self.swams = [Particle(m, seq_len) for m in range(self.M)]
        #print('finish building', len(self.swams))
    
    def fit(self, obj, x, y, max_iter, goal, pack_goal=0.05, batch_size=5, validation=False, x_val=[], y_val=[], validation_batch_size=5):
        min_improv = 0.0001
        patience = 10
        best_val = sys.float_info.max
        arrgg = 0
        for itr in range(max_iter):
            print('Iteration:', itr)
            seed = np.random.randint(1000)
            x = shuffle(x, random_state=seed)
            y = shuffle(y, random_state=seed)
            if validation:
                x_val = shuffle(x_val, random_state=seed)
                y_val = shuffle(y_val, random_state=seed)
            # training
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                for m in range(self.M):
                    #print('.', end='')
                    wmatrix = self.get_agent_wmatrix(m)
                    obj.__load_weight__(wmatrix)
                    self.swams[m].fitness, m_board = obj._evaluate(batch_x, batch_y, len(batch_x))
                best_fitness, best_wmatrix, swam_fitness = self.update_state()
                if best_fitness <= goal or swam_fitness <= pack_goal:
                    return best_fitness, best_wmatrix
                self.update_pool()
            # validation
            if validation:
                base = best_val
                for i in range(0, len(x_val), validation_batch_size):
                    batch_x = x_val[i:i+validation_batch_size]
                    batch_y = y_val[i:i+validation_batch_size]
                    for m in range(self.M):
                        #print('.', end='')
                        wmatrix = self.get_agent_wmatrix(m)
                        obj.__load_weight__(wmatrix)
                        self.swams[m].val_fitness, val_m_board = obj._evaluate(batch_x, batch_y, len(x_val))
                        if self.swams[m].val_fitness < best_val:
                            best_val = self.swams[m].val_fitness
                if best_val - base >= min_improv:
                    arrgg += 1
                    print('no improvement from base', base)
                else:
                    arrgg = 0
                    base = best_val
                # early stopping
                if arrgg == patience:
                    print('Early stop at {}'.format(itr))
                    return best_fitness, best_wmatrix
            self.update_learning_config(max_iter, itr)
            print('best:{}, overal:{}'.format(best_fitness, swam_fitness))
            if validation:
                print('validation:{}'.format(best_val))
        return best_fitness, best_wmatrix
    
    def update_state(self):
        #print('Update State...', end='')
        best_particle = None
        for particle in self.swams:
            if particle.best is None:
                particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
            elif particle.fitness < particle.best['fitness']:
                particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
                print('update local,', particle.fitness)
            if particle.best['fitness'] < self.global_best_fitness:
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
                weight = self.clip(p+velocity, self.weight_constraint)
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