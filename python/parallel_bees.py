'''
BEES
'''
# dependency
import multiprocessing as mp
import numpy as np
import copy
import sys
from math import sqrt
from sklearn.utils import shuffle
from utility import trim_tail, add, devision, get_nparams

class Particle(object):
    def __init__(self, idx, n_feature, units, bias, init_fn, weight_constraint):
        self._idx = idx
        self._fitness = sys.float_info.max
        self._val_fitness = sys.float_info.max
        self._best = None # {'fitness':0, 'wmatrix':[]}
        self.__build__(n_feature, units, bias, init_fn, weight_constraint)

    def __build__(self, n_feature, units, bias, init_fn, weight_constraint):
        n_params = get_nparams(n_feature, units, bias)
        self._velocity_matrix = np.array([np.zeros(1) for i in range(n_params)], dtype='float32')
        self._weight_matrix = list()
        prev = n_feature
        for u, b, fn in zip(units, bias, init_fn):
            layer_weight = fn(prev, u, b, weight_constraint)
            self._weight_matrix.extend(layer_weight)
            prev = u
        self._weight_matrix = np.array(self._weight_matrix, dtype='float32')

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

def parallel_evaluate(args):
    agent = args['agent']
    wmat = args['wmat']
    x = args['x']
    y = args['y']
    bs = args['bs']
    agent.__load_weight__(wmat)
    fitness, _ = agent._evaluate(x, y, bs)
    return args['idx'], fitness, None

class BEES(object):
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
        self.decay = config['decay']
        self.initializer = config['initializer'] # unused
        self.global_best_matrix = None
        self.global_best_fitness = sys.float_info.max
        self.n_workers = config['n_workers']

    def __build__(self, seq_len, n_feature, units, bias, init_fn):
        self.seq_len = seq_len
        self.swams = [Particle(m, n_feature, units, bias, init_fn, self.weight_constraint) for m in range(self.M)]
    
    def fit(self, obj, x, y, max_iter, goal, pack_goal=0.05, batch_size=5, verbose=False):
        min_improv = 0.0001
        patience = max_iter
        best_val = sys.float_info.max
        best_fitness, swam_fitness, best_wmatrix = sys.float_info.max, sys.float_info.max, None
        for itr in range(max_iter):
            print('Iteration:', itr)
            seed = np.random.randint(1000)
            x = shuffle(x, random_state=seed)
            y = shuffle(y, random_state=seed)
            '''
            if validation:
                x_val = shuffle(x_val, random_state=seed)
                y_val = shuffle(y_val, random_state=seed)
            '''
            # training
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                # begin multiprocessing
                args_input = [dict(idx=particle.idx, agent=copy.deepcopy(obj), wmat=particle.wmatrix, x=batch_x, y=batch_y, bs=batch_size) for particle in self.swams]
                pool = mp.Pool(processes=self.n_workers)
                outputs = pool.map(parallel_evaluate, args_input)
                pool.close()
                # end multiprocessing
                for value in outputs:
                    self.swams[value[0]].fitness = value[1]
                best_fitness, best_wmatrix, swam_fitness = self.update_state()
                '''
                # these condition will never satisfied
                if best_fitness <= goal or swam_fitness <= pack_goal:
                    return best_fitness, best_wmatrix
                '''
                self.update_pool()
            # end of iteration
            self.update_learning_config(max_iter, itr)
            if verbose:
                print('best:{}, overal:{}'.format(best_fitness, swam_fitness))
                obj.__load_weight__(best_wmatrix)
                y_pred = obj._predict(x)
                k = 0
                for pred, true in zip(y_pred, y.values.tolist()):
                    print(pred, true)
                    k += 1
                    if k == 5:
                        break
        return best_fitness, best_wmatrix
    
    def update_state(self):
        #print('Update State...', end='')
        best_particle = None
        for particle in self.swams:
            if particle.best is None:
                particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
            elif particle.fitness < particle.best['fitness']:
                particle.best = {'fitness':particle.fitness, 'wmatrix': particle.wmatrix}
                print('update local,', particle.idx, ':', particle.fitness)
            if particle.best['fitness'] < self.global_best_fitness:
                print('update global', particle.best['fitness'])
                self.global_best_fitness = particle.best['fitness']
                best_particle = particle.idx
        swam_fitness = [p.fitness for p in self.swams]
        if best_particle is not None:
            self.global_best_matrix = copy.deepcopy(self.swams[best_particle].best['wmatrix'])
        return self.global_best_fitness, self.global_best_matrix, np.sum(swam_fitness) / len(swam_fitness)
    
    def get_topk_matrix(self, k=5):
        id_list = list()
        for _ in range(k):
            best_id = None
            best = sys.float_info.max
            for particle in self.swams:
                if particle.fitness < best and particle.idx not in id_list:
                    best = particle.fitness
                    best_id = particle.idx
            id_list.append(best_id)
        topk_matrix = list()
        for particle in self.swams:
            if particle.idx in id_list:
                topk_matrix.append(particle.wmatrix)
        topk_matrix = np.array(topk_matrix, dtype='float32')
        mean_matrix = np.zeros(self.seq_len)
        for mat in topk_matrix:
            mean_matrix = add(mean_matrix, mat)
        return devision(mean_matrix, k)

    def get_mean_matrix(self):
        mean_matrix = np.zeros(self.seq_len)
        for particle in self.swams:
            mean_matrix = add(mean_matrix, particle.wmatrix)
        return devision(mean_matrix, self.M)

    def update_pool(self):
        #print('Update Pool')
        mean_matrix = self.get_topk_matrix(10)
        for particle in self.swams:
            #print('\n\nParticle:', particle.idx)
            new_matrix = []
            new_velocity = []
            for v,p,l,g in zip(particle.vmatrix, particle.wmatrix, particle.best['wmatrix'], self.global_best_matrix):
                #print(v, p, l, g, m)
                _velocity = (self.W * v) + \
                (self.C1 * np.random.uniform(0,1,1) * (l-p)) + \
                (self.C2 * np.random.uniform(0,1,1) * (g-p))
                velocity = self.clip(_velocity+v, self.velocity_constraint)
                #print('velo', velocity)
                new_velocity.append(velocity)
                weight = self.clip(p+velocity+ np.random.uniform(-0.1, 0.1, 1)*p, self.weight_constraint)
                new_matrix.append(weight)
            particle.wmatrix = np.array(new_matrix, dtype='float32')
            particle.vmatrix = np.array(new_velocity, dtype='float32')
            #print(particle.idx, particle.wmatrix.shape, particle.vmatrix.shape)

    def update_learning_config(self, max_iter, itr):
        if itr > 20 and itr % 20 == 0:
            self.C1 = np.max([self.MIN_C1, self.C1 * self.decay])
            self.C2 = np.min([self.MAX_C2, self.C2 + self.C2 * (1 - self.decay)])
            self.W = self.W * self.decay

    def clip(self, x, bound):
        #print('clip', type(x), type(bound[0]))
        x = x if x > bound[0] else np.array([bound[0]])
        return x if x < bound[1] else np.array([bound[1]])
    
    def get_agent_wmatrix(self, idx):
        return self.swams[idx].wmatrix
    
    @property
    def m(self):
        return self.M

def f1(args):
    return args['id'], args['gender']

if __name__ == '__main__':
    print('Happy MP')
    data_dict = dict(name='jonah', age='29', gender='male')
    #f1(name='jonah', age='29', gender='male')
    pool = mp.Pool(processes=8)
    inputs = [dict(id=i, gender=i%2) for i in range(1000)]
    outputs = pool.map(f1, inputs)
    print(outputs)
    print('waiting')
    #pool.join()
    print('closing')
    pool.close()