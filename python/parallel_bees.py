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
    def __init__(self, idx, n_feature, units, bias, init_fn, weight_constraint, velocity_constraint):
        self._idx = idx
        self._fitness = sys.float_info.max
        self._val_fitness = sys.float_info.max
        self._best = None # {'fitness':0, 'wmatrix':[]}
        self.__build__(n_feature, units, bias, init_fn, weight_constraint, velocity_constraint)

    def __build__(self, n_feature, units, bias, init_fn, weight_constraint, velocity_constraint):
        n_params = get_nparams(n_feature, units, bias)
        #self._velocity_matrix = np.array([np.random.uniform(velocity_constraint[0],velocity_constraint[1],1) for i in range(n_params)], dtype='float32')
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

#   args_input = [dict(agent=copy.deepcopy(obj), wmat=particle.wmatrix, x=batch_x, y=batch_y, \
#                       bs=batch_size) for particle in self.swams]
def parallel_evaluate(args):
    agent = args['agent']
    wmat = args['wmat']
    x = args['x']
    y = args['y']
    bs = args['bs']
    agent.__load_weight__(wmat)
    fitness, _ = agent._evaluate(x, y, bs)
    return args['idx'], fitness, None

#   input: for v,p,l,g,m in zip(particle.vmatrix, particle.wmatrix, particle.best['wmatrix'], self.global_best_matrix, mean_matrix):
#   output: wmatrix, vmatrix
def parallel_shift(args):
    pass

class BEES(object):
    def __init__(self, config):
        print('building object according to ', config)
        self.M = config['m']
        assert len(config['weight_constraint']) == 2
        self.weight_constraint = config['weight_constraint']
        self.C1, self.C2, self.C3 = config['c1'], config['c2'], config['c3']
        self.MAX_C1, self.MIN_C1 = config['max_c1'], config['min_c1']
        self.MAX_C2, self.MIN_C2 = config['max_c2'], config['min_c2']
        self.MAX_C3, self.MIN_C3 = config['max_c3'], config['min_c3']
        self.velocity_constraint = config['velocity_constraint']
        self.W = config['w']
        self.W_Decay = config['w_decay']
        self.initializer = config['initializer'] # unused
        self.global_best_matrix = None
        self.global_best_fitness = sys.float_info.max
        
    def __build__(self, seq_len, n_feature, units, bias, init_fn):
        self.seq_len = seq_len
        self.swams = [Particle(m, n_feature, units, bias, init_fn, self.weight_constraint, self.velocity_constraint) for m in range(self.M)]
        #print('finish building', len(self.swams))
    
    def fit(self, obj, x, y, max_iter, goal, pack_goal=0.05, batch_size=5, validation=False, x_val=[], y_val=[], validation_batch_size=5):
        min_improv = 0.0001
        patience = max_iter
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
                # begin multiprocessing
                args_input = [dict(idx=particle.idx, agent=copy.deepcopy(obj), wmat=particle.wmatrix, x=batch_x, y=batch_y, bs=len(batch_x)) for particle in self.swams]
                pool = mp.Pool(processes=2)
                outputs = pool.map(parallel_evaluate, args_input)
                pool.close()
                # end multiprocessing
                for value in outputs:
                    self.swams[value[0]].fitness = value[1]
                best_fitness, best_wmatrix, swam_fitness = self.update_state()
                if best_fitness <= goal or swam_fitness <= pack_goal:
                    return best_fitness, best_wmatrix
                self.update_pool()
            obj.__load_weight__(best_wmatrix)
            y_pred = obj._predict(x)
            k = 0
            for pred, true in zip(y_pred, y.values.tolist()):
                print(pred, true)
                k += 1
                if k == 5:
                    break
            # validation
            if validation:
                base = best_val
                obj.__load_weight__(best_wmatrix)
                for i in range(0, len(x_val), validation_batch_size):
                    batch_x = x_val[i:i+validation_batch_size]
                    batch_y = y_val[i:i+validation_batch_size]
                    val_fitness, _ = obj._evaluate(batch_x, batch_y, len(batch_x))
                    if val_fitness < best_val:
                        best_val = val_fitness
                if base - best_val < min_improv:
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
                print('update local,', particle.idx, ':', particle.fitness)
            if particle.best['fitness'] < self.global_best_fitness:
                #print('update global', particle.best['fitness'])
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
        mean_matrix = self.get_topk_matrix(self.M//10)
        for particle in self.swams:
            #print('\n\nParticle:', particle.idx)
            new_matrix = []
            new_velocity = []
            for v,p,l,g,m in zip(particle.vmatrix, particle.wmatrix, particle.best['wmatrix'], self.global_best_matrix, mean_matrix):
                #print(v, p, l, g, m)
                _velocity = (self.W * v) + \
                (self.C1 * np.random.uniform(0,1,1) * (l-p)) + \
                (self.C2 * np.random.uniform(0,1,1) * (g-p)) + \
                (self.C3 * np.random.uniform(0,1,1) * (m-p))
                velocity = self.clip(_velocity+v, self.velocity_constraint)
                #print('velo', velocity)
                new_velocity.append(velocity)
                weight = self.clip(p+velocity+ np.random.uniform(-0.1, 0.1, 1)*p, self.weight_constraint)
                new_matrix.append(weight)
            particle.wmatrix = np.array(new_matrix, dtype='float32')
            particle.vmatrix = np.array(new_velocity, dtype='float32')
            #print(particle.idx, particle.wmatrix.shape, particle.vmatrix.shape)

    def update_learning_config(self, max_iter, itr):
        pass
        '''
        #print('Update Learning Configuration')
        #self.C1 = self.MAX_C1 + (self.MAX_C1 - self.MIN_C1) / max_iter * itr
        #tmp_c1 = np.max([self.MAX_C1 - (self.MAX_C1 * itr / max_iter), self.MIN_C1])
        tmp_c1 = self.MAX_C1 - (self.MAX_C1-self.MIN_C1) / max_iter * itr
        tmp_c2 = self.MAX_C2 - (self.MAX_C2-self.MIN_C2) / max_iter * itr
        #tmp_c2 = np.max([self.MAX_C2 - (self.MAX_C2 * itr / max_iter), self.MIN_C2])
        tmp_c3 = self.MAX_C3 - (self.MAX_C3-self.MIN_C3) / max_iter * itr
        #total = tmp_c1 + tmp_c2 + tmp_c3
        self.C1 = tmp_c1# / total
        self.C2 = tmp_c2# / total
        self.C3 = tmp_c3# / total
        self.W = 1.6 - (1.6-0.4)/max_iter*itr
        print('\nItr: {}\nC1: {}, C2:{}, C3:{}, W:{}'.format(itr, self.C1, self.C2, self.C3, self.W))
        '''
        
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