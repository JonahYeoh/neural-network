'''
GA
'''
# dependency
import numpy as np
import copy
import sys
import random
from sklearn.utils import shuffle
from functools import reduce
from utility import trim_tail, summation

class Gene(object):
    def __init__(self, idx, seq_len):
        self._idx = idx
        self._weight_matrix = np.array([np.random.uniform(0,1,1) for i in range(seq_len)], dtype='float32')
        self._weight_matrix = trim_tail(self._weight_matrix)
        self._fitness = sys.float_info.max
        self._best = None # {'fitness':0, 'wmatrix':[]}

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

def sort_by_fitness(arg, alpha=100):
    return arg.fitness * np.random.randint(alpha) / alpha

class GA(object):
    def __init__(self, config):
        self.M = config['m']
        self.x_rate = config['x_rate']
        self.m_rate = config['m_rate']
        assert len(config['weight_constraint']) == 2
        self.weight_constraint = config['weight_constraint']
        self.radioactive_rating = config['radioactive_rating']
        self.grow_factor = config['grow_factor']
        self.global_best_matrix = None
        self.global_best_fitness = sys.float_info.max
        self.sort_fn = sort_by_fitness
        self.idx_tracker = 0
        self.rr = config['radioactive_rating'][0]
        self.gf = config['grow_factor'][0]
        self.crossover_rate = config['x_rate'][0]
        self.mutation_rate = config['m_rate'][0]

    def __build__(self, seq_len):
        self.seq_len = seq_len
        self.population = [Gene(m, seq_len) for m in range(self.M)]
        self.idx_tracker += self.M

    '''
    fit with x and y and return a fittest solution (wmatrix), 
    not necessary a perfect solution.
    termination condition: 
        itr == max_iter or
        best_fitness <= goal or
        pack_fitness <= pack_goal # this assumption
    '''
    def fit(self, obj, x, y, max_iter, goal, pack_goal=0.05, batch_size=5):
        for itr in range(max_iter):
            print('Iteration:', itr)
            seed = np.random.randint(1000)
            x = shuffle(x, random_state=seed)
            y = shuffle(y, random_state=seed)
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                for m in range(self.M):
                    #print('.', end='')
                    wmatrix = self.population[m].wmatrix
                    obj.__load_weight__(wmatrix)
                    self.population[m].fitness, m_board = obj._evaluate(batch_x, batch_y, len(batch_x))
                best_fitness, best_wmatrix, population_fitness = self.update_state()
                if best_fitness <= goal or population_fitness <= pack_goal:
                    return best_fitness, best_wmatrix
                self.update_pool()
            self.update_learning_config(max_iter, itr)
            print('best:{}, overal:{}'.format(best_fitness, population_fitness))
            '''
            for name, result in m_board.items():
                print('{}: {}'.format(name, result))
            '''
        return best_fitness, best_wmatrix

    def update_state(self):
        #print('Update State...', end='')
        best_gene = None
        for gene in self.population:
            if gene.best is None:
                gene.best = {'fitness':gene.fitness, 'wmatrix': gene.wmatrix}
            elif gene.fitness < gene.best['fitness']:
                gene.best = {'fitness':gene.fitness, 'wmatrix': gene.wmatrix}
                print('update local,', gene.fitness)
            if gene.best['fitness'] < self.global_best_fitness:
                print('update global', gene.best['fitness'])
                self.global_best_fitness = gene.best['fitness']
                best_gene = gene.idx
        # sum of element get inf
        # tried round and increased precision to f64
        population_fitness = [p.fitness for p in self.population]
        '''
        rounded_pop = np.round(population_fitness, decimals=6)
        print('population sum', np.sum(rounded_pop, dtype='float64'), len(population_fitness))
        '''
        if best_gene is not None:
            self.global_best_matrix = copy.deepcopy(self.get_agent_wmatrix(best_gene))
        return self.global_best_fitness, self.global_best_matrix, np.sum(population_fitness) / len(population_fitness)

    def update_pool(self):
        # Sampling Phase
        self.sampling(True)
        for step in range(self.M):
            #print('Step', step)
            # Begin Crossover phase
            if np.random.uniform(0,1,1) < self.crossover_rate:
                gene_a, gene_b = self.selection()
                child = self.crossover(self.population[gene_a].wmatrix, self.population[gene_b].wmatrix)
                new_gene = Gene(self.idx_tracker, len(child))
                self.idx_tracker += 1
                new_gene.wmatrix = child
                self.population.append(new_gene)
            # End Crossover phase
            # Begin Mutation Phase
            if np.random.uniform(0,1,1) < self.mutation_rate:
                gene_c = int(np.random.uniform(0,1,1)*self.M)%self.M
                alien = self.mutate(self.population[gene_c].wmatrix)
                new_gene = Gene(self.idx_tracker, len(alien))
                new_gene.wmatrix = alien
                self.idx_tracker += 1
                self.population.append(new_gene)
            # End Mutation Phase

    def sampling(self, do_shuffle=True):
        self.population.sort(key=self.sort_fn)
        self.population = copy.deepcopy(self.population[:self.M])
        if do_shuffle:
            self.population = shuffle(self.population)

    def selection(self):
        male, female = int(np.random.uniform(0,1,1)*self.M)%self.M, int(np.random.uniform(0,1,1)*self.M)%self.M
        while male == female:
            male, female = int(np.random.uniform(0,1,1)*self.M)%self.M, int(np.random.uniform(0,1,1)*self.M)%self.M
        return male, female

    def crossover(self, seq1, seq2):
        #print('crossover', type(seq1), type(seq2))
        assert seq1.shape[0] == seq2.shape[0]
        cut_1, cut_2 = int(seq1.shape[0] * np.random.uniform(0,1,1))% seq1.shape[0], int(seq1.shape[0] * np.random.uniform(0,1,1)) % seq1.shape[0]
        while cut_1 >= cut_2:
            cut_1, cut_2 = int(seq1.shape[0] * np.random.uniform(0,1,1))% seq1.shape[0], int(seq1.shape[0] * np.random.uniform(0,1,1)) % seq1.shape[0]
        child = copy.deepcopy(seq1[:cut_1])
        child = np.append(child, copy.deepcopy(seq2[cut_1:cut_2]), 0)
        child = np.append(child, copy.deepcopy(seq1[cut_2:]), 0)
        assert child.shape[0] == self.seq_len, '{}:{}'.format(child.shape[0], self.seq_len)
        return child

    def mutate(self, seq):
        alien_seq = copy.deepcopy(seq)
        n_points = int(len(seq) * 0.05)
        pin_sets = set()
        for points in range(n_points):
            pin = int(len(seq) * np.random.uniform(0,1,1)) % len(seq)
            while pin in pin_sets:
                pin = int(len(seq) * np.random.uniform(0,1,1)) % len(seq)
            new_code = alien_seq[pin] - (self.rr * (1 if np.random.uniform(0,1,1) < 0.5 else -1)) * self.gf
            alien_seq[pin] = self.clip(new_code, self.weight_constraint)
        return alien_seq
    
    def clip(self, x, bound):
        #print('clip', type(x), type(bound[0]))
        x = x if x > bound[0] else np.array([bound[0]])
        return x if x < bound[1] else np.array([bound[1]])

    def update_learning_config(self, max_iter, itr):
        self.rr = self.radioactive_rating[0] + (self.radioactive_rating[1] - self.radioactive_rating[0]) / max_iter * (max_iter - itr)
        self.gf = self.grow_factor[0] + (self.grow_factor[1] - self.grow_factor[0]) / max_iter * (max_iter - itr)
        self.crossover_rate = self.x_rate[0] + (self.x_rate[1] - self.x_rate[0]) / max_iter * (max_iter - itr)
        self.mutation_rate = self.m_rate[0] + (self.m_rate[1] - self.m_rate[0]) / max_iter * (max_iter - itr)

    def get_agent_wmatrix(self, idx):
        for gene in self.population:
            if gene.idx == idx:
                return gene.wmatrix
    
