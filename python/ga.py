'''
GA
'''
# dependency
import numpy as np
import copy
import sys
import random
from numpy.lib.arraysetops import unique
from sklearn.utils import shuffle
from functools import reduce
from utility import trim_tail, summation
from regularizers import l1_regularizer, l2_regularizer

class Gene(object):
    def __init__(self, idx, seq_len, aim, weight_constraint = None):
        self._idx = idx
        if weight_constraint is not None:
            self._weight_matrix = np.array([np.random.uniform(weight_constraint[0], weight_constraint[1],1) for i in range(seq_len)], dtype='float32')
            self._weight_matrix = trim_tail(self._weight_matrix)
            assert len(self._weight_matrix) == seq_len, 'id {}, {}'.format(idx, len(self._weight_matrix))
        else:
            self._weight_matrix = None
        if not aim:
            self._fitness = sys.float_info.max
        else:
            self._fitness = sys.float_info.min
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

class GA(object):
    def __init__(self, config):
        self.M = config['m']
        self.x_rate = config['x_rate']
        self.m_rate = config['m_rate']
        self.weight_constraint = config['weight_constraint']
        self.global_best_matrix = None
        self.idx_tracker = 0
        self.rr = config['radioactive_rating'][1] if type(config['radioactive_rating']) == list else config['radioactive_rating']
        self.gf = config['grow_factor'][1] if type(config['grow_factor']) == list else config['grow_factor']
        self.crossover_rate = config['x_rate'][1] if type(config['x_rate']) == list else config['x_rate']
        self.mutation_rate = config['m_rate'][0] if type( config['m_rate']) == list else  config['m_rate']
        self.radioactive_rating = config['radioactive_rating']
        self.grow_factor = config['grow_factor']
        self.scale_hp = config['scale_hyperparameter']
        self.regularizer = config['regularizer']
        
    def __build__(self, seq_len, aim):
        self.seq_len = seq_len
        self.aim = aim
        self.population = [Gene(m, seq_len, aim, self.weight_constraint) for m in range(self.M)]
        self.idx_tracker += self.M
        self.global_best_fitness = sys.float_info.min if aim else sys.float_info.max

    def fit(self, obj, x, y, max_iter, goal = 0, batch_size = 0, loss='categorical_crossentropy', verbose = 1):
        best_wmatrix = self.population[0].wmatrix
        history = list()
        for itr in range(max_iter):
            for m in range(len(self.population)):
                wmatrix = self.population[m].wmatrix
                obj.__load__(wmatrix)
                metrics = obj.evaluate(x, y, training=True, verbose = 0)
                self.population[m].fitness = metrics[loss] 
                if self.regularizer:
                    self.population[m].fitness += self.regularizer(wmatrix)
            best_fitness, best_wmatrix, _ = self.update_state(verbose)
            history.append(best_fitness)
            self.update_pool()
            self.update_learning_config(max_iter, itr)
            if verbose == 1:
                print('Iteration {}/{}: \t{}'.format(itr, max_iter, best_fitness))
        return best_fitness, best_wmatrix, history

    def update_state(self, verbose = 1):
        best_gene = None
        if not self.aim:
            for gene in self.population:
                if gene.best is None:
                    gene.best = {'fitness': gene.fitness, 'wmatrix': gene.wmatrix}
                    #print('child fitness', gene.fitness)
                elif gene.fitness <= gene.best['fitness']:
                    gene.best = {'fitness':gene.fitness, 'wmatrix': gene.wmatrix}
                    #print('update local,', gene.fitness)
                if gene.best['fitness'] <= self.global_best_fitness:
                    if verbose == 1:
                        print('update global', gene.best['fitness'])
                    self.global_best_fitness = gene.best['fitness']
                    best_gene = gene.idx
        else:
            for gene in self.population:
                if gene.best is None:
                    gene.best = {'fitness': gene.fitness, 'wmatrix': gene.wmatrix}
                    #print('child fitness', gene.fitness)
                elif gene.fitness >= gene.best['fitness']:
                    gene.best = {'fitness':gene.fitness, 'wmatrix': gene.wmatrix}
                    #print('update local,', gene.fitness)
                if gene.best['fitness'] >= self.global_best_fitness:
                    if verbose == 1:
                        print('update global', gene.best['fitness'])
                    self.global_best_fitness = gene.best['fitness']
                    best_gene = gene.idx
        population_fitness = [p.fitness for p in self.population]
        if best_gene is not None:
            self.global_best_matrix = copy.deepcopy(self.get_agent_wmatrix(best_gene))
        return self.global_best_fitness, self.global_best_matrix, np.mean(population_fitness)

    def update_pool(self, chaotic_rate = 0.5):
        self.sampling()
        mutate_lst = list()
        for _ in range(int(len(self.population) * chaotic_rate)):
            # Begin Crossover phase
            if np.random.uniform(0,1,1) > 1 - self.crossover_rate:
                gene_a, gene_b = self.selection()
                child = self.crossover(self.population[gene_a].wmatrix, self.population[gene_b].wmatrix)
                assert len(child) == len(self.population[gene_a].wmatrix)
                new_gene = Gene(self.idx_tracker, len(child), self.aim, None)
                self.idx_tracker += 1
                new_gene.wmatrix = copy.deepcopy(child)
                self.population.append(new_gene)
            # End Crossover phase
            # Begin Mutation Phase
            if np.random.uniform(0,1,1) > 1 - self.mutation_rate:
                # in case all are mutated, rare case
                if chaotic_rate == 1 and len(mutate_lst) == len(self.M):
                    continue
                gene_c = int(np.random.uniform(0,1,1)*self.M)%self.M
                while gene_c in mutate_lst:
                    gene_c = int(np.random.uniform(0,1,1)*self.M)%self.M
                mutate_lst.append(gene_c)
                alien_seq = self.mutate(self.population[gene_c].wmatrix)
                new_gene = Gene(self.idx_tracker, self.seq_len, self.aim, None)
                new_gene.wmatrix = alien_seq
                self.idx_tracker += 1
                self.population.append(new_gene)
            # End Mutation Phase

    def sampling(self):
        #print('came in:', len(self.population))
        unique_gene = list(np.unique([ np.sum(n.wmatrix) for n in self.population ]))
        #print('to refresh', len(self.population) - len(unique_gene))
        fitness_list = [p.fitness for p in self.population]
        shift = (np.mean(fitness_list) - np.min(fitness_list)) / 2
        '''
        if shift == 0:
            shift = 0.001
        '''
        if len(unique_gene) == len(self.population):
            self.population.sort(key=lambda p: p.fitness + np.random.uniform(-shift, shift, 1), \
                reverse = self.aim)
        else:
            # inefficient operation
            output_list = list()
            while len(unique_gene) != 0:
                s = unique_gene.pop()
                for gene in self.population:
                    if np.sum(gene.wmatrix) == s:
                        output_list.append(gene)
                        break
            self.population = output_list
            self.population.sort(key=lambda p: p.fitness + np.random.uniform(-shift, shift, 1), \
                reverse = self.aim)
            while len(self.population) < self.M: # insertion point doesn't increased the odds of being selected
                self.population.insert(0, Gene(self.idx_tracker, self.seq_len, self.aim, self.weight_constraint))
        self.population = self.population[:self.M]

    def selection(self):
        # adaptive shift
        fitness_list = [p.fitness for p in self.population]
        shift = (np.max(fitness_list) - np.min(fitness_list)) / 2
        '''
        if shift == 0:
            shift = 0.001
        '''
        self.population.sort(key=lambda p: p.fitness + np.random.uniform(-shift, shift, 1), reverse = self.aim)
        return 0, 1

    def crossover(self, seq1, seq2, method='single'): # single point, two points
        #print('crossover', type(seq1), type(seq2))
        assert seq1.shape[0] == seq2.shape[0] == self.seq_len
        child = None
        if method =='single':
            p1 = int(self.seq_len * np.random.uniform(0,1,1))% self.seq_len
            child = copy.deepcopy(seq1)[:p1]
            child = np.append(child, copy.deepcopy(seq2)[p1:])
        elif method == 'two':
            cut_1, cut_2 = int(self.seq_len * np.random.uniform(0,1,1))% self.seq_len, int(self.seq_len * np.random.uniform(0,1,1)) % self.seq_len
            while cut_1 >= cut_2:
                cut_1, cut_2 = int(self.seq_len * np.random.uniform(0,1,1))% self.seq_len, int(self.seq_len * np.random.uniform(0,1,1)) % self.seq_len
            child = copy.deepcopy(seq1[:cut_1])
            child = np.append(child, copy.deepcopy(seq2[cut_1:cut_2]))
            child = np.append(child, copy.deepcopy(seq1[cut_2:]))
            assert child.shape[0] == self.seq_len, '{}:{}'.format(child.shape[0], self.seq_len)
        else:
            # uniform
            mask = np.random.randint(0, 2, self.seq_len)
            child = copy.deepcopy(seq1)
            for idx in range(self.seq_len):
                if mask[idx] == 1:
                    child[idx] = seq2[idx]
        return child

    def mutate(self, seq, rate = 0.05):
        alien_seq = copy.deepcopy(seq)
        n_points = np.min([int(len(seq) *rate), 1])
        point_set = set() # prevent mutating the same point
        for _ in range(n_points):
            point = int(self.seq_len * np.random.uniform(0,1,1)) % self.seq_len
            while point in point_set:
                point = int(self.seq_len * np.random.uniform(0,1,1)) % self.seq_len
            point_set.add(point)
            new_code = alien_seq[point] - (self.rr * np.random.uniform(0,1,1)) + (self.gf * np.random.uniform(0,1,1))
            alien_seq[point] = self.clip(new_code, self.weight_constraint)
        return alien_seq
    
    def clip(self, x, bound):
        x = x if x > bound[0] else np.array([bound[0]])
        return x if x < bound[1] else np.array([bound[1]])

    def update_learning_config(self, max_iter, itr):
        if self.scale_hp:
            self.rr = self.radioactive_rating[1] - (self.radioactive_rating[1] - self.radioactive_rating[0]) / max_iter * (max_iter - itr)
            self.gf = self.grow_factor[1] - (self.grow_factor[1] - self.grow_factor[0]) / max_iter * (max_iter - itr)
            self.crossover_rate = self.x_rate[1] - (self.x_rate[1] - self.x_rate[0]) / max_iter * (max_iter - itr)
            self.mutation_rate = self.m_rate[0] + (self.m_rate[1] - self.m_rate[0]) / max_iter * (max_iter - itr)
    
    # get wmatrix by idx
    def get_agent_wmatrix(self, idx):
        for gene in self.population:
            if gene.idx == idx:
                return gene.wmatrix
    
