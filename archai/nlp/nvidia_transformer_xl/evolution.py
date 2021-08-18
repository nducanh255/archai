import random
import os
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch

from archai.nlp.nvidia_transformer_xl.utils import get_model, process_parameters, get_latency


model_config_defaults = {'d_head': None, 'n_token': 267736, 'dropout': 0.1, 'dropatt': 0.0, \
                        'd_embed': None, 'div_val': 4, 'pre_lnorm': False, 'tgt_len': 192, 'ext_len': 0, 'mem_len': 192, \
                        'same_length': False,'attn_type': 0,'clamp_len': -1, 'sample_softmax': -1, \
                        'cutoffs': [19997, 39997, 199997], 'tie_projs': [False, True, True, True], 'tie_weight': True, 'dtype': None}

class Converter(object):
    def __init__(self, n_layer_choice, d_model_choice, d_inner_choice, n_head_choice):
        self.n_layer_choice = n_layer_choice
        self.d_model_choice = d_model_choice
        self.d_inner_choice = d_inner_choice
        self.n_head_choice = n_head_choice
        
        self.max_n_layer = self.n_layer_choice[-1]
        
    def config2gene(self, config):
        gene = []

        sample_n_layer = config['n_layer']
        gene.append(config['d_model'])
        gene.append(sample_n_layer)

        for i in range(self.max_n_layer):
            if i < sample_n_layer:
                gene.append(config['d_inner'][i])
            else:
                gene.append(config['d_inner'][0])

        for i in range(self.max_n_layer):
            if i < sample_n_layer:
                gene.append(config['n_head'][i])
            else:
                gene.append(config['n_head'][0])

        return gene

    def gene2config(self, gene):
        config = {'d_model': None, 'n_layer': None, 'd_inner': None, 'n_head': None}

        current_index = 0
        config['d_model'] = gene[current_index]
        current_index += 1

        config['n_layer'] = gene[current_index]
        current_index += 1

        config['d_inner'] = gene[current_index: current_index + config['n_layer']]
        current_index += self.max_n_layer

        config['n_head'] = gene[current_index: current_index + config['n_layer']]
        current_index += self.max_n_layer

        return config

    def get_gene_choice(self, d_inner_min=None):
        gene_choice = []

        gene_choice.append(self.d_model_choice)
        gene_choice.append(self.n_layer_choice)

        for i in range(self.max_n_layer):
            if d_inner_min is not None:
                gene_choice.append(list(range(d_inner_min, self.d_inner_choice[-1], 50)))
            else:
                gene_choice.append(self.d_inner_choice)

        for i in range(self.max_n_layer):
            gene_choice.append(self.n_head_choice)

        return gene_choice


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Evolution(object):
    def __init__(self, results_path, population_size=125, parent_size=25, mutation_size=50, mutation_prob=0.3, crossover_size=50, n_iter=30,
                n_layer_choice=[5,6,7,8], d_model_choice=[64,128,256,512], d_inner_choice=list(range(512, 2048, 50)), n_head_choice=[2,4,8],
                param_constraint=4e6, latency_scale=1.):
        
        self.results_path = results_path
        self.population_size = population_size
        self.parent_size = parent_size
        self.mutation_size = mutation_size
        self.mutation_prob = mutation_prob
        self.crossover_size = crossover_size
        assert self.population_size == self.parent_size + self.mutation_size + self.crossover_size
        self.n_iter = n_iter

        self.converter = Converter(n_layer_choice, d_model_choice, d_inner_choice, n_head_choice)
        self.gene_choice = self.converter.get_gene_choice()
        # print('initial gene choice:', self.gene_choice)
        self.gene_len = len(self.gene_choice)
        
        self.param_constraint = param_constraint
        self.profile()
        self.latency_scale = latency_scale
        
        self.best_config = None
        self.pareto = {'population':[], 'params':[], 'latencies':[]}
        self.all_population = []
        self.all_params = []
        self.all_latencies = []

    def run_evo_search(self):
        population = self.random_sample(self.population_size)
        self.all_population = population

        logs = {'population': [population], 'parents': [], 'parents_scores': [], 'best_config': [], 'pareto': []}

        parents_score = []
        parents_params = []
        parents_latencies = []
        for i in range(self.n_iter):
            idx = 0 if i==0 else self.parent_size
            
            print(f"| Start Iteration {i}:")
            population_scores_unseen, population_params_unseen, population_latencies_unseen = self.get_scores(population[idx:])
            population_scores = parents_score + population_scores_unseen
            population_params = parents_params + population_params_unseen
            population_latencies = parents_latencies + population_latencies_unseen
            print(f"| Iteration {i}, Max score: {max(population_scores)}")
            
            sorted_ind = np.array(population_scores).argsort()[::-1][:self.parent_size]
            self.best_config = self.converter.gene2config(population[sorted_ind[0]])
            print(f"| Config for highest score model: {self.best_config}")
            print(f"| nParams for highest score model: {population_params[sorted_ind[0]]}")
            print(f"| Latency for highest score model: {population_latencies[sorted_ind[0]]}")

            parents_population = [population[m] for m in sorted_ind]
            parents_score = [population_scores[m] for m in sorted_ind]
            parents_params = [population_params[m] for m in sorted_ind]
            parents_latencies = [population_latencies[m] for m in sorted_ind]

            mutate_population = []
            k = 0
            while k < self.mutation_size:
                mutated_gene = self.mutate(random.choices(parents_population)[0])
                if self.satisfy_constraints(mutated_gene):
                    mutate_population.append(mutated_gene)
                    k += 1

            crossover_population = []
            k = 0
            while k < self.crossover_size:
                crossovered_gene = self.crossover(random.sample(parents_population, 2))
                if self.satisfy_constraints(crossovered_gene):
                    crossover_population.append(crossovered_gene)
                    k += 1

            population = parents_population + mutate_population + crossover_population

            logs['population'].append(population)
            logs['parents'].append(parents_population)
            logs['parents_scores'].append(parents_score)
            logs['best_config'].append(self.best_config)
            logs['pareto'].append(self.pareto)

            
            self.all_params += population_params
            self.all_latencies += population_latencies
            self.update_pareto_front()
            self.all_population += population
            
            self.plot_samples(iter=i)

        path_to_pkl = os.path.join(self.results_path, 'logs.pkl')
        with open(path_to_pkl, 'wb') as f:
            pickle.dump(logs, f)
        
        return self.best_config

    def crossover(self, genes):
        crossovered_gene = []
        for i in range(self.gene_len):
            if np.random.uniform() < 0.5:
                crossovered_gene.append(genes[0][i])
            else:
                crossovered_gene.append(genes[1][i])

        return crossovered_gene

    def mutate(self, gene):
        mutated_gene = []
        d_inner_min = None
        gene_choice = self.gene_choice
        for i in range(self.gene_len):
            if i==1:
                d_inner_min = int(1.7 * mutated_gene[-1])
                gene_choice = self.converter.get_gene_choice(d_inner_min=d_inner_min)
                # print('updated gene choice:', gene_choice)
            
            if np.random.uniform() < self.mutation_prob:
                mutated_gene.append(random.choices(gene_choice[i])[0])
            else:
                mutated_gene.append(gene[i])

        return mutated_gene
  
    def get_scores(self, genes):
        configs = []
        for gene in genes:
            configs.append(self.converter.gene2config(gene))

        scores = []
        params = []
        latencies = []
        for i, config in enumerate(configs):
            model_config = copy.deepcopy(model_config_defaults)
            model_config.update(config)
            model = get_model(model_config)
            
            curr_n_all_param, _, _, params_attention, params_ff = process_parameters(model, verbose=False)
            params.append(params_attention + params_ff)
            
            latency = get_latency(model, model_config)
            latencies.append(latency)
            
            # TODO: normalize params and latency by the maximum allowed
            score = ((params_attention + params_ff)*1./self.max_n_params) - (latency*1./self.max_latency) * self.latency_scale
            scores.append(score)
            print('indvidual %d -> params: %d, latency: %.4f, score: %.4f' % (i, params_attention+params_ff, latency, score))

        return scores, params, latencies
  
    def satisfy_constraints(self, gene):
        config = self.converter.gene2config(gene)
        model_config = copy.deepcopy(model_config_defaults)
        model_config.update(config)
        model = get_model(model_config)
        
        _, _, _, params_attention, params_ff = process_parameters(model, verbose=False)
        
        satisfy = True
        if (params_attention + params_ff) < self.param_constraint:
            print('gene {} did not satisfy nparam threshold: {}<{}'.format(gene, params_attention + params_ff, self.param_constraint))
            satisfy = False

        return satisfy

    def random_sample(self, sample_num):
        popu = []
        gene_choice = self.gene_choice
        i = 0
        while i < sample_num:
            samp_gene = []
            for k in range(self.gene_len):
                if k==1:
                    d_inner_min = int(1.7 * samp_gene[-1])
                    gene_choice = self.converter.get_gene_choice(d_inner_min=d_inner_min)
                    # print('updated gene choice:', gene_choice)

                samp_gene.append(random.choices(gene_choice[k])[0])

            if self.satisfy_constraints(samp_gene):
                popu.append(samp_gene)
                i += 1

        return popu

    def profile(self):
        gene = [self.gene_choice[i][-1] for i in range(self.gene_len)]
        config = self.converter.gene2config(gene)
        print('biggest config:', config)
        model_config = copy.deepcopy(model_config_defaults)
        model_config.update(config) 
        biggest_model = get_model(model_config)

        _, _, _, params_attention, params_ff = process_parameters(biggest_model, verbose=False)
        self.max_latency = get_latency(biggest_model, model_config)
        self.max_n_params = params_attention + params_ff

        print('In this search-space -> maximum number of parameters: {}, maximum latency: {}'.format(self.max_n_params, self.max_latency))

        return

    def update_pareto_front(self):
        self.pareto['population'] = []
        self.pareto['params'] = []
        self.pareto['latencies'] = []

        for i in range(len(self.all_population)):
            this_params, this_latency = self.all_params[i], self.all_latencies[i]
            is_pareto = True
            for j in range(len(self.all_params)):
                params, latency = self.all_params[j], self.all_latencies[j]
                if (params > this_params) and (latency < this_latency):
                    is_pareto = False
                    break 
            if is_pareto:
                self.pareto['population'].append(self.all_population[i])
                self.pareto['params'].append(self.all_params[i])
                self.pareto['latencies'].append(self.all_latencies[i])
    
        return

    def plot_samples(self, iter):
        plt.figure()
        plt.scatter(self.all_params, self.all_latencies, s=10)
        plt.scatter(self.pareto['params'], self.pareto['latencies'], s=10)
        plt.ylabel('Latency')
        plt.xlabel('Decoder nParams')
        plt.title('Pareto Curve')
        plt.grid(axis='y')
        plt.savefig(os.path.join(self.results_path, 'pareto_latency_iter{}.png'.format(iter)), bbox_inches="tight")


def test_converter():
    config = {
        'd_model': 512,
        'n_layer': 5,
        'd_inner': [2048, 2049, 2050, 2051, 2052],
        'n_head': [4, 6, 7, 8, 9],
    }

    args = {'n_layer_choice':[5,6,7,8], 
            'd_model_choice':[128, 256, 512], 
            'd_inner_choice':list(range(512, 2049, 100)), 
            'n_head_choice':[2,4,8]
            }

    converter = Converter(**args)
    gene_get = converter.config2gene(config)
    print('generated gene:', gene_get)

    config_get = converter.gene2config(gene_get)
    print('gene -> config:', config_get)
    model_config = copy.deepcopy(model_config_defaults)
    model_config.update(config_get)
    model = get_model(model_config)
    print(model)

    print('gene choices:', converter.get_gene_choice())


def test_evo_search():
    args = {'results_path': '/home/t-mojanj/Projects/archai/evo_search','population_size':50, 'parent_size':10, 'mutation_size':20, 'mutation_prob':0.3, 'crossover_size':20, 'n_iter':30,
            'n_layer_choice':[3,4,5,6,7,8], 'd_model_choice':[128, 256, 512], 'd_inner_choice':list(range(512, 2049, 50))+[2048], 'n_head_choice':[2,4,8],
            'param_constraint':4e6, 'latency_scale':2.
            }
    alg = Evolution(**args)
    best_config = alg.run_evo_search()
    print(best_config)

if __name__=='__main__':
    # test_converter()
    test_evo_search()