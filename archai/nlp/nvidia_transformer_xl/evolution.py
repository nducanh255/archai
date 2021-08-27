import random
import os
import pickle
import copy
import imageio
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
                param_constraint=4e6, latency_scale=1., n_threads=1, latency_repeat=5):
        
        self.results_path = os.path.join(results_path, 'param_threshold_{}'.format(param_constraint))
        os.makedirs(self.results_path, exist_ok=True)
        
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

        self.n_threads = n_threads # number of threads for latency measurement
        self.latency_repeat = latency_repeat # number of runs for mean latency computation
        
        self.best_config = None
        self.pareto = {'population':[], 'params':[], 'latencies':[]}
        self.all_population = []
        self.all_scores = []
        self.all_params = []
        self.all_latencies = []

        self.counts = {}

    def run_evo_search(self, pareto_search=False):
        # if pareto_search is Flase, only searches in the vicinity of the maximum score seen
        population = self.random_sample(self.population_size)
        self.all_population = population
        
        self.update_counts(population)

        logs = {'population':[population], 'params':[], 'latencies':[], 'parents':[], 'parents_scores':[], 'best_config':[], 'pareto':[]}

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
            assert len(population_scores)==self.population_size
            print(f"| Iteration {i}, Max score: {max(population_scores)}")

            self.all_scores += population_scores_unseen
            self.all_params += population_params_unseen
            self.all_latencies += population_latencies_unseen
            print('all_population len:', 'all_scores len:', len(self.all_scores), len(self.all_population), 'all_params len:', len(self.all_params), 'all_latencies len:', len(self.all_latencies))
            self.update_pareto_front()
            
            if pareto_search:
                count_weights = self.get_count_weights()
                print(count_weights)

                # selected_ind = np.random.choice(len(self.pareto['population']), size=self.parent_size, replace=(len(self.pareto['population'])<self.parent_size))
                selected_ind = np.random.choice(len(self.pareto['population']), size=self.parent_size, p=count_weights)
                parents_population = [self.pareto['population'][m] for m in selected_ind]
                parents_score = [self.pareto['scores'][m] for m in selected_ind]
                parents_params = [self.pareto['params'][m] for m in selected_ind]
                parents_latencies = [self.pareto['latencies'][m] for m in selected_ind]

            else:
                sorted_ind = np.array(population_scores).argsort()[::-1][:self.parent_size]
                self.best_config = self.converter.gene2config(population[sorted_ind[0]])
                self.best_param = population_params[sorted_ind[0]]
                self.best_latency = population_latencies[sorted_ind[0]]
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
            self.update_counts(population)

            logs['population'].append(population)
            logs['params'].append(population_params)
            logs['latencies'].append(population_latencies)
            logs['parents'].append(parents_population)
            logs['parents_scores'].append(parents_score)
            logs['best_config'].append(self.best_config)
            logs['pareto'].append(self.pareto)

            self.all_population += mutate_population + crossover_population

            self.plot_samples(iter=i, parents={'params':parents_params, 'latencies':parents_latencies})

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
            
            latency = get_latency(model, model_config, n_threads=self.n_threads, repeat=self.latency_repeat)
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

    def semi_brute_force(self, nsamples, batch=1000):
        path_to_population = os.path.join(self.results_path, 'init_population_bruteforce.pkl')
        if os.path.exists(path_to_population):
            with open(path_to_population, 'rb') as f:
                population = pickle.load(f)
            population = population[:nsamples]
        else:
            population = self.random_sample(nsamples)
            with open(path_to_population, 'wb') as f:
                pickle.dump(population, f)

        population_scores = []
        for idx in range(0, nsamples, batch):
            curr_population = population[idx:idx+batch]
            curr_population_scores, curr_population_params, curr_population_latencies = self.get_scores(curr_population)
            population_scores += curr_population_scores

            self.all_population += curr_population
            self.all_params += curr_population_params
            self.all_latencies += curr_population_latencies
            self.update_pareto_front()

            sorted_ind = np.array(population_scores).argsort()[::-1]
            self.best_config = self.converter.gene2config(self.all_population[sorted_ind[0]])
            self.best_param = self.all_params[sorted_ind[0]]
            self.best_latency = self.all_latencies[sorted_ind[0]]
            print(f"| Config for highest score model: {self.best_config}")
            print(f"| nParams for highest score model: {self.best_param}")
            print(f"| Latency for highest score model: {self.best_latency}")
            self.plot_samples()
            
            logs = {'population':population, 'params':curr_population_params, 'latencies':curr_population_latencies, 'scores':curr_population_scores, 'pareto':self.pareto}

            path_to_pkl = os.path.join(self.results_path, 'logs_bruteforce_{}.pkl'.format(idx))
            with open(path_to_pkl, 'wb') as f:
                print(f"=> Saving indices {idx}-{idx+batch}")
                pickle.dump(logs, f)
        
        
        sorted_ind = np.array(population_scores).argsort()[::-1]
        self.best_config = self.converter.gene2config(population[sorted_ind[0]])
        self.best_param = self.all_params[sorted_ind[0]]
        self.best_latency = self.all_latencies[sorted_ind[0]]
        print(f"| Config for highest score model: {self.best_config}")
        print(f"| nParams for highest score model: {self.best_param}")
        print(f"| Latency for highest score model: {self.best_latency}")
        self.plot_samples()
        self.update_pareto_front()
        
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
        self.pareto['scores'] = []
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
                self.pareto['scores'].append(self.all_scores[i])
                self.pareto['params'].append(self.all_params[i])
                self.pareto['latencies'].append(self.all_latencies[i])
    
        return

    def update_counts(self, population):
        n_repeated = 0
        for gene in population:
            key = ','.join([str(g) for g in gene])
            if key in self.counts.keys():
                self.counts[key] += 1
                n_repeated += 1
            else:
                self.counts[key] = 1      
        # print(len(self.counts.keys()))
        # print(n_repeated)
    
    def get_count_weights(self):
        pareto_counts = []
        for gene in self.pareto['population']:
            key = ','.join([str(g) for g in gene])
            pareto_counts.append(self.counts[key])
        print(pareto_counts)

        # total_counts = np.sum(pareto_counts)
        # count_weights = [count/total_counts for count in pareto_counts]

        counts_max = max(pareto_counts)
        counts_min = min(pareto_counts)
        counts_range = counts_max if (counts_max==counts_min) else (counts_max-counts_min)
        #------- scale between [0,1] to avoid numerical issues
        scaled_counts = [(count - counts_min)/counts_range for count in pareto_counts]
        count_weights = [1.0/(scaled_count + 1) for scaled_count in scaled_counts]
        count_weights = np.asarray(count_weights)/np.sum(count_weights)

        return count_weights
    
    def plot_samples(self, iter=None, parents=None):
        plt.figure()
        plt.scatter(self.all_params, np.asarray(self.all_latencies) * 1000., s=10)
        plt.scatter(self.pareto['params'], np.asarray(self.pareto['latencies']) * 1000., s=10)
        if self.best_config:
            plt.scatter(self.best_param, self.best_latency * 1000., c='y', s=50, marker='*', edgecolors='k', alpha=0.3)
        if parents:
            plt.scatter(parents['params'], np.asarray(parents['latencies']) * 1000., s=5, color='tab:green')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Decoder nParams')
        plt.ylim((70, 300))
        plt.xlim((4.2e6, 2.6e7))
        plt.title('Pareto Curve')
        plt.grid(axis='y')

        fname = 'pareto_latency_iter{}.png'.format(iter) if iter is not None else 'pareto_latency_bruteforce.png'
        plt.savefig(os.path.join(self.results_path, fname), bbox_inches="tight")


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
            'param_constraint':5e6, 'latency_scale':2., 'n_threads':1, 'latency_repeat':5
            }
    alg = Evolution(**args)

    # alg.semi_brute_force(nsamples=20000)

    best_config = alg.run_evo_search(pareto_search=True)
    print(best_config)

    images = []
    results_path = os.path.join(args['results_path'], 'param_threshold_{}'.format(args['param_constraint']))
    for i in range(args['n_iter']):
        fname = os.path.join(results_path, 'pareto_latency_iter{}.png'.format(i))
        images.append(imageio.imread(fname))
    imageio.mimsave(os.path.join(results_path, 'search_animation.gif'), images)

if __name__=='__main__':
    # test_converter()
    test_evo_search()