import numpy as np
import model, utils


class GeneticSearch:
    def __init__(self, model, filepath, epochs, nb_parents, mutation_rate,
                 popsize, nb_genes=16000):
        self.model = model
        self.filepath = filepath
        self.wav = utils.wav(filepath)
        #self.mfcc = utils.wav2mfcc(filepath)
        self.og_label = model.predict(filepath)
        self.epochs = epochs
        self.nb_parents = nb_parents
        self.mutation_rate = mutation_rate
        self.init_rate = 0.005
        self.crossover_method = 'spc'
        self.popsize = popsize
        self.nb_genes = nb_genes
        self.population = self.init_population()

    # Set up an initial population of slightly mutated wav-arrays
    def init_population(self):
        #population = np.vstack([self.wav]*self.popsize)
        population = np.empty([self.popsize, self.nb_genes])
        for index in range(self.popsize):
            chromosome = self.mutate(self.wav, init=True)
            population[index] = chromosome
        return population

    def crossover(self, par1, par2):
        if self.crossover_method == 'spc':
            crossover_point = np.random.randint(0, self.genes)
            ch1 = par1[:crossover_point].append(par2[crossover_point:])
            ch2 = par2[:crossover_point].append(par1[crossover_point:])
        else:
            raise NotImplementedError
        return ch1, ch2

    # First Try: Mutate nb_genes*mutation_rate random genes to
    # values from a uniform distribution [-1.0, 1.0] -> 0.015 first
    def mutate(self, chromosome, init=False):
        if not init:
            genes = np.random.choice(self.nb_genes, int(self.nb_genes*self.mutation_rate), replace=False)
        else:
            genes = np.random.choice(self.nb_genes, int(self.nb_genes*self.init_rate), replace=False)
        for index in genes:
            chromosome[index] = np.random.uniform(-1.0, 1.0)
        return chromosome

    # Difference between initial logits and logits of current population
    def score_fitness(self):
        
        raise NotImplementedError

    def get_fittest(self):
        raise NotImplementedError
