import numpy as np


class Population:
    def __init__(self, model, audio, epochs, nb_parents, mutation_rate,
                 crossover_method,  popsize, genes):
        self.model = model
        self.sample = audio
        self.epochs = epochs
        self.nb_parents = nb_parents
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        self.popsize = popsize
        self. genes = genes
        self.population = self.init_population()

    def init_population(self):
        population = np.empty(shape = self.nb_parents)
        for index, elem in range(self.popsize):
            population[index] = self.sample
            # change sample a bit
        return population

    def crossover(self, par1, par2):
        if self.crossover_method == 'spc':
            crossover_point = np.random.randint(0, self.genes)
            ch1 = par1[:crossover_point].append(par2[crossover_point:])
            ch2 = par2[:crossover_point].append(par1[crossover_point:])
        else:
            raise NotImplementedError
        return ch1, ch2

    def mutate(self, child):
        for gene in child:
            if np.random.random() < self.mutation_rate:
                # mutate
                pass
            else:
                pass
        return child

    def score_fitness(self):
        raise NotImplementedError

    def get_fittest(self):
        raise NotImplementedError
