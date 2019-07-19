import numpy as np
import model, utils
import librosa
import soundfile as sf
import os

class GeneticSearch:
    def __init__(self, model, filepath, epochs, nb_parents, mutation_rate,
                 popsize, nb_genes=16000):
        self.model = model
        self.filepath = filepath
        self.wav = utils.wav(filepath)
        self.og_label = model.predict(filepath)
        self.og_label_index = model.predict(filepath, index=True)
        self.epochs = epochs
        self.nb_parents = nb_parents
        self.mutation_rate = mutation_rate
        self.init_rate = 0.0005
        self.crossover_method = 'spc'
        self.popsize = popsize
        self.nb_genes = nb_genes
        self.population = None

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
            crossover_point = np.random.randint(0, self.nb_genes-1)
            ch1 = np.concatenate((par1[:crossover_point], par2[crossover_point:]))
            ch2 = np.concatenate((par2[:crossover_point], par1[crossover_point:]))
        else:
            raise NotImplementedError
        return ch1, ch2

    # Randomly selects pairs of the fittest individuals and generates the next population
    def mate_pool(self, pool):
        offspring = list()
        parents = [utils.random_pairs(pool) for i in range(len(pool))]
        # offspring = [self.crossover(*pair) for pair in parents]
        for pair in parents:
            offspring.extend(self.crossover(*pair))
        offspring = np.array(offspring)
        return np.array(offspring)

    # First Try: Mutate nb_genes*mutation_rate random genes to
    # values from a uniform distribution [-1.0, 1.0] -> 0.015 first
    def mutate(self, chromosome, init=False):
        if not init:
            genes = np.random.choice(self.nb_genes, int(self.nb_genes*self.mutation_rate), replace=False)
        else:
            genes = np.random.choice(self.nb_genes, int(self.nb_genes*self.init_rate), replace=False)
        for index in genes:
            chromosome[index] = chromosome[index]+np.random.uniform(-0.001, 0.001)
            # chromosome[index] = np.random.uniform(-1.0, 1.0)
        return chromosome

    # Sort by lowest confidence score of initial label
    def fit_sort(self):
        scores = np.empty(self.popsize)
        for index, elem in enumerate(self.population):
            scores[index] = self.model.get_confidence_scores(elem)[self.og_label_index]
        print(scores)
        print(scores.argsort())
        print(np.flip(scores.argsort()))
        sorted_pop = self.population[scores.argsort()]
        sorted_pop = np.flip(sorted_pop)
        return sorted_pop

    def get_fittest(self):
        #print(np.sum(self.fit_sort()[:self.nb_parents]))
        return self.fit_sort()[:self.nb_parents]

    # Tries to decrease confidence in og label
    # by applying crossover between the fittest members and mutating the offspring
    def search(self, out_dir, verbose=0):
        self.population = self.init_population()
        fittest = self.get_fittest()
        for epoch in range(self.epochs):
            offspring = self.mate_pool(fittest)
            self.population = np.array([self.mutate(chromosome) for chromosome in offspring])
            fittest = self.get_fittest()
            if verbose == 1:
                winner = fittest[0]
                label = self.model.predict_array(winner)
                print('Pertubated Prediction is {}, index {}'.format(label,
                                                                     self.model.predict_array(winner, index=True)))
                utils.save_array_to_wav(out_dir, 'epoch{}_{}.wav'.format(epoch, label), winner, 16000)
        winner = self.get_fittest()[0]
        print('Filepath: {}'.format(self.filepath))
        print('Initial Prediction was {}, index {}.'.format(self.og_label, self.og_label_index))
        print('Pertubated Prediction is {}, index {}'.format(self.model.predict_array(winner),
                                                             self.model.predict_array(winner, index=True)))
        utils.save_array_to_wav(out_dir, 'initial.wav', self.filepath, 16000)
        utils.save_array_to_wav(out_dir, 'final.wav', winner, 16000)

        # sf.write(os.path.join(out_dir, 'unpertubated1.wav'), utils.wav(self.filepath), 16000)
        # sf.write(os.path.join(out_dir, 'winner1.wav'), winner, 16000)