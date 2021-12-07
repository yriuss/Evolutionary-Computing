import numpy as np
import random as r
import matplotlib.pyplot as plt
from numpy import random
import time
from GA import GA
from functions import Griewank
from ES import ES

class ModelGA():
    def __init__(self, objective_function) -> None:
        self.max_or_min = 0
        self.objective_function = objective_function
        self.recombination = "whole arithmetic recombination"
        self.mutation = "uniform mutation"
        self.offspring_selection = "fitness proportional selection"
        self.parent_selection = "fitness proportional selection"
        self.alpha = 0.5
        self.avg = 0
        self.std_dev = 0
        self.mutation_rate = 0.01
        self.counter = 0

    def termination_condition(self, fitness):
        self.counter += 1
        print("Generation: ", self.counter)
        if(self.counter> 200):
            return False
        else:
            return True

class ModelES():
    def __init__(self, objective_function) -> None:
        self.max_or_min = 0
        self.objective_function = objective_function
        self.mutation = "uniform mutation"
        self.offspring_selection = "mu,lambda"
        self.parent_selection = "test"
        self.sigma = 2
        self.lam = 300
        self.mu = 40
        self.k = 10
        self.counter = 0
        self.mutation_avg = 0

    def termination_condition(self, fitness):
        self.counter += 1
        print("Generation: ", self.counter)
        print("Best fitness: ",fitness[np.argmax(fitness)])

        if(self.counter> 4000):
            return False
        else:
            return True

def main():
    griewank = Griewank()

    print(griewank.compute(np.array([-0.0707, 1.565])))
    #model = ModelGA(griewank.compute)
    #pop = np.random.uniform(low=-32, high=32, size=(700,2))
    #
    #ga_alg = GA(pop, 32.768, -32.768)
    #
    #start = time.time()
    #ga_alg.run(model)
    #stop = time.time()

    #print("Tempo de processamento foi de ", stop - start)
    model = ModelES(griewank.compute)
    pop = np.random.uniform(low=-32, high=32, size=(model.mu,2))

    
    es_alg = ES(pop, model, 32.768, -32.768)

    start = time.time()
    es_alg.run(model)
    stop = time.time()

    print("Tempo de processamento foi de ", stop - start)

if __name__=="__main__":
    main()