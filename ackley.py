import numpy as np
import random as r
import matplotlib.pyplot as plt
from numpy import random
import time
from GA import GA


class Model():
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
        print("Best fitness: ",fitness[np.argmax(fitness)])

        if(self.counter> 200):
            return False
        else:
            return True

def ackley(x):
    x1 = x[0]
    x2 = x[1]

    a = 20
    b = 0.2
    c = 2*np.pi
    
    sum1 = x1**2 + x2**2 
    sum2 = np.cos(c*x1) + np.cos(c*x2)
    term1 = - a * np.exp(-b * ((1/2.) * sum1**(0.5)))
    term2 = - np.exp((1/2.)*sum2)
    return term1 + term2 + a + np.exp(1)

def main():
    model = Model(ackley)
    pop = np.random.uniform(low=-32, high=32, size=(700,2))
    ga_alg = GA(pop, 32.768, -32.768)

    start = time.time()
    ga_alg.run(model)
    stop = time.time()

    print("Tempo de processamento foi de ", stop - start)

if __name__=="__main__":
    main()