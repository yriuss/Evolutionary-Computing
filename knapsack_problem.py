import numpy as np
import random as r
import matplotlib.pyplot as plt
from numpy import random
import time
from GA import GA

VM1 = 13
VM2 = 9
VM3 = 7

OBJECTS_VOLUME = np.array([3,2,1,2.2,1.4,3.8,0.2,0.1,0.13, 2.8, 1.5, 2, 3.1, 1.2, 1.7, 1.1, 0.3])
OBJECTS_VALUE = np.array([3, 2, 1, 2, 1, 4, 1, 1, 1, 3, 2, 2, 3, 1, 3, 2, 1])


class Model():
    def __init__(self, objective_function) -> None:
        self.max_or_min = 1
        self.objective_function = objective_function
        self.recombination = "one-point-crossover"
        self.mutation = "bit-flipping"
        self.offspring_selection = "fitness proportional selection"
        self.parent_selection = "fitness proportional selection"
        self.mutation_rate = 0.01
        self.counter = 0

    def termination_condition(self, fitness):
        self.counter += 1
        print("Generation: ", self.counter)
        print("Best fitness: ",fitness[np.argmax(fitness)])

        if(self.counter> 50):
            return False
        else:
            return True

def knapsack_problem(x):
    while(sum(x[0:17]*OBJECTS_VOLUME) > VM1+VM2+VM3):
        x[np.random.randint(low=0,high=17)] = 0
    
    return sum(x[0:17]*OBJECTS_VALUE)

def main():
    model = Model(knapsack_problem)
    pop = np.random.randint(low=0,high=2, size=(50,17))
    ga_alg = GA(pop)

    start = time.time()
    ga_alg.run(model)
    stop = time.time()

    print("Tempo de processamento foi de ", stop - start)

if __name__=="__main__":
    main()


#q = sum(vi*gi)