import numpy as np
import random as r
import matplotlib.pyplot as plt
from numpy import random
import time
from GA import GA
from functions import Colville, Griewank
from ES import ES

from GA import GA
from GA import Model as ModelGA

from functions import Trid

from ES import ES
from ES import Model as ModelES

UB = 10
LB = -10






def main():

    #ABAIXO É PARA O GA
    pop = np.random.uniform(low=LB, high=UB, size=(700,4))
    
    model = ModelGA(
    pop,
    Colville().compute, 
    "whole arithmetic recombination", 
    "uniform mutation",
    "fitness proportional selection",
    "fitness proportional selection",
    0.03,
    1000,
    UB,
    LB,
    0)
    
    ga_alg = GA(model)
    start = time.time()
    ga_alg.run(model)
    stop = time.time()
    ga_alg.fitness
    print("Tempo de processamento foi de ", stop - start)

    #ABAIXO É PARA O ES
    
    #pop_size = 40
    #pop = np.random.uniform(low=-32, high=32, size=(pop_size,4))
    #model = ModelES(
    #pop,
    #Colville().compute,
    #"uniform mutation",
    #"mu,lambda",
    #"random",
    #2,
    #300,
    #40,
    #10,
    #1500,
    #UB,
    #LB,
    #0)
    #
    #
    #es_alg = ES(model)
    #
    #start = time.time()
    #es_alg.run(model)
    #stop = time.time()
    #
    #print("Tempo de processamento foi de ", stop - start)

if __name__=="__main__":
    main()