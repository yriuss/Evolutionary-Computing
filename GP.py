import numpy as np
import random as r


class GP():
    def __init__(self, model) -> None:
        self.pop = model.pop
        self.fitness = np.zeros(model.pop.size)
        self.best_fitness = 0
        self.function_set = model.function_set
        

    def evaluate(self, model, is_offspring=False):
        evaluations = np.zeros(self.pop_size)
        if(is_offspring):
            for i in range(self.pop_size):
                evaluations[i] = model.objective_function(self.population[i])
            
        else:
            for i in range(self.pop_size):
                evaluations[i] = model.objective_function(self.population[i])
            self.best_fitness = model.objective_function(self.population[np.argmax(self.fitness)])
        
        

        if(model.max_or_min):
            if((evaluations<0).sum() > 0):
                evaluations = evaluations - np.amin(evaluations)
        else:
            evaluations = np.abs(evaluations - np.amax(evaluations))
        
        self.fitness = evaluations

    def return_best(self, model):
            print("The best gene is ", self.population[np.argmax(self.fitness)])

            print("Alternative gene ", self.population[1])
            print("Fitness of best ", model.objective_function(self.population[np.argmax(self.fitness)]))
            print("Fitness of alternative ", model.objective_function(self.population[2]))
    def run(self, model):
        condition = True

        while(condition):
            self.evaluate(model)
            self.select_parents(model)
            #print(self.fitness)
            self.recombine(model)
            self.mutate(model)
            self.evaluate(model, True)
            condition = model.termination_condition(self.best_fitness)
            #print(self.fitness)
            #self.select_offspring(model)
        self.return_best(model)



class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None
        self.depth = 0


class Model():
    def __init__(self, pop, objective_function, recombination, mutation, offspring_selection, parent_selection, mutation_rate, n_of_generations, UB, LB, max_or_min) -> None:
        self.max_or_min = max_or_min
        self.pop = pop
        self.UB = UB
        self.LB = LB
        self.objective_function = objective_function
        self.recombination = recombination
        self.mutation = mutation
        self.offspring_selection = offspring_selection
        self.parent_selection = parent_selection
        self.mutation_rate = mutation_rate
        self.counter = 0
        self.n_of_generations = n_of_generations
        self.fitness = []
        self.function_set = np.array(['*', '+', '-', '/'])
        self.maximum_depth = 0
        self.maximum_width = 0

    def termination_condition(self, fitness):
        self.counter += 1
        print("Generation: ", self.counter)
        print("Best fitness: ",fitness)
        self.fitness.append(fitness)
        if(self.counter> self.n_of_generations):
            return False
        else:
            return True




 #Function set {+, −, ·, /}
 #Terminal set IR ∪ {x, y}

#Tree-based mutation has two parameters:
#• the probability of choosing mutation at the junction with recombination
#• the probability of choosing an internal point within the parent as the root
#of the subtree to be replaced


#subtree crossover

#Tree-based recombination has two parameters:
#• the probability of choosing recombination at the junction with mutation
#• the probability of choosing internal nodes as crossover points