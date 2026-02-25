import random
import numpy as np
from deap import base, creator, tools, algorithms

from tsp_io import load_cities
from distance import build_distance_matrix
from ga import tour_length

# setting up fitness params
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def run_experiment(pop_size, cxpb, mutpb, ngen=200):
    # grab the cities and distances
    cities = load_cities("tsp.dat")
    dist = build_distance_matrix(cities)

    n = len(cities)

    # toolbox setup
    toolbox = base.Toolbox()

    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # eval function
    # add up all the miles using ga.py
    def evaluate(individual):
        return (tour_length(individual, dist),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    # Running the actual ea!
    algorithms.eaSimple(
        pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
        halloffame=hof, verbose=False
    )

    # take best dist
    return hof[0].fitness.values[0]

def main():
    # test out other parameters
    pop_sizes = [100, 300]
    cx_probs = [0.7, 0.9]
    mut_probs = [0.1, 0.2]
    
    print("running experiments!")
    
    for p in pop_sizes:
        for c in cx_probs:
            for m in mut_probs:
                best = run_experiment(pop_size=p, cxpb=c, mutpb=m, ngen=300)
                print(f"--- pop: {p} | cx: {c} | mut: {m} ---\n\tbest distance: {best:.2f}")

if __name__ == "__main__":
    main()