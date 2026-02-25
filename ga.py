import random
import numpy as np
from deap import base, creator, tools, algorithms

from tsp_io import load_cities
from distance import build_distance_matrix


#adds up the total miles of a tour
def tour_length(individual, dist):
    total = 0.0

    for i in range(len(individual) - 1):
        total += dist[individual[i]][individual[i + 1]]

    total += dist[individual[-1]][individual[0]]

    return total


def main():

    random.seed(42)
    np.random.seed(42)

    #load the cities from tsp.dat
    cities = load_cities("tsp.dat")

    #precompute all pairwise distances
    dist = build_distance_matrix(cities)

    n = len(cities)

    #minimize total distance
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #fitness function
    def evaluate(individual):
        return (tour_length(individual, dist),)

    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.9,
        mutpb=0.2,
        ngen=800,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best = hof[0]

    print("\nBest distance:", best.fitness.values[0])
    print("\nTour:")

    for idx in best:
        print(cities[idx][0])

    print("(back to start)")


if __name__ == "__main__":
    main()