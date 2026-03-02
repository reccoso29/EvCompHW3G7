import os
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

from tsp_io import load_cities
from distance import build_distance_matrix
from ga import tour_length


# fixed best parameters
POP = 1000
CX = 0.70
MUT = 0.0050
SEED = 100070095
NGEN = 300


def run_best():
    random.seed(SEED)
    np.random.seed(SEED)

    cities = load_cities("tsp.dat")
    dist = build_distance_matrix(cities)
    n = len(cities)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return (tour_length(individual, dist),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUT)

    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(
        pop, toolbox,
        cxpb=CX,
        mutpb=1.0,
        ngen=NGEN,
        halloffame=hof,
        verbose=False
    )

    return hof[0], cities


def plot_tour(individual, cities):
    os.makedirs("./graphs", exist_ok=True)

    lats = [cities[i][1] for i in individual]
    lons = [cities[i][2] for i in individual]

    lats.append(lats[0])
    lons.append(lons[0])

    plt.figure()
    plt.plot(lons, lats)
    plt.scatter(lons, lats, s=10)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Map of Best Tour")

    plt.savefig("./graphs/best_tour_map.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    best_individual, cities = run_best()
    plot_tour(best_individual, cities)