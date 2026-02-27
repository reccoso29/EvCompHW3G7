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
    # parameters for a much larger grid search
    # use ranges to test many combinations
    pop_sizes = range(100, 1001, 50)
    cx_probs = np.arange(0.5, 1.0, 0.05)
    mut_probs = np.arange(0.05, 0.5, 0.02)
    
    print("Starting grid search for the best params---")
    
    # store results from each run
    results = []
    
    # calculate total runs for a progress indicator
    total_runs = len(pop_sizes) * len(cx_probs) * len(mut_probs)
    run_count = 0
    
    for p in pop_sizes:
        for c in cx_probs:
            for m in mut_probs:
                run_count += 1
                print(f"  [{run_count}/{total_runs}] Testing: pop={p}, cx={c:.2f}, mut={m:.2f}")
                best = run_experiment(pop_size=p, cxpb=c, mutpb=m, ngen=300)
                results.append({"pop": p, "cx": c, "mut": m, "best": best})

    # sort results by best distance (lower is better) and print top 5
    results.sort(key=lambda r: r["best"])
    print("\n--- Top 5 Experiment Results! ---")
    for i, res in enumerate(results[:5]):
        print(f"#{i+1}: pop={res['pop']}, cx={res['cx']:.2f}, mut={res['mut']:.2f} -> best distance: {res['best']:.2f}")

if __name__ == "__main__":
    main()