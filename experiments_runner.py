import random
import numpy as np
from deap import base, creator, tools, algorithms

from tsp_io import load_cities
from distance import build_distance_matrix
from ga import tour_length

from statistics import StatisticsRecorder

# setting up fitness params
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def run_experiment(pop_size, cxpb, mutpb, ngen=200):
    # grab the cities and distances
    cities = load_cities("tsp.dat")
    dist = build_distance_matrix(cities)
    n = len(cities)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n), n)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return (tour_length(individual, dist),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOrdered)

    # per-allele mutation
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutpb)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cxpb, mutpb=1.0, ngen=ngen,
        halloffame=hof, stats=stats, verbose=False
    )

    best_per_generation = [entry["min"] for entry in logbook]
    return hof[0].fitness.values[0], best_per_generation


def main():
    # variables for experiment
    pop_sizes = [200, 500, 1000]
    cx_probs = [0.65, 0.70, 0.75, 0.80]
    mut_probs = [0.0010, 0.0020, 0.0050, 0.0100]

    repeats = 50
    base_seed = 1

    print("Starting grid search for the best params---")

    recorder = StatisticsRecorder("./statistics")
    results = []

    total_runs = len(pop_sizes) * len(cx_probs) * len(mut_probs) * repeats
    run_count = 0

    for p in pop_sizes:
        for c in cx_probs:
            for m in mut_probs:
                for r in range(repeats):
                    run_count += 1
                    
                    # per-run seed
                    seed = base_seed + p*100000 + int(c*1000)*100 + int(m*1000)*10 + r
                    random.seed(seed)
                    np.random.seed(seed)

                    print(f"[{run_count}/{total_runs}] pop={p}, cx={c}, mut={m}, run={r+1}")

                    best, best_per_gen = run_experiment(
                        pop_size=p,
                        cxpb=c,
                        mutpb=m,
                        ngen=300
                    )

                    recorder.add_run(best_per_gen, {
                        "pop": p,
                        "cx": c,
                        "mut": m
                    })

                    results.append({
                        "pop": p,
                        "cx": c,
                        "mut": m,
                        "run": r+1,
                        "best": best,
                        "seed": seed
                    })

    recorder.save_npz("grid_search_stats.npz")

    # prints full descending leaderboard
    results.sort(key=lambda r: r["best"])

    print("\n--- All Runs Leaderboard ---")
    for i, res in enumerate(results):
        print(
            f"#{i+1}: pop={res['pop']}, cx={res['cx']:.2f}, "
            f"mut={res['mut']:.4f}, run={res['run']} "
            f"-> best distance: {res['best']:.2f}, seed={res['seed']}"
        )


if __name__ == "__main__":
    main()