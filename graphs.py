import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def make_graphs(npz_path, out_dir="./graphs"):
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path)

    gens = data["generations"]
    per_run = data["best_fitness_per_run"]
    best_across = data["best_fitness_across_runs"]
    mean = data["mean_best_fitness"]
    std = data["std_best_fitness"]
    ci_lo = data["ci95_low"]
    ci_hi = data["ci95_high"]

    # mean + 95% CI
    plt.figure()
    plt.plot(gens, mean)
    plt.fill_between(gens, ci_lo, ci_hi, alpha=0.25)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Mean Best Fitness with 95% CI")
    plt.savefig(os.path.join(out_dir, "mean_best_fitness_ci95.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # best across runs
    plt.figure()
    plt.plot(gens, best_across)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best Fitness Across Runs")
    plt.savefig(os.path.join(out_dir, "best_across_runs.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # std per generation
    plt.figure()
    plt.plot(gens, std)
    plt.xlabel("Generation")
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation of Best Fitness")
    plt.savefig(os.path.join(out_dir, "std_best_fitness.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # spaghetti plot for fun (probably not useful)
    plt.figure()
    for i in range(per_run.shape[0]):
        plt.plot(gens, per_run[i], alpha=0.15)
    plt.plot(gens, mean)
    plt.fill_between(gens, ci_lo, ci_hi, alpha=0.25)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best Fitness of All Runs")
    plt.savefig(os.path.join(out_dir, "all_runs.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    path = "./statistics/grid_search_stats.npz"
    make_graphs(path)