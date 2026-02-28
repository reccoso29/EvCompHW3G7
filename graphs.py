import os
import numpy as np
import matplotlib.pyplot as plt


def make_graphs(npz_path, out_dir="./graphs"):
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path)

    gens = data["generations"]
    means = data["combo_means"]
    ci_lo = data["combo_ci_lo"]
    ci_hi = data["combo_ci_hi"]
    params = data["combo_params"]

    # best fitness across all runs
    best_overall = np.min(means[:, -1])
    best_idx = np.argmin(means[:, -1])
    best_curve = means[best_idx]

    plt.figure()
    plt.plot(gens, best_curve)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Best Fitness Achieved by Generation")
    plt.savefig(os.path.join(out_dir, "best_fitness_overall.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # comparison plot helper
    def plot_comparison(filter_func, title, filename):
        plt.figure()

        for i, (p, c, m) in enumerate(params):
            if filter_func(p, c, m):
                label = f"{p if title=='Population' else (c if title=='Crossover' else m)}"
                plt.plot(gens, means[i], label=label)
                plt.fill_between(gens, ci_lo[i], ci_hi[i], alpha=0.25)

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(title)
        plt.legend(title=title)

        plt.savefig(os.path.join(out_dir, filename),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # population comparison plot
    plot_comparison(
        lambda p, c, m: c == 0.80 and m == 0.10,
        "Mean Best Fitness by Generation for Varying Population",
        "population_comparison.pdf"
    )

    # crossover comparison plot
    plot_comparison(
        lambda p, c, m: p == 500 and m == 0.10,
        "Mean Best Fitness by Generation for Varying Crossover",
        "crossover_comparison.pdf"
    )

    # mutation comparison plot
    plot_comparison(
        lambda p, c, m: p == 500 and c == 0.80,
        "Mean Best Fitness by Generation for Varying Mutation",
        "mutation_comparison.pdf"
    )


if __name__ == "__main__":
    make_graphs("./statistics/grid_search_stats.npz")