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

    # pick "default" fixed values as the middle of each tested set
    pops = sorted(set(params[:, 0]))
    cxs = sorted(set(params[:, 1]))
    muts = sorted(set(params[:, 2]))

    default_pop = pops[len(pops) // 2]
    default_cx = cxs[len(cxs) // 2]
    default_mut = muts[len(muts) // 2]

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
    def plot_comparison(filter_func, title, filename, legend_title, var_name):
        plt.figure()

        items = []
        for i, (p, c, m) in enumerate(params):
            if filter_func(p, c, m):
                if var_name == "Population":
                    val = int(p)
                    label = f"{val}"
                elif var_name == "Crossover":
                    val = float(c)
                    label = f"{val:.2f}"
                else:  # Mutation
                    val = float(m)
                    label = f"{val:.4f}"

                items.append((val, i, label))

        items.sort(key=lambda x: x[0])

        for val, i, label in items:
            plt.plot(gens, means[i], label=label)
            plt.fill_between(gens, ci_lo[i], ci_hi[i], alpha=0.25)

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(title)
        plt.legend(title=legend_title)

        plt.savefig(os.path.join(out_dir, filename),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # population comparison plot
    plot_comparison(
        lambda p, c, m: c == default_cx and m == default_mut,
        "Mean Best Fitness by Generation for Varying Population",
        "population_comparison.pdf",
        "Population",
        "Population"
    )

    # crossover comparison plot
    plot_comparison(
        lambda p, c, m: p == default_pop and m == default_mut,
        "Mean Best Fitness by Generation for Varying Crossover",
        "crossover_comparison.pdf",
        "Crossover",
        "Crossover"
    )

    # mutation comparison plot
    plot_comparison(
        lambda p, c, m: p == default_pop and c == default_cx,
        "Mean Best Fitness by Generation for Varying Mutation",
        "mutation_comparison.pdf",
        "Mutation",
        "Mutation"
    )


if __name__ == "__main__":
    make_graphs("./statistics/grid_search_stats.npz")