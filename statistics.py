import os
import numpy as np


class StatisticsRecorder:
    def __init__(self, out_dir="./statistics"):
        self.out_dir = out_dir
        self.runs = []
        self.params = []

    # store best fitness for one run
    def add_run(self, best_per_gen, params=None):
        self.runs.append(np.asarray(best_per_gen, dtype=float))
        self.params.append(params if params is not None else {})

    # compute aggregate stats across runs
    def compute(self):
        # shape: (n_runs, n_gens)
        data = np.vstack(self.runs)
        gens = np.arange(data.shape[1])

        best_across = np.min(data, axis=0)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mean)

        # 95% CI
        if data.shape[0] > 1:
            sem = std / np.sqrt(data.shape[0])
            ci_half = 1.96 * sem
        else:
            ci_half = np.zeros_like(mean)

        ci_low = mean - ci_half
        ci_high = mean + ci_half

        return {
            "generations": gens,
            "best_fitness_per_run": data,
            "best_fitness_across_runs": best_across,
            "mean_best_fitness": mean,
            "std_best_fitness": std,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }

    # save as npz file
    def save_npz(self, filename):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, filename)
        stats = self.compute()

        # also store pop/cx/mut if provided
        pops = [p.get("pop", np.nan) for p in self.params]
        cxs = [p.get("cx", np.nan) for p in self.params]
        muts = [p.get("mut", np.nan) for p in self.params]

        stats["run_pop"] = np.asarray(pops, dtype=float)
        stats["run_cx"] = np.asarray(cxs, dtype=float)
        stats["run_mut"] = np.asarray(muts, dtype=float)

        np.savez_compressed(path, **stats)
        return path