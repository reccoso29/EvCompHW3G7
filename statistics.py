import os
import numpy as np


class StatisticsRecorder:
    def __init__(self, out_dir="./statistics"):
        self.out_dir = out_dir
        self.runs = []
        self.params = []

    # stores best fitness for one run
    def add_run(self, best_per_gen, params=None):
        self.runs.append(np.asarray(best_per_gen, dtype=float))
        self.params.append(params)

    # computes stats across runs
    def compute(self):
        data = np.vstack(self.runs)

        pops = np.array([p["pop"] for p in self.params])
        cxs  = np.array([p["cx"]  for p in self.params])
        muts = np.array([p["mut"] for p in self.params])

        unique_combos = list(set(zip(pops, cxs, muts)))

        combo_means = []
        combo_stds  = []
        combo_ci_lo = []
        combo_ci_hi = []
        combo_params = []

        for (p, c, m) in unique_combos:
            mask = (pops == p) & (cxs == c) & (muts == m)
            combo_data = data[mask]

            mean = combo_data.mean(axis=0)
            std  = combo_data.std(axis=0, ddof=1)

            sem = std / np.sqrt(combo_data.shape[0])
            ci_half = 1.96 * sem

            combo_means.append(mean)
            combo_stds.append(std)
            combo_ci_lo.append(mean - ci_half)
            combo_ci_hi.append(mean + ci_half)
            combo_params.append((p, c, m))

        return {
            "generations": np.arange(data.shape[1]),
            "combo_means": np.array(combo_means),
            "combo_stds": np.array(combo_stds),
            "combo_ci_lo": np.array(combo_ci_lo),
            "combo_ci_hi": np.array(combo_ci_hi),
            "combo_params": np.array(combo_params)
        }

    # saves as an npz file
    def save_npz(self, filename):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, filename)
        stats = self.compute()
        np.savez_compressed(path, **stats)
        return path