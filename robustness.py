import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genetic_algorithm import run_ga
from utils import load_split, evaluate_baseline, evaluate_selected_features

SEEDS = [1, 7, 21, 55, 99]

X_train, X_test, y_train, y_test, feature_names = load_split()

baseline = evaluate_baseline(X_train, X_test, y_train, y_test)
print(f"Baseline test accuracy: {baseline['test_accuracy']:.4f}\n")

results = []
histories = []

for seed in SEEDS:
    best_ind, best_fit, history = run_ga(
        X_train, y_train,
        pop_size=20, generations=30,
        crossover_rate=0.8, mutation_rate=0.02,
        elitism=1, seed=seed, verbose=False,
    )
    selected_idx = np.where(best_ind == 1)[0]
    test_acc = evaluate_selected_features(X_train, X_test, y_train, y_test, selected_idx)

    results.append({"seed": seed, "n_features": len(selected_idx), "test_acc": test_acc})
    histories.append(history)
    print(f"seed={seed:3d} | features={len(selected_idx):2d} | test accuracy={test_acc:.4f}")

accs = np.array([r["test_acc"] for r in results])
n_feats = np.array([r["n_features"] for r in results])
n_beat_baseline = int((accs >= baseline["test_accuracy"]).sum())

print("\n" + "=" * 60)
print("SUMMARY ACROSS SEEDS")
print("=" * 60)
print(f"Baseline (30 features): {baseline['test_accuracy']:.4f}")
print(f"GA accuracy: mean={accs.mean():.4f}  std={accs.std():.4f}  "
      f"min={accs.min():.4f}  max={accs.max():.4f}")
print(f"GA features used: mean={n_feats.mean():.1f}  std={n_feats.std():.1f}  "
      f"min={n_feats.min()}  max={n_feats.max()}")
print(f"Seeds where GA >= baseline: {n_beat_baseline}/{len(SEEDS)}")

plt.figure(figsize=(8, 5))
for seed, history in zip(SEEDS, histories):
    plt.plot(range(1, len(history) + 1), history, alpha=0.7, label=f"seed {seed}")
plt.axhline(y=baseline["search_fitness"], color="red", linestyle="--", linewidth=2,
            label=f"Baseline, all 30 features = {baseline['search_fitness']:.4f}")
plt.xlabel("Generation")
plt.ylabel("Best fitness (CV accuracy - feature penalty)")
plt.title("GA Convergence Across 5 Seeds")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("robustness_convergence.png", dpi=150)
print("\nsaved robustness_convergence.png")

with open("robustness_summary.txt", "w") as f:
    f.write("Multi-seed robustness results\n")
    f.write("=" * 40 + "\n")
    f.write(f"Baseline held-out test accuracy (30 features): {baseline['test_accuracy']:.4f}\n\n")
    for r in results:
        f.write(f"seed={r['seed']}: features={r['n_features']}, test_acc={r['test_acc']:.4f}\n")
    f.write(f"\nMean test accuracy: {accs.mean():.4f} (std {accs.std():.4f})\n")
    f.write(f"Mean features used: {n_feats.mean():.1f} (std {n_feats.std():.1f})\n")
    f.write(f"Seeds where GA >= baseline: {n_beat_baseline}/{len(SEEDS)}\n")
print("saved robustness_summary.txt")
