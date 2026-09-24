import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genetic_algorithm import run_ga
from utils import load_split, evaluate_baseline, evaluate_selected_features

X_train, X_test, y_train, y_test, feature_names = load_split()

baseline = evaluate_baseline(X_train, X_test, y_train, y_test)

print(f"Baseline CV accuracy on train (30 features): {baseline['cv_accuracy']:.4f}")
print(f"Baseline test accuracy (30 features):        {baseline['test_accuracy']:.4f}\n")

best_individual, best_fitness_score, history = run_ga(
    X_train, y_train,
    pop_size=20, generations=30,
    crossover_rate=0.8, mutation_rate=0.02,
    elitism=1, seed=42, verbose=True,
)

selected_idx = np.where(best_individual == 1)[0]
selected_names = [feature_names[i] for i in selected_idx]

ga_test_acc = evaluate_selected_features(X_train, X_test, y_train, y_test, selected_idx)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Baseline test accuracy (30 features): {baseline['test_accuracy']:.4f}")
print(f"GA test accuracy ({len(selected_idx)} features):       {ga_test_acc:.4f}")
print(f"GA best fitness during search: {best_fitness_score:.4f}")
print(f"\nSelected features ({len(selected_idx)}/30):")
for name in selected_names:
    print(f"  - {name}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(history) + 1), history, marker="o", markersize=3,
         label="GA best fitness (CV accuracy - feature penalty)")
plt.axhline(y=baseline["search_fitness"], color="red", linestyle="--",
            label=f"Baseline, all 30 features = {baseline['search_fitness']:.4f}")
plt.xlabel("Generation")
plt.ylabel("Fitness (CV accuracy - feature penalty)")
plt.title("GA Convergence - Feature Selection on Breast Cancer Wisconsin")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("convergence.png", dpi=150)
print("\nsaved convergence.png")

with open("summary.txt", "w") as f:
    f.write(f"Baseline CV accuracy on train (30 features): {baseline['cv_accuracy']:.4f}\n")
    f.write(f"Baseline held-out test accuracy (30 features): {baseline['test_accuracy']:.4f}\n")
    f.write(f"GA held-out test accuracy ({len(selected_idx)} features): {ga_test_acc:.4f}\n")
    f.write(f"Selected features: {', '.join(selected_names)}\n")
