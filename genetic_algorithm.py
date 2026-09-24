import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

FEATURE_PENALTY = 0.0015
SEARCH_N_ESTIMATORS = 50
CV_FOLDS = 3


def create_individual(n_features, rng):
    individual = rng.integers(0, 2, size=n_features)
    if individual.sum() == 0:
        individual[rng.integers(0, n_features)] = 1
    return individual


def create_population(pop_size, n_features, rng):
    return [create_individual(n_features, rng) for _ in range(pop_size)]


def fitness(individual, X_train, y_train, cache=None):
    key = tuple(individual.tolist()) if cache is not None else None
    if cache is not None and key in cache:
        return cache[key]

    selected = np.where(individual == 1)[0]
    if len(selected) == 0:
        return 0.0

    X_sel = X_train[:, selected]
    clf = RandomForestClassifier(n_estimators=SEARCH_N_ESTIMATORS, random_state=42, n_jobs=-1)
    cv_acc = cross_val_score(clf, X_sel, y_train, cv=CV_FOLDS, n_jobs=1).mean()
    score = cv_acc - FEATURE_PENALTY * len(selected)

    if cache is not None:
        cache[key] = score
    return score


def tournament_selection(population, fitnesses, rng, k=3):
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


def crossover(parent1, parent2, rng, crossover_rate=0.8):
    child1, child2 = parent1.copy(), parent2.copy()
    if rng.random() < crossover_rate:
        mask = rng.integers(0, 2, size=len(parent1)).astype(bool)
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
    return child1, child2


def mutate(individual, rng, mutation_rate=0.02):
    for i in range(len(individual)):
        if rng.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    if individual.sum() == 0:
        individual[rng.integers(0, len(individual))] = 1
    return individual


def run_ga(X_train, y_train, pop_size=20, generations=30,
           crossover_rate=0.8, mutation_rate=0.02,
           elitism=1, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    cache = {}

    population = create_population(pop_size, n_features, rng)
    history = []

    best_individual = None
    best_fitness = -1.0

    for gen in range(generations):
        fitnesses = [fitness(ind, X_train, y_train, cache=cache) for ind in population]

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx].copy()

        history.append(best_fitness)

        if verbose:
            n_selected = int(population[gen_best_idx].sum())
            print(f"Gen {gen+1:3d}/{generations} | best so far: {best_fitness:.4f} | "
                  f"features: {n_selected} | cache: {len(cache)}")

        sorted_idx = np.argsort(fitnesses)[::-1]
        new_population = [population[i].copy() for i in sorted_idx[:elitism]]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, rng)
            parent2 = tournament_selection(population, fitnesses, rng)
            child1, child2 = crossover(parent1, parent2, rng, crossover_rate)
            child1 = mutate(child1, rng, mutation_rate)
            child2 = mutate(child2, rng, mutation_rate)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    return best_individual, best_fitness, history
