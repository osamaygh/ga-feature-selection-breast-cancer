import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from genetic_algorithm import fitness as ga_fitness


def load_split(test_size=0.3, random_state=42):
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, data.feature_names


def evaluate_baseline(X_train, X_test, y_train, y_test):
    n_features = X_train.shape[1]

    cv_accuracy = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X_train, y_train, cv=5
    ).mean()

    search_fitness = ga_fitness(np.ones(n_features, dtype=int), X_train, y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))

    return {
        "cv_accuracy": cv_accuracy,
        "search_fitness": search_fitness,
        "test_accuracy": test_accuracy,
    }


def evaluate_selected_features(X_train, X_test, y_train, y_test, selected_idx):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train[:, selected_idx], y_train)
    return accuracy_score(y_test, clf.predict(X_test[:, selected_idx]))
