from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {list(data.target_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

baseline_accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"\nBaseline accuracy (all 30 features): {baseline_accuracy:.4f}")
