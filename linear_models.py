
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

THRESHOLD = 0.5  
SEED = 42    


def _standardize(X_train, X_test):
    """z-score; среднее и σ считаются только по train."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd_safe, (X_test - mu) / sd_safe, mu, sd_safe


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Линейная регрессия, пакетный градиентный спуск по MSE."""
    Xtr, Xte, mu, sd = _standardize(data.train_features, data.test_features)
    y = np.asarray(data.train_target, dtype=float).ravel()
    n, d = Xtr.shape
    w, b = np.zeros(d), 0.0
    history = []
    for _ in range(data.epochs):
        err = Xtr @ w + b - y
        history.append(float(np.mean(err ** 2)))
        w -= data.learning_rate * (2.0 / n) * (Xtr.T @ err)
        b -= data.learning_rate * (2.0 / n) * err.sum()
    pred = Xte @ w + b
    w_orig = w / sd
    b_orig = float(b - np.sum(w * mu / sd))
    return RegressionResult(weights=w_orig, bias=b_orig, loss_history=history, predictions=pred)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Бинарная логистическая регрессия, пакетный GD по бинарной кросс-энтропии."""
    Xtr, Xte, mu, sd = _standardize(data.train_features, data.test_features)
    y = np.asarray(data.train_target, dtype=float).ravel()
    n, d = Xtr.shape
    w, b = np.zeros(d), 0.0
    eps = 1e-12
    history = []
    for _ in range(data.epochs):
        p = _sigmoid(Xtr @ w + b)
        history.append(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
        g = p - y
        w -= data.learning_rate * (Xtr.T @ g) / n
        b -= data.learning_rate * g.sum() / n
    proba = _sigmoid(Xte @ w + b)
    return ClassificationResult(
        weights=w / sd,
        bias=float(b - np.sum(w * mu / sd)),
        loss_history=history,
        probabilities=proba,
        predictions=(proba >= THRESHOLD).astype(int),
    )


def _metrics(y_true, proba, pred) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнение собственной логистической регрессии, DecisionTree и KNN на одном split."""
    ytr = np.asarray(data.train_target).ravel()
    yte = np.asarray(data.test_target).ravel()
    Xtr, Xte, _, _ = _standardize(data.train_features, data.test_features)

    own = train_logistic_classifier(data)
    tree = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr) 

    return BaselineComparisonResult(
        logistic_regression=_metrics(yte, own.probabilities, own.predictions),
        decision_tree=_metrics(yte, tree.predict_proba(Xte)[:, 1], tree.predict(Xte)),
        knn=_metrics(yte, knn.predict_proba(Xte)[:, 1], knn.predict(Xte)),
    )
