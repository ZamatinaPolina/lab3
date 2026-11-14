import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


REG_LR, REG_EPOCHS = 0.1, 500
CLF_LR, CLF_EPOCHS = 0.1, 2000
THRESHOLD = 0.5
SEED = 42


def _standardize(X_train, X_test):
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
    Xtr, Xte, mu, sd = _standardize(data.X_train, data.X_test)
    y = np.asarray(data.y_train, dtype=float).ravel()
    n, d = Xtr.shape
    w, b = np.zeros(d), 0.0
    history = []
    for _ in range(REG_EPOCHS):
        err = Xtr @ w + b - y
        history.append(float(np.mean(err ** 2)))
        w -= REG_LR * (2.0 / n) * (Xtr.T @ err)
        b -= REG_LR * (2.0 / n) * err.sum()
    pred = Xte @ w + b
    w_orig = w / sd
    b_orig = float(b - np.sum(w * mu / sd))
    return RegressionResult(weights=w_orig, bias=b_orig, mse_history=history, predictions=pred)

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Бинарная логистическая регрессия, пакетный GD по бинарной кросс-энтропии."""
    Xtr, Xte, _, _ = _standardize(data.X_train, data.X_test)
    y = np.asarray(data.y_train, dtype=float).ravel()
    n, d = Xtr.shape
    w, b = np.zeros(d), 0.0
    eps = 1e-12
    history = []
    for _ in range(CLF_EPOCHS):
        p = _sigmoid(Xtr @ w + b)
        history.append(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
        g = p - y
        w -= CLF_LR * (Xtr.T @ g) / n
        b -= CLF_LR * g.sum() / n
    proba = _sigmoid(Xte @ w + b)
    return ClassificationResult(probabilities=proba, predictions=(proba >= THRESHOLD).astype(int), loss_history=history)

def _metrics(y_true, proba, pred):
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнение own_logistic, sklearn LogisticRegression, DecisionTree и KNN на одном split."""
    ytr = np.asarray(data.y_train).ravel()
    yte = np.asarray(data.y_test).ravel()
    Xtr, Xte, _, _ = _standardize(data.X_train, data.X_test)
    own = train_logistic_classifier(data)
    out = {"own_logistic": _metrics(yte, own.probabilities, own.predictions)}
    models = {
        "logistic_regression": LogisticRegression(C=1e6, max_iter=5000),
        "decision_tree": DecisionTreeClassifier(random_state=SEED),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr)
        out[name] = _metrics(yte, m.predict_proba(Xte)[:, 1], m.predict(Xte))
    return BaselineComparisonResult(metrics=out)
