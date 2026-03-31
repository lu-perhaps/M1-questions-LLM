import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def generar_caso_de_uso_evaluar_modelo_por_grupos():
    rng = np.random.default_rng()

    n_groups = int(rng.integers(8, 13))
    group_sizes = rng.integers(6, 13, size=n_groups)

    rows = []
    group_effects = rng.normal(0, 0.8, size=n_groups)

    for g, size in enumerate(group_sizes):
        for _ in range(int(size)):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(1, 1.2)
            x3 = rng.uniform(-2, 2)

            logit = 1.1 * x1 - 0.7 * x2 + 0.5 * x3 + group_effects[g]
            prob = 1 / (1 + np.exp(-logit))
            y = rng.binomial(1, prob)

            rows.append({
                "grupo": f"G{g+1:02d}",
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "target": y
            })

    df = pd.DataFrame(rows)

    if df["target"].nunique() < 2:
        half = len(df) // 2
        df.loc[:half, "target"] = 0
        df.loc[half:, "target"] = 1

    n_splits = int(rng.integers(3, min(6, n_groups + 1)))

    input_data = {
        "df": df.copy(),
        "group_col": "grupo",
        "target_col": "target",
        "n_splits": n_splits,
    }

    X = df.drop(columns=["grupo", "target"]).select_dtypes(include=[np.number])
    y = df["target"].to_numpy()
    groups = df["grupo"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(float(acc))

    accuracies = np.array(accuracies, dtype=float)

    output_data = {
        "accuracy_promedio": float(np.mean(accuracies)),
        "accuracy_por_fold": accuracies,
        "n_grupos": int(df["grupo"].nunique()),
    }

    return input_data, output_data