import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def generar_caso_de_uso_calcular_dependencia_parcial():
    rng = np.random.default_rng()

    n = int(rng.integers(100, 180))

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0.5, 1.2, n)
    x3 = rng.uniform(-2, 2, n)
    x4 = rng.normal(0, 1, n)

    logits = 1.5 * x1 - 1.0 * x2 + 0.8 * x3 + rng.normal(0, 0.6, n)
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)

    if len(np.unique(y)) < 2:
        y[: n // 2] = 0
        y[n // 2:] = 1
        rng.shuffle(y)

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "target": y
    })

    feature_objetivo = rng.choice(["x1", "x2", "x3", "x4"])
    grid_size = int(rng.integers(8, 15))

    input_data = {
        "df": df.copy(),
        "target_col": "target",
        "feature_objetivo": feature_objetivo,
        "grid_size": grid_size
    }

    # ==========================
    # Calcular salida esperada
    # ==========================
    X = df.drop(columns=["target"])
    y_arr = df["target"].to_numpy()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_arr)

    feature_values = X[feature_objetivo]
    grid = np.linspace(feature_values.min(), feature_values.max(), grid_size)

    preds = []

    for val in grid:
        X_temp = X.copy()
        X_temp[feature_objetivo] = val
        prob = model.predict_proba(X_temp)[:, 1].mean()
        preds.append(float(prob))

    output_data = {
        "valores_feature": grid,
        "predicciones_promedio": np.array(preds),
        "nombre_feature": feature_objetivo
    }

    return input_data, output_data