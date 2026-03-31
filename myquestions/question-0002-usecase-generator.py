import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def generar_caso_de_uso_evaluar_lift_por_deciles():
    rng = np.random.default_rng()

    n = int(rng.integers(80, 161))

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0.5, 1.2, n)

    logits = 1.1 * x1 - 0.8 * x2 + rng.normal(0, 0.6, n)
    probs_reales = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs_reales)

    if len(np.unique(y)) < 2:
        y[: n // 2] = 0
        y[n // 2 :] = 1
        rng.shuffle(y)

    score = probs_reales + rng.normal(0, 0.08, n)
    score = np.clip(score, 0, 1)

    df = pd.DataFrame({
        "score_modelo": score,
        "target": y
    })

    n_bins = int(rng.integers(4, 7))

    input_data = {
        "df": df.copy(),
        "score_col": "score_modelo",
        "target_col": "target",
        "n_bins": n_bins
    }

    tasa_global = df["target"].mean()

    segmentos = pd.qcut(
        df["score_modelo"],
        q=n_bins,
        duplicates="drop"
    )

    tabla = (
        df.assign(segmento_intervalo=segmentos)
        .groupby("segmento_intervalo", observed=False)
        .agg(
            score_min=("score_modelo", "min"),
            score_max=("score_modelo", "max"),
            n=("target", "size"),
            tasa_positivos=("target", "mean")
        )
        .reset_index()
    )

    tabla["lift"] = tabla["tasa_positivos"] / tasa_global

    tabla = tabla.sort_values("score_max", ascending=False).reset_index(drop=True)
    tabla["segmento"] = np.arange(1, len(tabla) + 1)

    tabla = tabla[
        ["segmento", "score_min", "score_max", "n", "tasa_positivos", "lift"]
    ].copy()

    auc = float(roc_auc_score(df["target"], df["score_modelo"]))

    output_data = {
        "tabla_lift": tabla,
        "auc": auc,
        "n_segmentos": int(len(tabla))
    }

    return input_data, output_data
