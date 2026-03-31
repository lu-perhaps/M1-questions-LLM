import numpy as np
import pandas as pd


def generar_caso_de_uso_detectar_saltos_anomalos():
    rng = np.random.default_rng()

    n_grupos = int(rng.integers(4, 8))
    grupos = [f"G{i+1:02d}" for i in range(n_grupos)]

    filas = []

    for grupo in grupos:
        n_fechas = int(rng.integers(10, 16))
        fecha_inicio = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 20)))
        fechas = pd.date_range(start=fecha_inicio, periods=n_fechas, freq="D")

        base = rng.normal(loc=50, scale=8, size=n_fechas)
        tendencia = np.linspace(0, rng.normal(0, 5), n_fechas)
        valores = base + tendencia

        if rng.random() < 0.8 and n_fechas >= 6:
            idx_salto = int(rng.integers(3, n_fechas))
            valores[idx_salto] += rng.choice([-1, 1]) * rng.uniform(25, 50)

        for fecha, valor in zip(fechas, valores):
            filas.append({
                "grupo": grupo,
                "fecha": fecha,
                "valor": float(valor)
            })

    df = pd.DataFrame(filas)

    ventana = int(rng.integers(2, 5))
    umbral = float(rng.choice([1.2, 1.5, 2.0, 2.5]))

    input_data = {
        "df": df.copy(),
        "grupo_col": "grupo",
        "fecha_col": "fecha",
        "valor_col": "valor",
        "ventana": ventana,
        "umbral": umbral
    }

    trabajo = df.copy()
    trabajo["fecha"] = pd.to_datetime(trabajo["fecha"])
    trabajo = trabajo.sort_values(["grupo", "fecha"]).reset_index(drop=True)

    promedio_movil = (
        trabajo.groupby("grupo")["valor"]
        .transform(lambda s: s.shift(1).rolling(window=ventana, min_periods=ventana).mean())
    )

    trabajo["promedio_movil"] = promedio_movil

    denominador = trabajo["promedio_movil"].abs()
    trabajo["desviacion_relativa"] = np.where(
        denominador > 0,
        (trabajo["valor"] - trabajo["promedio_movil"]).abs() / denominador,
        np.nan
    )

    trabajo["es_anomalia"] = (
        trabajo["promedio_movil"].notna()
        & (trabajo["desviacion_relativa"] > umbral)
    )

    output_data = trabajo[
        ["grupo", "fecha", "valor", "promedio_movil", "desviacion_relativa", "es_anomalia"]
    ].copy()

    return input_data, output_data