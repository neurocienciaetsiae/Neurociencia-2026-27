"""
H3. REGRESIÓN LOGÍSTICA GEE CON CLÚSTERES

Hipótesis:
Una recomendación correcta aumenta la probabilidad de acertar.

Variable dependiente:
    acierto
    1 = acierto
    0 = fallo o no_response

Variable independiente principal:
    recomendacion_correcta
    1 = recomendación correcta
    0 = recomendación incorrecta

Variable de control:
    phase
    NEUTRAL
    ECOLOGICAL

Clúster:
    participant

Cada participante puede aportar varios ensayos en ambas fases.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# 1. IMPORTACIÓN DE DATOS
# ============================================================

carpeta_script = Path(__file__).resolve().parent

ruta_archivo = (
    carpeta_script
    / "datos_combinados.xlsx"
)

ruta_resultados = (
    carpeta_script
    / "resultados_H3_GEE_clusters.xlsx"
)


if not ruta_archivo.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo:\n{ruta_archivo}\n\n"
        "Coloca datos_combinados.xlsx en la misma carpeta "
        "que H3_GEE_clusters.py."
    )


analisis_regresion = pd.read_excel(
    ruta_archivo,
    engine="openpyxl"
)


tabla_analisis_regresion = pd.DataFrame(
    analisis_regresion
)


tabla_analisis_regresion.columns = [
    str(columna).strip()
    for columna in tabla_analisis_regresion.columns
]


# ============================================================
# 2. COMPROBACIÓN DE COLUMNAS
# ============================================================

columnas_necesarias = [
    "participant",
    "phase",
    "automation_error",
    "answer",
    "correct"
]


columnas_faltantes = [
    columna
    for columna in columnas_necesarias
    if columna not in tabla_analisis_regresion.columns
]


if columnas_faltantes:
    raise KeyError(
        "Faltan estas columnas:\n"
        + "\n".join(
            f"  - {columna}"
            for columna in columnas_faltantes
        )
        + "\n\nColumnas disponibles:\n"
        + "\n".join(
            f"  - {columna}"
            for columna
            in tabla_analisis_regresion.columns
        )
    )


# ============================================================
# 3. PREPARACIÓN DEL PARTICIPANTE
# ============================================================

tabla_analisis_regresion["participant"] = (
    tabla_analisis_regresion["participant"]
    .astype("string")
    .str.strip()
    .str.replace(
        r"\.0$",
        "",
        regex=True
    )
)


# ============================================================
# 4. PREPARACIÓN DE LA FASE
# ============================================================

tabla_analisis_regresion["phase"] = (
    tabla_analisis_regresion["phase"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# Se admiten solamente las dos fases del experimento.

tabla_analisis_regresion = (
    tabla_analisis_regresion[
        tabla_analisis_regresion[
            "phase"
        ].isin(
            [
                "NEUTRAL",
                "ECOLOGICAL"
            ]
        )
    ]
    .copy()
)


# ============================================================
# 5. PREPARACIÓN DE LA RESPUESTA
# ============================================================

tabla_analisis_regresion["answer_limpio"] = (
    tabla_analisis_regresion["answer"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# ============================================================
# 6. CREAR LA VARIABLE ACIERTO
# ============================================================

tabla_analisis_regresion["acierto"] = (
    pd.to_numeric(
        tabla_analisis_regresion["correct"],
        errors="coerce"
    )
)


# Los ensayos sin respuesta se consideran no aciertos.

valores_no_respuesta = {
    "no_response",
    "no response",
    "timeout",
    "time out",
    "sin respuesta",
    "tiempo agotado",
    "se acabo el tiempo",
    "se acabó el tiempo"
}


es_no_respuesta = (
    tabla_analisis_regresion[
        "answer_limpio"
    ].isin(valores_no_respuesta)
)


tabla_analisis_regresion.loc[
    es_no_respuesta,
    "acierto"
] = 0


# ============================================================
# 7. CREAR RECOMENDACIÓN CORRECTA
# ============================================================

tabla_analisis_regresion[
    "automation_error_numerico"
] = pd.to_numeric(
    tabla_analisis_regresion[
        "automation_error"
    ],
    errors="coerce"
)


# automation_error = 0:
# recomendación correcta = 1
#
# automation_error = 1:
# recomendación correcta = 0

tabla_analisis_regresion[
    "recomendacion_correcta"
] = (
    tabla_analisis_regresion[
        "automation_error_numerico"
    ]
    .map({
        0: 1,
        1: 0
    })
)


# ============================================================
# 8. SELECCIÓN DE CASOS VÁLIDOS
# ============================================================

columnas_modelo = [
    "participant",
    "session",
    "phase",
    "block",
    "trial",
    "answer",
    "correct",
    "automation_error",
    "acierto",
    "recomendacion_correcta"
]


columnas_modelo = [
    columna
    for columna in columnas_modelo
    if columna in tabla_analisis_regresion.columns
]


datos = (
    tabla_analisis_regresion[
        columnas_modelo
    ]
    .dropna(
        subset=[
            "participant",
            "phase",
            "acierto",
            "recomendacion_correcta"
        ]
    )
    .copy()
)


datos["acierto"] = (
    datos["acierto"]
    .astype(int)
)


datos["recomendacion_correcta"] = (
    datos[
        "recomendacion_correcta"
    ]
    .astype(int)
)


datos = datos[
    datos["acierto"].isin([0, 1])
    &
    datos[
        "recomendacion_correcta"
    ].isin([0, 1])
].copy()


# ============================================================
# 9. COMPROBACIÓN DE LOS DATOS
# ============================================================

if datos.empty:
    raise ValueError(
        "No quedan ensayos válidos para realizar H3."
    )


if datos["participant"].nunique() < 2:
    raise ValueError(
        "Se necesitan al menos dos participantes."
    )


if datos["acierto"].nunique() != 2:
    raise ValueError(
        "La variable acierto debe contener 0 y 1."
    )


if datos["recomendacion_correcta"].nunique() != 2:
    raise ValueError(
        "La variable recomendacion_correcta "
        "debe contener 0 y 1."
    )


if datos["phase"].nunique() != 2:
    raise ValueError(
        "La variable phase debe contener "
        "NEUTRAL y ECOLOGICAL."
    )


# ============================================================
# 10. INFORMACIÓN DE LOS CLÚSTERES
# ============================================================

numero_clusters = (
    datos["participant"]
    .nunique()
)


ensayos_por_cluster = (
    datos.groupby("participant")
    .size()
    .describe()
)


ensayos_por_participante_fase = (
    datos.groupby(
        [
            "participant",
            "phase"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print("\n" + "=" * 70)
print("H3. REGRESIÓN LOGÍSTICA GEE")
print("=" * 70)


print(
    "\nEnsayos analizados:",
    len(datos)
)


print(
    "Número de clústeres:",
    numero_clusters
)


print(
    "Fases incluidas:",
    sorted(datos["phase"].unique())
)


print("\nResumen de ensayos por clúster:")

print(ensayos_por_cluster)


print("\nEnsayos por participante y fase:")

print(ensayos_por_participante_fase)


# ============================================================
# 11. TABLA DE FRECUENCIAS
# ============================================================

tabla_frecuencias = pd.crosstab(
    datos["recomendacion_correcta"],
    datos["acierto"],
    margins=True
)


tabla_frecuencias = (
    tabla_frecuencias.rename(
        index={
            0: "Recomendación incorrecta",
            1: "Recomendación correcta",
            "All": "Total"
        },
        columns={
            0: "No acierto",
            1: "Acierto",
            "All": "Total"
        }
    )
)


print("\nTabla de frecuencias:")

print(tabla_frecuencias)


# ============================================================
# 12. PORCENTAJES DE ACIERTO
# ============================================================

tabla_porcentajes = (
    pd.crosstab(
        datos["recomendacion_correcta"],
        datos["acierto"],
        normalize="index"
    )
    * 100
)


tabla_porcentajes = (
    tabla_porcentajes.rename(
        index={
            0: "Recomendación incorrecta",
            1: "Recomendación correcta"
        },
        columns={
            0: "No acierto (%)",
            1: "Acierto (%)"
        }
    )
)


print("\nPorcentajes:")

print(
    tabla_porcentajes.round(2)
)


# ============================================================
# 13. REGRESIÓN LOGÍSTICA GEE
# ============================================================

# Modelo:
#
# acierto ~ recomendacion_correcta + fase
#
# La fase NEUTRAL se establece como categoría de referencia.
#
# El efecto de recomendacion_correcta indica si una
# recomendación correcta aumenta la probabilidad de acertar,
# controlando las diferencias entre fases.
#
# groups="participant" establece los clústeres.

formula_modelo = (
    "acierto ~ recomendacion_correcta "
    "+ C(phase, Treatment(reference='NEUTRAL'))"
)


modelo = smf.gee(
    formula=formula_modelo,
    groups="participant",
    data=datos,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable()
)


resultado_modelo = modelo.fit()


# ============================================================
# 14. MOSTRAR EL MODELO
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS DEL MODELO GEE")
print("=" * 70)


print(
    resultado_modelo.summary()
)


# ============================================================
# 15. EXTRAER LOS RESULTADOS
# ============================================================

intervalos = (
    resultado_modelo.conf_int()
)


tabla_resultados = pd.DataFrame({
    "B": resultado_modelo.params,
    "SE robusto": resultado_modelo.bse,
    "z": resultado_modelo.tvalues,
    "p": resultado_modelo.pvalues,
    "OR": np.exp(
        resultado_modelo.params
    ),
    "IC95 inferior OR": np.exp(
        intervalos[0]
    ),
    "IC95 superior OR": np.exp(
        intervalos[1]
    )
})


print("\nResultados principales:")

print(
    tabla_resultados.round(4)
)


# ============================================================
# 16. EXTRAER EL EFECTO DE H3
# ============================================================

efecto_h3 = tabla_resultados.loc[
    "recomendacion_correcta"
]


def formato_decimal(valor, decimales=3):

    return (
        f"{valor:.{decimales}f}"
        .replace(".", ",")
    )


def formato_p(valor):

    if valor < 0.001:
        return "p<0,001"

    return (
        f"p={valor:.3f}"
        .replace(".", ",")
    )


texto_resultado = (
    "H3. Se aplicó una regresión logística GEE, "
    "agrupando los ensayos por participante y "
    "controlando la fase experimental. "
    "El efecto de recibir una recomendación correcta fue: "
    f"B={formato_decimal(efecto_h3['B'])}; "
    f"SE robusto="
    f"{formato_decimal(efecto_h3['SE robusto'])}; "
    f"z={formato_decimal(efecto_h3['z'])}; "
    f"{formato_p(efecto_h3['p'])}; "
    f"OR={formato_decimal(efecto_h3['OR'])}; "
    "IC 95 % "
    f"[{formato_decimal(efecto_h3['IC95 inferior OR'])}; "
    f"{formato_decimal(efecto_h3['IC95 superior OR'])}]."
)


print("\n" + "=" * 70)
print("RESULTADO PARA H3")
print("=" * 70)


print(texto_resultado)


# ============================================================
# 17. INTERPRETACIÓN
# ============================================================

if (
    efecto_h3["p"] < 0.05
    and efecto_h3["OR"] > 1
):

    interpretacion = (
        "Una recomendación correcta aumenta "
        "significativamente la probabilidad de acertar, "
        "controlando la fase y la agrupación de los ensayos "
        "por participante. H3 queda respaldada."
    )


elif efecto_h3["p"] < 0.05:

    interpretacion = (
        "El efecto es estadísticamente significativo, "
        "pero aparece en dirección contraria a H3."
    )


else:

    interpretacion = (
        "La recomendación correcta no presenta un efecto "
        "estadísticamente significativo sobre la probabilidad "
        "de acierto. H3 no queda respaldada."
    )


print("\nInterpretación:")

print(interpretacion)


# ============================================================
# 18. GUARDAR LOS RESULTADOS
# ============================================================

resultado_redactado = pd.DataFrame({
    "Resultado": [
        texto_resultado
    ],
    "Interpretación": [
        interpretacion
    ]
})


resumen_clusters = pd.DataFrame({
    "Indicador": [
        "Ensayos analizados",
        "Número de clústeres",
        "Mínimo de ensayos por clúster",
        "Máximo de ensayos por clúster",
        "Media de ensayos por clúster"
    ],
    "Valor": [
        len(datos),
        numero_clusters,
        ensayos_por_cluster["min"],
        ensayos_por_cluster["max"],
        ensayos_por_cluster["mean"]
    ]
})


try:
    with pd.ExcelWriter(
        ruta_resultados,
        engine="openpyxl"
    ) as writer:

        datos.to_excel(
            writer,
            sheet_name="Datos H3",
            index=False
        )

        tabla_frecuencias.to_excel(
            writer,
            sheet_name="Frecuencias"
        )

        tabla_porcentajes.to_excel(
            writer,
            sheet_name="Porcentajes"
        )

        tabla_resultados.to_excel(
            writer,
            sheet_name="Modelo GEE"
        )

        ensayos_por_participante_fase.to_excel(
            writer,
            sheet_name="Clusters"
        )

        resumen_clusters.to_excel(
            writer,
            sheet_name="Resumen clusters",
            index=False
        )

        resultado_redactado.to_excel(
            writer,
            sheet_name="Resultado redactado",
            index=False
        )


except PermissionError as error:

    raise PermissionError(
        "Cierra resultados_H3_GEE_clusters.xlsx "
        "y vuelve a ejecutar el programa."
    ) from error


print("\nAnálisis terminado.")

print(
    "Archivo generado:",
    ruta_resultados
)