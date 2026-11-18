"""
H4. SEGUIMIENTO DE LA RECOMENDACIÓN SEGÚN LA FASE

Hipótesis:
El participante tiende más a seguir la recomendación de la
automatización en la fase ecológica que en la neutral.

Variable dependiente:
    seguimiento

    1 = la respuesta coincide con la recomendación
    0 = la respuesta no coincide o hubo no_response

Variable independiente:
    phase

    NEUTRAL = categoría de referencia
    ECOLOGICAL = fase que se compara con NEUTRAL

Clúster:
    participant

Análisis:
Regresión logística GEE binomial.
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
    / "resultados_H4_GEE_clusters.xlsx"
)


if not ruta_archivo.exists():
    raise FileNotFoundError(
        f"No se encuentra el archivo:\n{ruta_archivo}\n\n"
        "Coloca datos_combinados.xlsx en la misma carpeta "
        "que H4_GEE_clusters.py."
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


print("\n" + "=" * 70)
print("H4. SEGUIMIENTO DE LA RECOMENDACIÓN SEGÚN LA FASE")
print("=" * 70)

print(
    "\nArchivo:",
    ruta_archivo.name
)

print(
    "Filas originales:",
    len(tabla_analisis_regresion)
)


# ============================================================
# 2. COMPROBACIÓN DE COLUMNAS
# ============================================================

columnas_necesarias = [
    "participant",
    "phase",
    "conflict",
    "automation_error",
    "answer"
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


# Conservar solamente las fases válidas.

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


# Crear fase_binaria para facilitar la interpretación:
#
# NEUTRAL = 0
# ECOLOGICAL = 1

tabla_analisis_regresion["fase_binaria"] = (
    tabla_analisis_regresion[
        "phase"
    ].map({
        "NEUTRAL": 0,
        "ECOLOGICAL": 1
    })
)


# ============================================================
# 5. PREPARACIÓN DE CONFLICT
# ============================================================

tabla_analisis_regresion["conflict_limpio"] = (
    tabla_analisis_regresion["conflict"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# Si no existe conflicto, la acción correcta es maintain.
# Si existe conflicto, la acción correcta es change.

mapa_accion_correcta = {
    "no": "maintain",
    "false": "maintain",
    "0": "maintain",
    "0.0": "maintain",

    "yes": "change",
    "si": "change",
    "sí": "change",
    "true": "change",
    "1": "change",
    "1.0": "change"
}


tabla_analisis_regresion["accion_correcta"] = (
    tabla_analisis_regresion[
        "conflict_limpio"
    ].map(
        mapa_accion_correcta
    )
)


# ============================================================
# 6. PREPARACIÓN DE AUTOMATION_ERROR
# ============================================================

tabla_analisis_regresion[
    "automation_error_numerico"
] = pd.to_numeric(
    tabla_analisis_regresion[
        "automation_error"
    ],
    errors="coerce"
)


# ============================================================
# 7. RECONSTRUIR LA RECOMENDACIÓN DE LA AUTOMATIZACIÓN
# ============================================================

def obtener_recomendacion(fila):
    """
    Reconstruye la recomendación de la automatización.

    automation_error = 0:
        recomienda la acción correcta.

    automation_error = 1:
        recomienda la acción contraria.
    """

    accion_correcta = fila[
        "accion_correcta"
    ]

    automation_error = fila[
        "automation_error_numerico"
    ]

    if pd.isna(accion_correcta):
        return pd.NA

    if pd.isna(automation_error):
        return pd.NA

    if automation_error == 0:
        return accion_correcta

    if automation_error == 1:

        if accion_correcta == "maintain":
            return "change"

        if accion_correcta == "change":
            return "maintain"

    return pd.NA


tabla_analisis_regresion[
    "recomendacion_automatizacion"
] = tabla_analisis_regresion.apply(
    obtener_recomendacion,
    axis=1
)


# ============================================================
# 8. PREPARACIÓN DE LA RESPUESTA DEL PARTICIPANTE
# ============================================================

tabla_analisis_regresion["answer_limpio"] = (
    tabla_analisis_regresion["answer"]
    .astype("string")
    .str.strip()
    .str.lower()
)


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


# ============================================================
# 9. CREAR LA VARIABLE SEGUIMIENTO
# ============================================================

def calcular_seguimiento(fila):
    """
    1 = la respuesta coincide con la recomendación.
    0 = no coincide o hubo no_response.
    """

    respuesta = fila[
        "answer_limpio"
    ]

    recomendacion = fila[
        "recomendacion_automatizacion"
    ]

    if respuesta in valores_no_respuesta:
        return 0

    if pd.isna(recomendacion):
        return np.nan

    if respuesta == "":
        return np.nan

    if respuesta == recomendacion:
        return 1

    return 0


tabla_analisis_regresion["seguimiento"] = (
    tabla_analisis_regresion.apply(
        calcular_seguimiento,
        axis=1
    )
)


# ============================================================
# 10. SELECCIÓN DE CASOS VÁLIDOS
# ============================================================

columnas_modelo = [
    "source_file",
    "participant",
    "session",
    "phase",
    "fase_binaria",
    "block",
    "trial",
    "conflict",
    "automation_error",
    "answer",
    "accion_correcta",
    "recomendacion_automatizacion",
    "seguimiento"
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
            "fase_binaria",
            "recomendacion_automatizacion",
            "seguimiento"
        ]
    )
    .copy()
)


datos["fase_binaria"] = (
    datos["fase_binaria"]
    .astype(int)
)


datos["seguimiento"] = (
    datos["seguimiento"]
    .astype(int)
)


datos = datos[
    datos["fase_binaria"].isin([0, 1])
    &
    datos["seguimiento"].isin([0, 1])
].copy()


# ============================================================
# 11. COMPROBACIÓN DE LOS DATOS
# ============================================================

if datos.empty:
    raise ValueError(
        "No quedan ensayos válidos para realizar H4."
    )


if datos["participant"].nunique() < 2:
    raise ValueError(
        "Se necesitan al menos dos participantes."
    )


if datos["fase_binaria"].nunique() != 2:
    raise ValueError(
        "Deben existir ensayos de las fases "
        "NEUTRAL y ECOLOGICAL."
    )


if datos["seguimiento"].nunique() != 2:
    raise ValueError(
        "La variable seguimiento debe contener 0 y 1."
    )


# ============================================================
# 12. INFORMACIÓN DE LOS CLÚSTERES
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
# 13. COMPROBAR LA RECONSTRUCCIÓN DE LA RECOMENDACIÓN
# ============================================================

tabla_comprobacion = pd.crosstab(
    [
        datos["conflict"],
        datos["automation_error"]
    ],
    datos["recomendacion_automatizacion"],
    margins=True
)


print(
    "\nComprobación de las recomendaciones reconstruidas:"
)

print(tabla_comprobacion)


# ============================================================
# 14. TABLA DE FRECUENCIAS
# ============================================================

tabla_frecuencias = pd.crosstab(
    datos["phase"],
    datos["seguimiento"],
    margins=True
)


tabla_frecuencias = (
    tabla_frecuencias.rename(
        index={
            "NEUTRAL": "Fase neutral",
            "ECOLOGICAL": "Fase ecológica",
            "All": "Total"
        },
        columns={
            0: "No sigue",
            1: "Sigue",
            "All": "Total"
        }
    )
)


print("\nTabla de frecuencias:")

print(tabla_frecuencias)


# ============================================================
# 15. PORCENTAJES DE SEGUIMIENTO
# ============================================================

tabla_porcentajes = (
    pd.crosstab(
        datos["phase"],
        datos["seguimiento"],
        normalize="index"
    )
    * 100
)


tabla_porcentajes = (
    tabla_porcentajes.rename(
        index={
            "NEUTRAL": "Fase neutral",
            "ECOLOGICAL": "Fase ecológica"
        },
        columns={
            0: "No sigue (%)",
            1: "Sigue (%)"
        }
    )
)


print("\nPorcentajes de seguimiento:")

print(
    tabla_porcentajes.round(2)
)


# ============================================================
# 16. REGRESIÓN LOGÍSTICA GEE
# ============================================================

# Modelo:
#
# seguimiento ~ fase_binaria
#
# NEUTRAL = 0
# ECOLOGICAL = 1
#
# El coeficiente de fase_binaria indica si la probabilidad
# de seguir la recomendación es mayor en ECOLOGICAL.
#
# groups="participant" define los clústeres.

modelo = smf.gee(
    formula=(
        "seguimiento ~ fase_binaria"
    ),
    groups="participant",
    data=datos,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable()
)


resultado_modelo = modelo.fit()


# ============================================================
# 17. MOSTRAR EL MODELO
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS DEL MODELO GEE")
print("=" * 70)


print(
    resultado_modelo.summary()
)


# ============================================================
# 18. EXTRAER LOS RESULTADOS
# ============================================================

intervalos = (
    resultado_modelo.conf_int()
)


tabla_resultados = pd.DataFrame({
    "B": resultado_modelo.params,
    "SE robusto": resultado_modelo.bse,
    "z": resultado_modelo.tvalues,
    "p bilateral": resultado_modelo.pvalues,
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


efecto_h4 = tabla_resultados.loc[
    "fase_binaria"
]


# H4 plantea una dirección concreta:
#
# ECOLOGICAL > NEUTRAL.
#
# Si B es positivo, el p unilateral es p bilateral / 2.
#
# Si B es negativo, el p unilateral se calcula en la
# dirección contraria.

if efecto_h4["B"] >= 0:

    p_unilateral = (
        efecto_h4["p bilateral"]
        / 2
    )

else:

    p_unilateral = (
        1
        - efecto_h4["p bilateral"] / 2
    )


tabla_resultados.loc[
    "fase_binaria",
    "p unilateral H4"
] = p_unilateral


print("\nResultados principales:")

print(
    tabla_resultados.round(4)
)


# ============================================================
# 19. FORMATO DE LOS RESULTADOS
# ============================================================

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


# ============================================================
# 20. REDACTAR EL RESULTADO
# ============================================================

porcentaje_neutral = (
    datos.loc[
        datos["phase"] == "NEUTRAL",
        "seguimiento"
    ]
    .mean()
    * 100
)


porcentaje_ecologica = (
    datos.loc[
        datos["phase"] == "ECOLOGICAL",
        "seguimiento"
    ]
    .mean()
    * 100
)


diferencia_porcentual = (
    porcentaje_ecologica
    - porcentaje_neutral
)


texto_resultado = (
    "H4. El seguimiento de la recomendación fue del "
    f"{formato_decimal(porcentaje_neutral, 2)} % "
    "en la fase neutral y del "
    f"{formato_decimal(porcentaje_ecologica, 2)} % "
    "en la fase ecológica. "
    "Se aplicó una regresión logística GEE, agrupando "
    "los ensayos por participante. "
    "El efecto de la fase ecológica frente a la neutral fue: "
    f"B={formato_decimal(efecto_h4['B'])}; "
    f"SE robusto="
    f"{formato_decimal(efecto_h4['SE robusto'])}; "
    f"z={formato_decimal(efecto_h4['z'])}; "
    f"{formato_p(efecto_h4['p bilateral'])} bilateral; "
    f"{formato_p(p_unilateral)} unilateral; "
    f"OR={formato_decimal(efecto_h4['OR'])}; "
    "IC 95 % "
    f"[{formato_decimal(efecto_h4['IC95 inferior OR'])}; "
    f"{formato_decimal(efecto_h4['IC95 superior OR'])}]."
)


print("\n" + "=" * 70)
print("RESULTADO PARA H4")
print("=" * 70)


print(texto_resultado)


print(
    "\nDiferencia en puntos porcentuales:",
    formato_decimal(
        diferencia_porcentual,
        2
    )
)


# ============================================================
# 21. INTERPRETACIÓN
# ============================================================

if (
    p_unilateral < 0.05
    and efecto_h4["OR"] > 1
):

    interpretacion = (
        "La probabilidad de seguir la recomendación es "
        "significativamente mayor en la fase ecológica "
        "que en la neutral. H4 queda respaldada."
    )


elif efecto_h4["p bilateral"] < 0.05:

    interpretacion = (
        "Existe una diferencia significativa entre fases, "
        "pero aparece en dirección contraria a H4."
    )


else:

    interpretacion = (
        "No se observa una diferencia estadísticamente "
        "significativa en el seguimiento entre la fase "
        "ecológica y la neutral. H4 no queda respaldada."
    )


print("\nInterpretación:")

print(interpretacion)


# ============================================================
# 22. GUARDAR LOS RESULTADOS
# ============================================================

resultado_redactado = pd.DataFrame({
    "Indicador": [
        "Seguimiento en neutral (%)",
        "Seguimiento en ecológica (%)",
        "Diferencia en puntos porcentuales",
        "Resultado",
        "Interpretación"
    ],
    "Valor": [
        porcentaje_neutral,
        porcentaje_ecologica,
        diferencia_porcentual,
        texto_resultado,
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
            sheet_name="Datos H4",
            index=False
        )

        tabla_comprobacion.to_excel(
            writer,
            sheet_name="Comprobar recomendacion"
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
        "Cierra resultados_H4_GEE_clusters.xlsx "
        "y vuelve a ejecutar el programa."
    ) from error


print("\nAnálisis terminado.")

print(
    "Archivo generado:",
    ruta_resultados
)