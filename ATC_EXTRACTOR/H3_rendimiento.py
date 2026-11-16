"""
H3. RENDIMIENTO

Hipótesis:
Una recomendación correcta aumenta la probabilidad de acertar.

Variable dependiente:
    acierto_binario
    1 = acierto
    0 = fallo o timeout

Variable independiente:
    recomendacion_correcta_binaria
    1 = recomendación correcta
    0 = recomendación incorrecta

Análisis:
Regresión logística binaria simple.
"""

from pathlib import Path
import sys
import unicodedata

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

ARCHIVO_DATOS = Path(
    r"C:\Users\Santiago Menasalvas\Desktop"
    r"\Trust ATC personalidad\datos_experimento.xlsx"
)

HOJA_EXCEL = 0

# Cambia estos encabezados si en tu Excel tienen otro nombre.
COLUMNA_RESULTADO = "resultado"

COLUMNA_RECOMENDACION_CORRECTA = (
    "recomendacion_correcta"
)

ARCHIVO_RESULTADOS = (
    ARCHIVO_DATOS.parent
    / "resultados_H3.xlsx"
)


# ============================================================
# 2. FUNCIONES
# ============================================================

def normalizar(valor):
    """
    Convierte un valor a texto, elimina espacios,
    pasa a minúsculas y elimina tildes.
    """

    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()

    texto = "".join(
        caracter
        for caracter in unicodedata.normalize(
            "NFD",
            texto
        )
        if unicodedata.category(caracter) != "Mn"
    )

    return texto


def codificar_resultado(valor):
    """
    Codificación de H3:

    1 = acierto
    0 = fallo, error o timeout
    """

    texto = normalizar(valor)

    valores_acierto = {
        "acierto",
        "correcto",
        "correcta",
        "1",
        "1.0"
    }

    valores_no_acierto = {
        "fallo",
        "error",
        "incorrecto",
        "incorrecta",
        "timeout",
        "time out",
        "tiempo agotado",
        "se acabo el tiempo",
        "sin respuesta",
        "0",
        "0.0"
    }

    if texto in valores_acierto:
        return 1

    if texto in valores_no_acierto:
        return 0

    return np.nan


def codificar_recomendacion(valor):
    """
    Codificación de la recomendación:

    1 = recomendación correcta
    0 = recomendación incorrecta
    """

    texto = normalizar(valor)

    valores_correcta = {
        "correcta",
        "correcto",
        "si",
        "yes",
        "true",
        "1",
        "1.0"
    }

    valores_incorrecta = {
        "incorrecta",
        "incorrecto",
        "no",
        "false",
        "0",
        "0.0"
    }

    if texto in valores_correcta:
        return 1

    if texto in valores_incorrecta:
        return 0

    return np.nan


def numero_espanol(numero, decimales=2):
    """
    Da formato español a un número:
    1.25 pasa a 1,25.
    """

    return f"{numero:.{decimales}f}".replace(
        ".",
        ","
    )


def formato_p(valor_p):
    """
    Da formato al valor p.
    """

    if valor_p < 0.001:
        return "p<0,001"

    return (
        f"p={valor_p:.3f}"
        .replace(".", ",")
    )


# ============================================================
# 3. IMPORTACIÓN DE DATOS
# ============================================================

if not ARCHIVO_DATOS.exists():
    sys.exit(
        "\nERROR: no se encuentra el archivo:\n"
        f"{ARCHIVO_DATOS}\n\n"
        "Modifica ARCHIVO_DATOS al principio "
        "del programa."
    )

datos_originales = pd.read_excel(
    ARCHIVO_DATOS,
    sheet_name=HOJA_EXCEL,
    engine="openpyxl"
)

datos_originales.columns = [
    str(columna).strip()
    for columna in datos_originales.columns
]

print("\nColumnas del Excel:")

for columna in datos_originales.columns:
    print(f"  - {columna}")


# ============================================================
# 4. COMPROBAR COLUMNAS
# ============================================================

columnas_necesarias = [
    COLUMNA_RESULTADO,
    COLUMNA_RECOMENDACION_CORRECTA
]

columnas_faltantes = [
    columna
    for columna in columnas_necesarias
    if columna not in datos_originales.columns
]

if columnas_faltantes:
    sys.exit(
        "\nERROR: faltan estas columnas:\n"
        + "\n".join(
            f"  - {columna}"
            for columna in columnas_faltantes
        )
    )


# ============================================================
# 5. CODIFICACIÓN
# ============================================================

datos_originales["acierto_binario"] = (
    datos_originales[
        COLUMNA_RESULTADO
    ].apply(codificar_resultado)
)

datos_originales[
    "recomendacion_correcta_binaria"
] = datos_originales[
    COLUMNA_RECOMENDACION_CORRECTA
].apply(codificar_recomendacion)


# Mostrar valores que no se han podido codificar

resultado_no_codificado = datos_originales.loc[
    datos_originales["acierto_binario"].isna(),
    COLUMNA_RESULTADO
].dropna().unique()

recomendacion_no_codificada = datos_originales.loc[
    datos_originales[
        "recomendacion_correcta_binaria"
    ].isna(),
    COLUMNA_RECOMENDACION_CORRECTA
].dropna().unique()

if len(resultado_no_codificado) > 0:
    print(
        "\nValores de resultado no reconocidos:"
    )

    for valor in resultado_no_codificado:
        print(f"  - {valor}")

if len(recomendacion_no_codificada) > 0:
    print(
        "\nValores de recomendación no reconocidos:"
    )

    for valor in recomendacion_no_codificada:
        print(f"  - {valor}")


# Conservar únicamente los casos válidos

datos = datos_originales[
    [
        COLUMNA_RESULTADO,
        COLUMNA_RECOMENDACION_CORRECTA,
        "acierto_binario",
        "recomendacion_correcta_binaria"
    ]
].dropna(
    subset=[
        "acierto_binario",
        "recomendacion_correcta_binaria"
    ]
).copy()

datos["acierto_binario"] = (
    datos["acierto_binario"].astype(int)
)

datos["recomendacion_correcta_binaria"] = (
    datos[
        "recomendacion_correcta_binaria"
    ].astype(int)
)


# ============================================================
# 6. COMPROBAR LAS CATEGORÍAS
# ============================================================

categorias_acierto = set(
    datos["acierto_binario"].unique()
)

categorias_recomendacion = set(
    datos[
        "recomendacion_correcta_binaria"
    ].unique()
)

if categorias_acierto != {0, 1}:
    sys.exit(
        "\nERROR: la variable de acierto debe "
        "contener las categorías 0 y 1.\n"
        f"Categorías encontradas: "
        f"{categorias_acierto}"
    )

if categorias_recomendacion != {0, 1}:
    sys.exit(
        "\nERROR: la recomendación debe contener "
        "las categorías 0 y 1.\n"
        f"Categorías encontradas: "
        f"{categorias_recomendacion}"
    )


# ============================================================
# 7. TABLA DE FRECUENCIAS
# ============================================================

tabla_frecuencias = pd.crosstab(
    datos["recomendacion_correcta_binaria"],
    datos["acierto_binario"],
    margins=True
)

tabla_frecuencias.index.name = (
    "Recomendación correcta"
)

tabla_frecuencias.columns.name = (
    "Acierto"
)


# ============================================================
# 8. PORCENTAJES
# ============================================================

tabla_porcentajes = (
    pd.crosstab(
        datos[
            "recomendacion_correcta_binaria"
        ],
        datos["acierto_binario"],
        normalize="index"
    )
    * 100
)

tabla_porcentajes.index.name = (
    "Recomendación correcta"
)

tabla_porcentajes.columns.name = (
    "Acierto"
)


# ============================================================
# 9. REGRESIÓN LOGÍSTICA
# ============================================================

X = datos[
    ["recomendacion_correcta_binaria"]
].copy()

X = sm.add_constant(
    X,
    has_constant="add"
)

y = datos["acierto_binario"]

try:
    modelo = sm.Logit(
        y,
        X
    ).fit(
        disp=False
    )

except Exception as error:
    sys.exit(
        "\nERROR: no se pudo calcular la "
        "regresión logística.\n"
        "Comprueba que existan aciertos y no "
        "aciertos en los grupos.\n\n"
        f"Detalle del error:\n{error}"
    )


# ============================================================
# 10. RESULTADOS
# ============================================================

intervalos = modelo.conf_int()

resultados = pd.DataFrame({
    "B": modelo.params,
    "SE": modelo.bse,
    "z": modelo.tvalues,
    "p": modelo.pvalues,
    "OR": np.exp(modelo.params),
    "IC95_inferior_OR": np.exp(
        intervalos[0]
    ),
    "IC95_superior_OR": np.exp(
        intervalos[1]
    )
})

resultado_predictor = resultados.loc[
    "recomendacion_correcta_binaria"
]

texto_resultado = (
    "H3. Recomendación correcta: "
    f"B={numero_espanol(resultado_predictor['B'])}; "
    f"SE={numero_espanol(resultado_predictor['SE'])}; "
    f"z={numero_espanol(resultado_predictor['z'])}; "
    f"{formato_p(resultado_predictor['p'])}; "
    f"OR={numero_espanol(resultado_predictor['OR'])}; "
    "IC 95 % "
    f"[{numero_espanol(resultado_predictor['IC95_inferior_OR'])}; "
    f"{numero_espanol(resultado_predictor['IC95_superior_OR'])}]."
)


# ============================================================
# 11. MOSTRAR RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("H3. RENDIMIENTO")
print("=" * 70)

print(f"\nCasos analizados: {len(datos)}")

print(
    "\nCodificación:"
    "\n  Acierto: 1"
    "\n  Fallo o timeout: 0"
    "\n  Recomendación correcta: 1"
    "\n  Recomendación incorrecta: 0"
)

print("\nTabla de frecuencias:")
print(tabla_frecuencias)

print("\nPorcentajes por grupo:")
print(
    tabla_porcentajes.round(2)
)

print("\nResultados de la regresión:")
print(
    resultados.round(4)
)

print("\nRESULTADO PARA EL INFORME:")
print(texto_resultado)


# ============================================================
# 12. GUARDAR RESULTADOS
# ============================================================

with pd.ExcelWriter(
    ARCHIVO_RESULTADOS,
    engine="openpyxl"
) as writer:

    datos.to_excel(
        writer,
        sheet_name="Datos codificados",
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

    resultados.to_excel(
        writer,
        sheet_name="Regresion"
    )

    pd.DataFrame({
        "Resultado para el informe": [
            texto_resultado
        ]
    }).to_excel(
        writer,
        sheet_name="Resultado redactado",
        index=False
    )

print("\nArchivo generado:")
print(ARCHIVO_RESULTADOS)