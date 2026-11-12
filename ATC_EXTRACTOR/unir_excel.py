"""Une los Excel de comportamiento (PXXX_SYY_PHASE.xlsx) en un único fichero.

Busca recursivamente en --carpeta los ficheros cuyo nombre sigue el patrón
P###_S##_<FASE>.xlsx (p. ej. P001_S01_NEUTRAL.xlsx) y apila, hoja por hoja,
TRIALS, QUESTIONNAIRES y EMOTIONS. Los ficheros de eye-tracking (###_##_ET*.xlsx)
no siguen el patrón y se ignoran.

Uso:
    python unir_excel.py --carpeta output
    python unir_excel.py --carpeta output --salida combinado.xlsx
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PATRON = re.compile(r"^P(\d{3})_S(\d{2})_([A-Z]+)\.xlsx$")
HOJAS = ["TRIALS", "QUESTIONNAIRES", "EMOTIONS"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--carpeta", required=True, type=Path,
                        help="Carpeta con los ficheros PXXX_SYY_PHASE.xlsx")
    parser.add_argument("--salida", type=Path, default=None,
                        help="Fichero de salida (por defecto <carpeta>/../datos_combinados.xlsx)")
    args = parser.parse_args()

    carpeta = args.carpeta.resolve()
    salida = (args.salida or carpeta.parent / "datos_combinados.xlsx").resolve()

    ficheros = sorted(
        f for f in carpeta.rglob("*.xlsx")
        if PATRON.match(f.name) and f.resolve() != salida
    )
    if not ficheros:
        sys.exit(f"No se encontraron ficheros PXXX_SYY_PHASE.xlsx en {carpeta}")

    vistos = set()
    partes = {h: [] for h in HOJAS}
    for f in ficheros:
        if f.name in vistos:
            print(f"  [aviso] duplicado ignorado: {f}")
            continue
        vistos.add(f.name)
        p, s, fase = PATRON.match(f.name).groups()
        libro = pd.read_excel(f, sheet_name=None)
        for hoja in HOJAS:
            if hoja not in libro:
                print(f"  [aviso] {f.name} no tiene hoja {hoja}")
                continue
            df = libro[hoja].copy()
            df.insert(0, "source_file", f.name)
            partes[hoja].append(df)
        print(f"  ok  {f.name}")

    with pd.ExcelWriter(salida, engine="openpyxl") as w:
        for hoja in HOJAS:
            if partes[hoja]:
                df = pd.concat(partes[hoja], ignore_index=True)
                df.to_excel(w, sheet_name=hoja, index=False)
                print(f"{hoja}: {len(df)} filas")

    print(f"\n{len(vistos)} ficheros unidos -> {salida}")


if __name__ == "__main__":
    main()
