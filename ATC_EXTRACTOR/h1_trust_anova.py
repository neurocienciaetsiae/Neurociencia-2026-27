from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


DEFAULT_INPUT_DIR = Path("output")
DEFAULT_OUTPUT_DIR = Path("output")
BLOCKS_H1 = [2, 3, 4, 5]
PHASES_H1 = ["NEUTRAL", "ECOLOGICAL"]
RESULTS_XLSX_NAME = "h1_trust_results.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "H1: ANOVA intrasujetos Fase (2) x Bloque (4) sobre confianza media por bloque, "
            "con contrastes a priori 3 vs 4 y 4 vs 5."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directorio con los archivos Excel exportados (por defecto: output/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio para guardar resultados (por defecto: output/).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="P*_S*_*.xlsx",
        help="Patrón de ficheros Excel a procesar.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Si se activa, además del .xlsx también exporta CSV.",
    )
    return parser.parse_args()


def load_trials_from_exports(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos con patrón {pattern!r} en {input_dir}."
        )

    records: list[pd.DataFrame] = []

    for file_path in files:
        try:
            trials = pd.read_excel(file_path, sheet_name="TRIALS")
        except Exception as exc:  # pragma: no cover
            print(f"[AVISO] No se pudo leer TRIALS en {file_path.name}: {exc}")
            continue

        required = {"participant", "phase", "block", "trust"}
        if not required.issubset(trials.columns):
            missing = required.difference(trials.columns)
            print(
                f"[AVISO] {file_path.name} no tiene columnas requeridas: {sorted(missing)}"
            )
            continue

        data = trials[["participant", "phase", "block", "trust"]].copy()
        data["source_file"] = file_path.name
        records.append(data)

    if not records:
        raise ValueError("No se pudo cargar ninguna hoja TRIALS válida.")

    out = pd.concat(records, ignore_index=True)
    out["trust"] = pd.to_numeric(out["trust"], errors="coerce")
    out["block"] = pd.to_numeric(out["block"], errors="coerce").astype("Int64")
    out["phase"] = out["phase"].astype(str).str.upper().str.strip()

    participant_numeric = pd.to_numeric(out["participant"], errors="coerce")
    out["participant"] = np.where(
        participant_numeric.notna(),
        participant_numeric.astype("Int64").astype(str).str.zfill(3),
        out["participant"].astype(str),
    )

    return out


def aggregate_mean_trust(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data.loc[
        data["phase"].isin(PHASES_H1)
        & data["block"].isin(BLOCKS_H1)
        & data["trust"].notna()
    ].copy()

    means = (
        filtered.groupby(["participant", "phase", "block"], as_index=False)["trust"]
        .mean()
        .rename(columns={"trust": "mean_trust"})
    )

    return means


def keep_complete_participants(cell_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    expected_cells = len(PHASES_H1) * len(BLOCKS_H1)

    counts = cell_means.groupby("participant").size()
    complete_participants = counts[counts == expected_cells].index

    complete = cell_means[cell_means["participant"].isin(complete_participants)].copy()
    return complete, complete_participants


def run_rm_anova(complete_means: pd.DataFrame) -> pd.DataFrame:
    try:
        statsmodels_anova = __import__(
            "statsmodels.stats.anova",
            fromlist=["AnovaRM"],
        )
        anova_rm = getattr(statsmodels_anova, "AnovaRM")
    except Exception as exc:
        raise ImportError(
            "No se pudo importar statsmodels. Instala el paquete con: pip install statsmodels"
        ) from exc

    anova_data = complete_means.copy()
    anova_data["block"] = anova_data["block"].astype(str)

    model = anova_rm(
        data=anova_data,
        depvar="mean_trust",
        subject="participant",
        within=["phase", "block"],
    )
    fit = model.fit()
    table = fit.anova_table.reset_index().rename(columns={"index": "effect"})
    return table


def planned_contrasts(complete_means: pd.DataFrame) -> pd.DataFrame:
    # Contrastes a priori sobre bloque, colapsando fase dentro de participante.
    block_by_participant = (
        complete_means.groupby(["participant", "block"], as_index=False)["mean_trust"]
        .mean()
        .pivot(index="participant", columns="block", values="mean_trust")
        .sort_index(axis=1)
    )

    def paired_result(block_a: int, block_b: int, label: str) -> dict:
        x = block_by_participant[block_a]
        y = block_by_participant[block_b]

        t_stat, p_value = ttest_rel(x, y, nan_policy="omit")
        diff = x - y
        dz = diff.mean() / diff.std(ddof=1)

        return {
            "contrast": label,
            "block_a": block_a,
            "block_b": block_b,
            "n": int(diff.notna().sum()),
            "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "mean_diff_a_minus_b": float(diff.mean()),
            "t": float(t_stat),
            "p_uncorrected": float(p_value),
            "cohens_dz": float(dz),
        }

    results = [
        paired_result(3, 4, "B3_vs_B4"),
        paired_result(4, 5, "B4_vs_B5"),
    ]
    contrasts = pd.DataFrame(results)

    # Holm para 2 contrastes a priori.
    contrasts = contrasts.sort_values("p_uncorrected").reset_index(drop=True)
    m = len(contrasts)
    holm = []
    for i, p in enumerate(contrasts["p_uncorrected"], start=1):
        holm.append(min((m - i + 1) * p, 1.0))
    contrasts["p_holm"] = holm

    # Volvemos al orden lógico de hipótesis.
    order = {"B3_vs_B4": 0, "B4_vs_B5": 1}
    contrasts["_order"] = contrasts["contrast"].map(order)
    contrasts = contrasts.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return contrasts


def descriptives(complete_means: pd.DataFrame) -> pd.DataFrame:
    desc = (
        complete_means.groupby(["phase", "block"])["mean_trust"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_participants"})
    )
    return desc


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_trials_from_exports(args.input_dir, args.pattern)
    cell_means = aggregate_mean_trust(raw_data)
    complete_means, complete_participants = keep_complete_participants(cell_means)

    if complete_means.empty:
        raise ValueError(
            "No hay participantes completos para Fase(2) x Bloque(4). "
            "Revisa datos de trust en bloques 2, 3, 4 y 5 para NEUTRAL/ECOLOGICAL."
        )

    anova_table = run_rm_anova(complete_means)
    contrasts = planned_contrasts(complete_means)
    desc = descriptives(complete_means)

    cell_means_path = args.output_dir / "h1_trust_cell_means.csv"
    desc_path = args.output_dir / "h1_trust_descriptives.csv"
    anova_path = args.output_dir / "h1_trust_anova_table.csv"
    contrasts_path = args.output_dir / "h1_trust_contrasts.csv"
    xlsx_path = args.output_dir / RESULTS_XLSX_NAME

    if args.export_csv:
        complete_means.sort_values(["participant", "phase", "block"]).to_csv(
            cell_means_path, index=False
        )
        desc.to_csv(desc_path, index=False)
        anova_table.to_csv(anova_path, index=False)
        contrasts.to_csv(contrasts_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        complete_means.sort_values(["participant", "phase", "block"]).to_excel(
            writer,
            sheet_name="cell_means",
            index=False,
        )
        desc.to_excel(writer, sheet_name="descriptives", index=False)
        anova_table.to_excel(writer, sheet_name="anova", index=False)
        contrasts.to_excel(writer, sheet_name="contrasts", index=False)

    print("=== H1: Confianza por bloque ===")
    print(f"Participantes incluidos (completos): {len(complete_participants)}")
    print(f"Participantes excluidos: {cell_means['participant'].nunique() - len(complete_participants)}")
    print("\nANOVA intrasujetos Fase(2) x Bloque(4):")
    print(anova_table.to_string(index=False))
    print("\nContrastes a priori:")
    print(contrasts.to_string(index=False))
    print("\nArchivos guardados:")
    if args.export_csv:
        print(f"- {cell_means_path}")
        print(f"- {desc_path}")
        print(f"- {anova_path}")
        print(f"- {contrasts_path}")
    print(f"- {xlsx_path}")


if __name__ == "__main__":
    main()
