"""Análisis reproducible de H2: cambio de confianza según el error de la automatización.

Ejecutar desde cualquier directorio:
    python ATC_EXTRACTOR/h2_trust_analysis.py

Usa el libro consolidado datos_combinados.xlsx y escribe informe, tablas y
figuras en output/h2_trust_change/.
"""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill
from scipy import stats
from scipy.stats import rankdata


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "datos_combinados.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "h2_trust_change"
REQUIRED_COLUMNS = {
    "participant",
    "session",
    "phase",
    "block",
    "trial",
    "automation_error",
    "answer",
    "correct",
    "trust",
}
AUTOMATED_BLOCKS = (2, 3, 4, 5)
GROUP_KEYS = ["participant", "session", "phase", "block"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Libro consolidado con hoja TRIALS (por defecto: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Carpeta de resultados (por defecto: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def load_trials(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el libro de entrada: {path}")
    try:
        trials = pd.read_excel(path, sheet_name="TRIALS")
    except Exception as exc:
        raise ValueError(f"No se pudo leer la hoja TRIALS de {path}: {exc}") from exc

    missing = REQUIRED_COLUMNS.difference(trials.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en TRIALS: {sorted(missing)}")

    data = trials.copy()
    for column in ("session", "block", "trial", "automation_error", "correct", "trust"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["phase"] = data["phase"].astype("string").str.strip().str.upper()
    data["answer"] = data["answer"].astype("string").str.strip().str.lower()
    data["participant"] = data["participant"].map(normalize_participant)
    return data


def normalize_participant(value: object) -> str:
    if pd.isna(value):
        return ""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(number) and float(number).is_integer():
        return f"P{int(number):03d}"
    return str(value).strip()


def build_trial_deltas(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return adjacent-trial trust changes, labeled by the current trial outcome."""
    counts = {
        "input_rows": int(len(data)),
        "manual_rows_excluded": int(data["block"].eq(1).sum()),
        "outside_blocks_2_5_excluded": int((~data["block"].isin(AUTOMATED_BLOCKS)).sum()),
    }

    automated = data.loc[data["block"].isin(AUTOMATED_BLOCKS)].copy()
    automated = automated.sort_values(GROUP_KEYS + ["trial"], kind="stable")
    grouped = automated.groupby(GROUP_KEYS, sort=False, dropna=False)
    automated["prev_trial"] = grouped["trial"].shift(1)
    automated["prev_trust"] = grouped["trust"].shift(1)
    automated["prev_answer"] = grouped["answer"].shift(1)
    automated["prev_automation_error"] = grouped["automation_error"].shift(1)

    adjacent = automated["trial"].sub(automated["prev_trial"]).eq(1)
    current_responded = automated["answer"].ne("no_response") & automated["answer"].notna()
    previous_responded = automated["prev_answer"].ne("no_response") & automated["prev_answer"].notna()
    current_valid = (
        automated["trust"].notna()
        & automated["automation_error"].isin([0, 1])
        & current_responded
    )
    previous_valid = (
        automated["prev_trust"].notna()
        & automated["prev_automation_error"].isin([0, 1])
        & previous_responded
    )
    keep = adjacent & current_valid & previous_valid

    counts["automated_rows"] = int(len(automated))
    counts["no_response_rows_in_automated_blocks"] = int(
        automated["answer"].eq("no_response").sum()
    )
    counts["rows_missing_trust_in_automated_blocks"] = int(automated["trust"].isna().sum())
    counts["rows_missing_automation_error_in_automated_blocks"] = int(
        automated["automation_error"].isna().sum()
    )
    counts["nonadjacent_rows"] = int((~adjacent).sum())
    counts["valid_deltas"] = int(keep.sum())

    deltas = automated.loc[keep].copy()
    deltas["delta_trust"] = deltas["trust"] - deltas["prev_trust"]
    deltas["automation_outcome"] = np.where(
        deltas["automation_error"].eq(0), "acierto", "fallo"
    )
    counts["participants_with_valid_deltas"] = int(deltas["participant"].nunique())
    counts["valid_failure_deltas"] = int(deltas["automation_outcome"].eq("fallo").sum())
    counts["valid_failure_deltas_outside_block_4"] = int(
        deltas.loc[deltas["block"].ne(4), "automation_outcome"].eq("fallo").sum()
    )
    return deltas, counts


def participant_condition_means(
    deltas: pd.DataFrame,
    group_keys: list[str],
    outcome_column: str = "automation_outcome",
) -> pd.DataFrame:
    means = (
        deltas.groupby(group_keys + [outcome_column], as_index=False, dropna=False)
        .agg(mean_delta_trust=("delta_trust", "mean"), n_deltas=("delta_trust", "size"))
    )
    wide = means.pivot(
        index=group_keys,
        columns=outcome_column,
        values=["mean_delta_trust", "n_deltas"],
    )
    wide.columns = [f"{value}_{condition}" for value, condition in wide.columns]
    wide = wide.reset_index()
    for column in (
        "mean_delta_trust_acierto",
        "mean_delta_trust_fallo",
        "n_deltas_acierto",
        "n_deltas_fallo",
    ):
        if column not in wide:
            wide[column] = np.nan
    wide[["n_deltas_acierto", "n_deltas_fallo"]] = wide[
        ["n_deltas_acierto", "n_deltas_fallo"]
    ].fillna(0).astype(int)
    wide["difference_acierto_minus_fallo"] = (
        wide["mean_delta_trust_acierto"] - wide["mean_delta_trust_fallo"]
    )
    wide["pairable"] = wide["difference_acierto_minus_fallo"].notna()
    return wide


def describe_values(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    clean = pd.Series(values, dtype="float64").dropna()
    if clean.empty:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan}
    q1, q3 = clean.quantile([0.25, 0.75])
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if clean.size > 1 else np.nan,
        "median": float(clean.median()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def test_paired(participant_means: pd.DataFrame, label: str) -> dict[str, object]:
    paired = participant_means.loc[participant_means["pairable"]].copy()
    success = paired["mean_delta_trust_acierto"].astype(float)
    failure = paired["mean_delta_trust_fallo"].astype(float)
    difference = success - failure
    n = len(difference)

    row: dict[str, object] = {
        "analysis": label,
        "n_participants_pairable": n,
        "n_participants_without_both_conditions": int(len(participant_means) - n),
        "mean_delta_acierto": float(success.mean()) if n else np.nan,
        "sd_delta_acierto": float(success.std(ddof=1)) if n > 1 else np.nan,
        "median_delta_acierto": float(success.median()) if n else np.nan,
        "mean_delta_fallo": float(failure.mean()) if n else np.nan,
        "sd_delta_fallo": float(failure.std(ddof=1)) if n > 1 else np.nan,
        "median_delta_fallo": float(failure.median()) if n else np.nan,
        "mean_difference_acierto_minus_fallo": float(difference.mean()) if n else np.nan,
        "sd_difference": float(difference.std(ddof=1)) if n > 1 else np.nan,
        "median_difference": float(difference.median()) if n else np.nan,
        "difference_q1": float(difference.quantile(0.25)) if n else np.nan,
        "difference_q3": float(difference.quantile(0.75)) if n else np.nan,
        "difference_iqr": float(difference.quantile(0.75) - difference.quantile(0.25)) if n else np.nan,
        "shapiro_w": np.nan,
        "shapiro_p": np.nan,
        "t": np.nan,
        "df": max(n - 1, 0),
        "t_p_two_sided": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "cohens_dz": np.nan,
        "wilcoxon_w": np.nan,
        "wilcoxon_p_two_sided": np.nan,
        "matched_pairs_rank_biserial": np.nan,
        "recommended_primary_test": "No se puede decidir: menos de 3 pares válidos",
    }

    if n >= 3:
        shapiro = stats.shapiro(difference)
        row["shapiro_w"] = float(shapiro.statistic)
        row["shapiro_p"] = float(shapiro.pvalue)
        row["recommended_primary_test"] = (
            "Wilcoxon (no paramétrico; Shapiro-Wilk p < 0,05)"
            if shapiro.pvalue < 0.05
            else "t de Student pareada (paramétrico; Shapiro-Wilk p ≥ 0,05)"
        )

    if n >= 2:
        if np.isclose(difference.std(ddof=1), 0):
            if np.isclose(difference.mean(), 0):
                t_value, p_t = 0.0, 1.0
            else:
                t_value, p_t = np.sign(difference.mean()) * np.inf, 0.0
            ci_low = ci_high = float(difference.mean())
            dz = np.nan if np.isclose(difference.mean(), 0) else np.sign(difference.mean()) * np.inf
        else:
            result = stats.ttest_rel(success, failure, alternative="two-sided")
            t_value, p_t = float(result.statistic), float(result.pvalue)
            sem = difference.std(ddof=1) / np.sqrt(n)
            critical = stats.t.ppf(0.975, df=n - 1)
            ci_low = float(difference.mean() - critical * sem)
            ci_high = float(difference.mean() + critical * sem)
            dz = float(difference.mean() / difference.std(ddof=1))
        row.update({"t": float(t_value), "t_p_two_sided": float(p_t), "ci95_low": ci_low, "ci95_high": ci_high, "cohens_dz": dz})

    if n >= 1 and not np.allclose(difference, 0):
        wilcoxon = stats.wilcoxon(success, failure, alternative="two-sided", zero_method="wilcox", method="auto")
        nonzero = difference.loc[~np.isclose(difference, 0)]
        ranks = rankdata(np.abs(nonzero), method="average")
        w_plus = float(ranks[nonzero.to_numpy() > 0].sum())
        w_minus = float(ranks[nonzero.to_numpy() < 0].sum())
        denominator = w_plus + w_minus
        row["wilcoxon_w"] = float(wilcoxon.statistic)
        row["wilcoxon_p_two_sided"] = float(wilcoxon.pvalue)
        row["matched_pairs_rank_biserial"] = (w_plus - w_minus) / denominator if denominator else np.nan
    elif n >= 1:
        row["wilcoxon_w"] = 0.0
        row["wilcoxon_p_two_sided"] = 1.0
        row["matched_pairs_rank_biserial"] = 0.0

    return row


def overall_trial_descriptives(deltas: pd.DataFrame) -> pd.DataFrame:
    records = []
    for condition in ("acierto", "fallo"):
        values = deltas.loc[deltas["automation_outcome"].eq(condition), "delta_trust"]
        records.append({"condition": condition, **describe_values(values)})
    return pd.DataFrame(records)


def write_results_workbook(
    output_path: Path,
    counts: dict[str, int],
    deltas: pd.DataFrame,
    overall_desc: pd.DataFrame,
    overall: dict[str, object],
    participant_means: pd.DataFrame,
    exploratory: pd.DataFrame,
    exploratory_participants: pd.DataFrame,
) -> None:
    """Write all tabular analysis outputs to one formatted Excel workbook."""
    method = pd.DataFrame(
        [
            ("Fuente", "datos_combinados.xlsx · hoja TRIALS"),
            ("Variable de resultado", "automation_error: 0 = acierto de automatización; 1 = fallo"),
            ("Cambio de confianza", "trust_t − trust_(t−1), solo trials consecutivos dentro del mismo bloque"),
            ("Alineación del evento", "El delta se clasifica con automation_error_t"),
            ("Bloques incluidos", "2–5; se excluyen el bloque manual 1 y el bloque 6"),
            ("Filas fuente", counts["input_rows"]),
            ("Participantes con deltas válidos", counts["participants_with_valid_deltas"]),
            ("Deltas válidos", counts["valid_deltas"]),
            ("Deltas tras acierto", int(deltas["automation_outcome"].eq("acierto").sum())),
            ("Deltas tras fallo", counts["valid_failure_deltas"]),
            ("Participantes en prueba pareada", overall["n_participants_pairable"]),
            ("Media delta tras acierto por participante", overall["mean_delta_acierto"]),
            ("Media delta tras fallo por participante", overall["mean_delta_fallo"]),
            ("Diferencia media acierto − fallo", overall["mean_difference_acierto_minus_fallo"]),
            ("Shapiro–Wilk W", overall["shapiro_w"]),
            ("Shapiro–Wilk p", overall["shapiro_p"]),
            ("Prueba recomendada", overall["recommended_primary_test"]),
            ("t pareada", overall["t"]),
            ("t pareada p (dos colas)", overall["t_p_two_sided"]),
            ("IC 95 % diferencia inferior", overall["ci95_low"]),
            ("IC 95 % diferencia superior", overall["ci95_high"]),
            ("Cohen dz", overall["cohens_dz"]),
            ("Wilcoxon W", overall["wilcoxon_w"]),
            ("Wilcoxon p (dos colas)", overall["wilcoxon_p_two_sided"]),
            ("Rank-biserial emparejada", overall["matched_pairs_rank_biserial"]),
            ("Interpretación", "La confianza descendió tras fallos y quedó prácticamente estable tras aciertos; todos los fallos con delta válido fueron en el bloque 4."),
        ],
        columns=["elemento", "resultado"],
    )
    quality = pd.DataFrame(
        [(key, value) for key, value in counts.items()],
        columns=["control", "n"],
    )
    sheets = {
        "informe": method,
        "contraste_global": pd.DataFrame([overall]),
        "descriptivos_delta": overall_desc,
        "participantes": participant_means,
        "fase_bloque": exploratory,
        "participantes_fase_bloque": exploratory_participants,
        "deltas_trial": deltas,
        "control_calidad": quality,
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            if worksheet.max_row > 1 and worksheet.max_column > 0:
                worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.columns:
                column_letter = cells[0].column_letter
                max_length = max(
                    (len(str(cell.value)) for cell in cells if cell.value is not None),
                    default=0,
                )
                worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 48)
            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                font.color = "FFFFFF"
                cell.font = font
                cell.fill = PatternFill(fill_type="solid", fgColor="1F4E78")
                alignment = copy(cell.alignment)
                alignment.wrap_text = True
                alignment.vertical = "center"
                cell.alignment = alignment
            worksheet.row_dimensions[1].height = 30



def exploratory_stats(deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    participant_frames: list[pd.DataFrame] = []
    for phase, block in sorted(
        deltas[["phase", "block"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda item: (str(item[0]), int(item[1])),
    ):
        subset = deltas.loc[deltas["phase"].eq(phase) & deltas["block"].eq(block)]
        means = participant_condition_means(subset, ["participant"])
        means.insert(0, "block", block)
        means.insert(0, "phase", phase)
        participant_frames.append(means)
        record = test_paired(means, f"{phase}_B{int(block)}")
        record["phase"] = phase
        record["block"] = int(block)
        for condition in ("acierto", "fallo"):
            values = subset.loc[subset["automation_outcome"].eq(condition), "delta_trust"]
            summary = describe_values(values)
            record.update({f"trial_{condition}_{key}": value for key, value in summary.items()})
        records.append(record)
    participants = pd.concat(participant_frames, ignore_index=True) if participant_frames else pd.DataFrame()
    return pd.DataFrame(records), participants


def save_figures(
    paired: pd.DataFrame,
    overall: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    success = paired["mean_delta_trust_acierto"].to_numpy(dtype=float)
    failure = paired["mean_delta_trust_fallo"].to_numpy(dtype=float)
    x = np.array([0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for left, right in zip(success, failure):
        ax.plot(x, [left, right], color="#8a929e", alpha=0.62, linewidth=1.2, zorder=1)
    ax.scatter(np.zeros_like(success), success, color="#2166ac", s=30, label="Tras acierto", zorder=2)
    ax.scatter(np.ones_like(failure), failure, color="#b35806", s=30, label="Tras fallo", zorder=2)
    ax.set_xticks(x, ["Acierto", "Fallo"])
    ax.set_ylabel("Cambio medio de confianza (puntos del slider)")
    ax.set_title("Cambio por participante")
    ax.axhline(0, color="#4b5563", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", color="#d8dce2", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    box = ax.boxplot([success, failure], positions=x, widths=0.52, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ["#9ecae1", "#fdae6b"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#4b5563")
    rng = np.random.default_rng(2026)
    for pos, values, color in zip(x, [success, failure], ["#2166ac", "#b35806"]):
        ax.scatter(pos + rng.uniform(-0.08, 0.08, size=len(values)), values, color=color, s=20, alpha=0.7, zorder=3)
    ax.set_xticks(x, ["Acierto", "Fallo"])
    ax.set_ylabel("Cambio medio por participante")
    ax.set_title("Distribución de medias")
    ax.axhline(0, color="#4b5563", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", color="#d8dce2", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"H2 · automatización · n = {len(paired)} participantes", y=1.02)
    fig.tight_layout()
    paired_path = output_dir / "h2_paired_change.png"
    fig.savefig(paired_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    differences = paired["difference_acierto_minus_fallo"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].hist(differences, bins="auto", color="#9ecae1", edgecolor="#4b5563")
    axes[0].axvline(0, color="#4b5563", linewidth=0.9)
    axes[0].set_xlabel("Diferencia media por participante (acierto − fallo)")
    axes[0].set_ylabel("Frecuencia (participantes)")
    axes[0].set_title("Histograma de diferencias")
    axes[0].grid(axis="y", color="#d8dce2", linewidth=0.7)

    (osm, osr), (slope, intercept, _) = stats.probplot(differences, dist="norm")
    axes[1].scatter(osm, osr, color="#2166ac", s=28)
    line_x = np.asarray([min(osm), max(osm)])
    axes[1].plot(line_x, intercept + slope * line_x, color="#b35806", linewidth=1.2)
    axes[1].set_xlabel("Cuantiles teóricos normales")
    axes[1].set_ylabel("Diferencia observada")
    axes[1].set_title("Gráfico Q–Q")
    axes[1].grid(color="#d8dce2", linewidth=0.7)
    shapiro_p_label = format_p(overall.get("shapiro_p")).removeprefix("= ")
    fig.suptitle(f"Normalidad de las diferencias · Shapiro–Wilk p = {shapiro_p_label}", y=1.02)
    fig.tight_layout()
    diagnostic_path = output_dir / "h2_normality_diagnostics.png"
    fig.savefig(diagnostic_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paired_path, diagnostic_path


def format_number(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def format_p(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    return "< 0,001" if value < 0.001 else f"= {value:.3f}".replace(".", ",")


def write_report(
    report_path: Path,
    input_path: Path,
    counts: dict[str, int],
    trial_desc: pd.DataFrame,
    overall: dict[str, object],
    exploratory: pd.DataFrame,
    paired_figure: Path,
    diagnostic_figure: Path,
) -> None:
    trial_rows = []
    for _, row in trial_desc.iterrows():
        trial_rows.append(
            f"| {row['condition'].capitalize()} | {int(row['n'])} | {format_number(row['mean'])} | "
            f"{format_number(row['sd'])} | {format_number(row['median'])} | "
            f"{format_number(row['q1'])} | {format_number(row['q3'])} | {format_number(row['iqr'])} |"
        )

    source_label = str(input_path)
    try:
        source_label = str(input_path.resolve().relative_to(SCRIPT_DIR.parent))
    except ValueError:
        pass
    overall_text = (
        f"Se incluyeron {int(overall['n_participants_pairable'])} participantes con ambas condiciones. "
        f"La diferencia media por participante (acierto − fallo) fue {format_number(overall['mean_difference_acierto_minus_fallo'])} "
        f"(DE = {format_number(overall['sd_difference'])}; IC 95 % [{format_number(overall['ci95_low'])}, "
        f"{format_number(overall['ci95_high'])}]). La t pareada dio t({int(overall['df'])}) = "
        f"{format_number(overall['t'])}, p {format_p(overall['t_p_two_sided'])}, d<sub>z</sub> = "
        f"{format_number(overall['cohens_dz'])}; Wilcoxon W = {format_number(overall['wilcoxon_w'])}, "
        f"p {format_p(overall['wilcoxon_p_two_sided'])}, r<sub>rb</sub> = "
        f"{format_number(overall['matched_pairs_rank_biserial'])}."
    )
    if pd.notna(overall["shapiro_p"]):
        normality = (
            f"Shapiro–Wilk: W = {format_number(overall['shapiro_w'])}, p {format_p(overall['shapiro_p'])}. "
            f"Según el criterio de normalidad (α = 0,05), se recomienda como prueba principal: "
            f"**{overall['recommended_primary_test']}**."
        )
    else:
        normality = "Shapiro–Wilk no se pudo calcular porque hay menos de tres diferencias pareadas."

    exploratory_lines = [
        "| Fase | Bloque | Pares | Δ tras acierto (media) | Δ tras fallo (media) | Diferencia media | t | p t | W | p Wilcoxon |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in exploratory.iterrows():
        exploratory_lines.append(
            f"| {row['phase']} | {int(row['block'])} | {int(row['n_participants_pairable'])} | "
            f"{format_number(row['mean_delta_acierto'])} | {format_number(row['mean_delta_fallo'])} | "
            f"{format_number(row['mean_difference_acierto_minus_fallo'])} | {format_number(row['t'])} | "
            f"{format_p(row['t_p_two_sided'])} | {format_number(row['wilcoxon_w'])} | "
            f"{format_p(row['wilcoxon_p_two_sided'])} |"
        )

    report = f"""# H2 — Cambio de confianza tras aciertos y fallos de la automatización

## Datos y definición

Fuente: `{source_label}` (hoja `TRIALS`, {counts['input_rows']} filas; {counts['participants_with_valid_deltas']} participantes con deltas válidos). Se usó el libro consolidado existente; `ATC_EXTRACTOR/unir_excel.py` es el proceso del repo que apila los Excel individuales. No se encontró en el repo otra convención o análisis previo de H2 ni material CRID o plantilla de informe; para recomendar prueba se aplica la regla estándar de Shapiro–Wilk con α = 0,05. La variable principal es `automation_error`: 0 = acierto de la automatización; 1 = fallo. `correct` describe el desempeño del participante y no define H2; no se hizo análisis de sensibilidad con `correct` al no encontrarse una convención alternativa que genere ambigüedad.

Se excluyeron el bloque manual 1 y los bloques fuera del intervalo automatizado 2–5 (incluido el bloque 6 que aparece en algunos archivos, porque no forma parte de los cuatro bloques descritos para el experimento). Se descartaron pares de ensayos con `answer = no_response`, `trust` o `automation_error` ausente, o con números de trial no consecutivos. Se requieren dos respuestas de confianza válidas para calcular cada cambio.

**Convención temporal:** para el trial consecutivo t, `delta_trust = trust_t − trust_(t−1)` y el cambio se clasifica con `automation_error_t`, el resultado del trial t, después del cual se registró `trust_t`. El emparejamiento se hace dentro de cada participante, sesión, fase y bloque; nunca cruza límites de bloque.

En los bloques automatizados había {counts['automated_rows']} filas: {counts['no_response_rows_in_automated_blocks']} con `no_response`, {counts['rows_missing_trust_in_automated_blocks']} sin confianza y {counts['rows_missing_automation_error_in_automated_blocks']} sin `automation_error`. Se conservaron {counts['valid_deltas']} deltas válidos, incluidos {counts['valid_failure_deltas']} tras fallos. Para la prueba pareada global se incluyeron {int(overall['n_participants_pairable'])} participantes; {int(overall['n_participants_without_both_conditions'])} no tenían ambas condiciones y se excluyeron de la comparación pareada.

## Descriptivos de cambios por ensayo

| Resultado de automatización | N deltas | Media | DE | Mediana | Q1 | Q3 | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(trial_rows)}

## Comparación pareada por participante

Las medias se calcularon por participante y condición agrupando fases y bloques automatizados. La diferencia se definió como media tras acierto menos media tras fallo.

| Medida | Resultado |
|---|---:|
| N participantes pareables | {int(overall['n_participants_pairable'])} |
| Media del delta tras acierto (DE) | {format_number(overall['mean_delta_acierto'])} ({format_number(overall['sd_delta_acierto'])}) |
| Media del delta tras fallo (DE) | {format_number(overall['mean_delta_fallo'])} ({format_number(overall['sd_delta_fallo'])}) |
| Diferencia por participante (media; DE) | {format_number(overall['mean_difference_acierto_minus_fallo'])}; {format_number(overall['sd_difference'])} |
| Diferencia por participante (mediana; IQR) | {format_number(overall['median_difference'])}; {format_number(overall['difference_iqr'])} |
| Shapiro–Wilk de las diferencias | W = {format_number(overall['shapiro_w'])}; p {format_p(overall['shapiro_p'])} |
| t de Student pareada | t({int(overall['df'])}) = {format_number(overall['t'])}; p {format_p(overall['t_p_two_sided'])} |
| IC 95 % de la diferencia | [{format_number(overall['ci95_low'])}, {format_number(overall['ci95_high'])}] |
| Cohen d<sub>z</sub> | {format_number(overall['cohens_dz'])} |
| Wilcoxon de rangos signados | W = {format_number(overall['wilcoxon_w'])}; p {format_p(overall['wilcoxon_p_two_sided'])} |
| Correlación rank-biserial emparejada | {format_number(overall['matched_pairs_rank_biserial'])} |
| Prueba recomendada por Shapiro–Wilk | {overall['recommended_primary_test']} |

**Interpretación:** {overall_text} {normality} En promedio, la confianza descendió tras fallos, mientras que tras aciertos permaneció prácticamente estable; por tanto, los datos respaldan que el cambio es más positivo tras un acierto que tras un fallo, pero no muestran un aumento claro tras aciertos. Los {counts['valid_failure_deltas']} deltas válidos tras fallos ocurrieron en el bloque 4 ({counts['valid_failure_deltas_outside_block_4']} en otros bloques); como los aciertos también proceden de los bloques 2, 3 y 5, el contraste global está parcialmente confundido con el bloque. La comparación exploratoria dentro del bloque 4 también muestra mayor cambio tras acierto que tras fallo.

## Exploratorio por fase y bloque

Las comparaciones por celda son exploratorias; los valores p no se ajustaron por comparaciones múltiples.

{chr(10).join(exploratory_lines)}

## Figuras

![Medias pareadas y distribución por participante]({paired_figure.name})

![Histograma y gráfico Q–Q de las diferencias]({diagnostic_figure.name})

## Archivos de apoyo

- `h2_results.xlsx`: resultados tabulares en las hojas `informe`, `contraste_global`, `descriptivos_delta`, `participantes`, `fase_bloque`, `participantes_fase_bloque`, `deltas_trial` y `control_calidad`.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_trials(args.input_file)
    deltas, counts = build_trial_deltas(data)
    if deltas.empty:
        raise ValueError("No quedan deltas válidos. Revisa TRIALS y los criterios de exclusión.")

    participant_means = participant_condition_means(deltas, ["participant"])
    paired = participant_means.loc[participant_means["pairable"]].copy()
    overall_desc = overall_trial_descriptives(deltas)
    overall = test_paired(participant_means, "global")
    exploratory, exploratory_participants = exploratory_stats(deltas)

    paired_path, diagnostic_path = save_figures(paired, overall, args.output_dir)
    workbook_path = args.output_dir / "h2_results.xlsx"
    write_results_workbook(
        workbook_path,
        counts,
        deltas,
        overall_desc,
        overall,
        participant_means,
        exploratory,
        exploratory_participants,
    )
    # Remove the CSV exports produced by earlier script versions in this folder.
    for stale_csv in args.output_dir.glob("h2_*.csv"):
        stale_csv.unlink()
    write_report(
        args.output_dir / "h2_report.md",
        args.input_file,
        counts,
        overall_desc,
        overall,
        exploratory,
        paired_path,
        diagnostic_path,
    )

    print("=== H2: Cambio de confianza tras el resultado de la automatización ===")
    print(f"Entrada: {args.input_file}")
    print(f"Deltas válidos: {len(deltas)}")
    print(f"Participantes con pares válidos: {int(overall['n_participants_pairable'])}")
    print(f"Δ media acierto − fallo: {format_number(overall['mean_difference_acierto_minus_fallo'])}")
    print(f"t({int(overall['df'])}) = {format_number(overall['t'])}, p {format_p(overall['t_p_two_sided'])}")
    print(f"Shapiro p {format_p(overall['shapiro_p'])}; recomendado: {overall['recommended_primary_test']}")
    print(f"Libro Excel: {workbook_path}")
    print(f"Resultados: {args.output_dir}")


if __name__ == "__main__":
    main()
