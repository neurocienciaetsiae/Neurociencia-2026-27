"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: session_recorder.py
Autora: Ana Ribote Gutierrez
Version: 1.4
Fecha: 06/06/2026

Grabador de sesion online.

Estructura de carpetas de salida (NUEVO v1.4):
  session_folder/
    markers.csv                    <- evento experimental (raiz: nexo raw-processed)
    raw/
      eeg_raw.csv
      pupil.csv
      gaze2d.csv
    processed/
      eeg_preprocessed.csv
      pupil_preprocessed.csv       <- NUEVO: pupila con saneado streaming
      gaze2d_preprocessed.csv      <- NUEVO: gaze con saneado streaming
      blink_events.csv
      trial_summary.csv
      metrics_timeseries.csv
    session_timeline_summary.png   <- generado al cerrar

Saneado streaming de ET: invalida valores fuera de rango fisiologico
(<2mm o >8mm para pupila, fuera de [0,1] para gaze), NaN/inf y ceros,
y calcula la media binocular cuando ambos ojos son validos.
NO interpola huecos (eso necesita contexto temporal y se hace solo
en extract_trial_features). El usuario puede interpolar offline.

Cambios v1.4:
- Estructura raw/ y processed/.
- Anadidos write_pupil_preprocessed() y write_gaze2d_preprocessed().
- generate_final_report() actualizado para leer metricas desde processed/
  y markers desde la raiz.

Cambios v1.3:
- BUGFIX: write_trial_summary usaba getattr(trial_info, "t_baseline", 0.0)
  pero el campo en TrialInfo se llama t_baseline_start.
  Los campos t_start, t_end, correct y confidence tampoco existen en TrialInfo:
    * t_start -> no existe; se sustituye por t_baseline_start.
    * t_end   -> no existe; se escribe el timestamp del evento de respuesta.
    * correct -> no existe; se elimina (no hay feedback automatico en ATC).
    * confidence -> no existe como campo; en su lugar se usa confidence_high o
      confidence_low segun la condicion del trial.
  La firma de write_trial_summary ahora recibe timestamp_end para registrar
  el momento real del fin del trial.
- BUGFIX: se añade write_blink_events() para guardar cada parpadeo detectado
  con su timestamp, duracion y contexto experimental.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class RecorderContext:
    phase: str = "IDLE"
    block: int = 0
    trial: int = 0
    condition: str = ""
    current_marker: str = ""


class SessionRecorder:
    def __init__(
        self,
        base_folder: str | Path,
        participant: str,
        session: str,
        eeg_channel_names: list[str],
    ):
        self.base_folder = Path(base_folder)
        self.participant = (participant or "-").strip()
        self.session = (session or "-").strip()
        self.eeg_channel_names = eeg_channel_names

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.session_folder = (
            self.base_folder
            / self.participant
            / self.session
            / timestamp_str
        )
        self.session_folder.mkdir(parents=True, exist_ok=True)

        self._files: dict = {}
        self._writers: dict = {}

        self._metrics_t0: float | None = None
        self._closed = False
        self._report_generated = False

        self._open_all_files()

    def _open_csv(self, name: str, header: Iterable[str], subfolder: str = ""):
        if subfolder:
            folder = self.session_folder / subfolder
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
        else:
            path = self.session_folder / name
        f = open(path, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(list(header))
        # Clave del writer: subfolder/name para evitar colisiones futuras
        # si llegara a haber un mismo nombre en raw y processed.
        key = f"{subfolder}/{name}" if subfolder else name
        self._files[key] = f
        self._writers[key] = writer

    def _open_all_files(self):
        common_context = [
            "phase", "block", "trial", "condition", "current_marker",
        ]

        # ------------ RAW: senales tal como llegan de LSL ------------
        self._open_csv(
            "eeg_raw.csv",
            ["timestamp_lsl", *self.eeg_channel_names, *common_context],
            subfolder="raw",
        )
        self._open_csv(
            "gaze2d.csv",
            ["timestamp_lsl", "gaze_x", "gaze_y", *common_context],
            subfolder="raw",
        )
        self._open_csv(
            "pupil.csv",
            ["timestamp_lsl", "pupil_left", "pupil_right", *common_context],
            subfolder="raw",
        )

        # ------------ MARKERS: en la raiz como nexo raw-processed ----
        self._open_csv(
            "markers.csv",
            ["timestamp_lsl", "marker", "phase_after_event", "block", "trial", "condition"],
        )

        # ------------ PROCESSED: senales y derivados ya tratados -----
        self._open_csv(
            "eeg_preprocessed.csv",
            ["timestamp_lsl", *self.eeg_channel_names, *common_context],
            subfolder="processed",
        )
        # NUEVO v1.4: ET preprocesado (saneado streaming, sin interpolacion).
        self._open_csv(
            "pupil_preprocessed.csv",
            [
                "timestamp_lsl",
                "pupil_left_clean",   # NaN si invalido
                "pupil_right_clean",  # NaN si invalido
                "pupil_mean_clean",   # media binocular
                *common_context,
            ],
            subfolder="processed",
        )
        self._open_csv(
            "gaze2d_preprocessed.csv",
            [
                "timestamp_lsl",
                "gaze_x_clean",       # NaN si fuera de [0,1]
                "gaze_y_clean",       # NaN si fuera de [0,1]
                *common_context,
            ],
            subfolder="processed",
        )
        self._open_csv(
            "blink_events.csv",
            ["timestamp_start_lsl", "timestamp_end_lsl", "duration_s", "kind", *common_context],
            subfolder="processed",
        )
        self._open_csv(
            "trial_summary.csv",
            [
                "block", "trial", "condition",
                "t_baseline_start",
                "t_baseline_end",
                "t_response",
                "response",
                "confidence_value",
                "trust_value", "workload_value", "fatigue_value",
            ],
            subfolder="processed",
        )
        self._open_csv(
            "metrics_timeseries.csv",
            [
                "timestamp_lsl", "elapsed_s",
                "trust_value", "workload_value", "fatigue_value",
                "phase", "block", "trial", "condition", "current_marker",
            ],
            subfolder="processed",
        )

    # ------------------------------------------------------------------
    # Contexto experimental
    # ------------------------------------------------------------------

    def _ctx_to_list(self, context: RecorderContext) -> list:
        return [
            context.phase,
            context.block,
            context.trial,
            context.condition,
            context.current_marker,
        ]

    # ------------------------------------------------------------------
    # Escritores de streams brutos
    # ------------------------------------------------------------------

    def write_eeg_raw(
        self,
        timestamp_lsl: float,
        values: list[float],
        context: RecorderContext,
    ):
        if self._closed:
            return
        n = min(len(values), len(self.eeg_channel_names))
        row_values = list(values[:n])
        if n < len(self.eeg_channel_names):
            row_values.extend([""] * (len(self.eeg_channel_names) - n))
        self._writers["raw/eeg_raw.csv"].writerow(
            [timestamp_lsl, *row_values, *self._ctx_to_list(context)]
        )

    def write_eeg_preprocessed(
        self,
        timestamp_lsl: float,
        values: list[float],
        context: RecorderContext,
    ):
        if self._closed:
            return
        n = min(len(values), len(self.eeg_channel_names))
        row_values = list(values[:n])
        if n < len(self.eeg_channel_names):
            row_values.extend([""] * (len(self.eeg_channel_names) - n))
        self._writers["processed/eeg_preprocessed.csv"].writerow(
            [timestamp_lsl, *row_values, *self._ctx_to_list(context)]
        )

    def write_gaze2d(
        self,
        timestamp_lsl: float,
        gaze_x: float,
        gaze_y: float,
        context: RecorderContext,
    ):
        if self._closed:
            return
        self._writers["raw/gaze2d.csv"].writerow(
            [timestamp_lsl, gaze_x, gaze_y, *self._ctx_to_list(context)]
        )

    def write_pupil(
        self,
        timestamp_lsl: float,
        pupil_left: float,
        pupil_right: float,
        context: RecorderContext,
    ):
        if self._closed:
            return
        self._writers["raw/pupil.csv"].writerow(
            [timestamp_lsl, pupil_left, pupil_right, *self._ctx_to_list(context)]
        )

    # NUEVO v1.4: escritores para ET preprocesado streaming.
    def write_pupil_preprocessed(
        self,
        timestamp_lsl: float,
        pupil_left_clean: float,
        pupil_right_clean: float,
        pupil_mean_clean: float,
        context: RecorderContext,
    ):
        if self._closed:
            return
        self._writers["processed/pupil_preprocessed.csv"].writerow(
            [
                timestamp_lsl,
                pupil_left_clean,
                pupil_right_clean,
                pupil_mean_clean,
                *self._ctx_to_list(context),
            ]
        )

    def write_gaze2d_preprocessed(
        self,
        timestamp_lsl: float,
        gaze_x_clean: float,
        gaze_y_clean: float,
        context: RecorderContext,
    ):
        if self._closed:
            return
        self._writers["processed/gaze2d_preprocessed.csv"].writerow(
            [
                timestamp_lsl,
                gaze_x_clean,
                gaze_y_clean,
                *self._ctx_to_list(context),
            ]
        )

    def write_marker(
        self,
        timestamp_lsl: float,
        marker: str,
        context: RecorderContext,
    ):
        if self._closed:
            return
        self._writers["markers.csv"].writerow([
            timestamp_lsl, marker, context.phase,
            context.block, context.trial, context.condition,
        ])

    def write_blink_event(
        self,
        timestamp_start_lsl: float,
        timestamp_end_lsl: float,
        duration_s: float,
        kind: str,
        context: RecorderContext,
    ):
        """
        Graba un evento de parpadeo individual con timestamp y contexto experimental.
        Llamar desde processor cada vez que ETPipeline detecte un nuevo blink.
        """
        if self._closed:
            return
        self._writers["processed/blink_events.csv"].writerow([
            timestamp_start_lsl,
            timestamp_end_lsl,
            duration_s,
            kind,
            *self._ctx_to_list(context),
        ])

    def write_trial_summary(
        self,
        trial_info,
        trust_value: float,
        workload_value: float,
        fatigue_value: float,
        timestamp_end: float = 0.0,
    ):
        """
        Graba un resumen de trial al final del mismo.

        Parametros
        ----------
        trial_info : TrialInfo
            Informacion del trial (de markers.py).
        trust_value, workload_value, fatigue_value : float
        timestamp_end : float
            Timestamp LSL del momento de respuesta (CONF_H / CONF_L).
        """
        if self._closed:
            return

        # CORREGIDO v1.3: usar los nombres reales de TrialInfo
        t_baseline_start = getattr(trial_info, "t_baseline_start", 0.0)
        t_baseline_end   = getattr(trial_info, "t_baseline_end", 0.0)
        condition        = getattr(trial_info, "condition", "")

        # Valor de confianza: el campo activo segun la condicion
        if condition == "HIGH":
            confidence_value = getattr(trial_info, "confidence_high", -1.0)
        elif condition == "LOW":
            confidence_value = getattr(trial_info, "confidence_low", -1.0)
        else:
            confidence_value = -1.0

        row = [
            getattr(trial_info, "block", 0),
            getattr(trial_info, "trial", 0),
            condition,
            t_baseline_start,
            t_baseline_end,
            timestamp_end,
            getattr(trial_info, "response", ""),
            confidence_value,
            trust_value,
            workload_value,
            fatigue_value,
        ]
        self._writers["processed/trial_summary.csv"].writerow(row)

    def write_metrics(
        self,
        timestamp_lsl: float,
        trust_value: float | None,
        workload_value: float | None,
        fatigue_value: float | None,
        context: RecorderContext,
    ):
        if self._closed:
            return

        timestamp_lsl = float(timestamp_lsl)

        if self._metrics_t0 is None:
            self._metrics_t0 = timestamp_lsl

        elapsed_s = timestamp_lsl - self._metrics_t0

        row = [
            timestamp_lsl,
            elapsed_s,
            "" if trust_value is None else float(trust_value),
            "" if workload_value is None else float(workload_value),
            "" if fatigue_value is None else float(fatigue_value),
            context.phase,
            context.block,
            context.trial,
            context.condition,
            context.current_marker,
        ]
        self._writers["processed/metrics_timeseries.csv"].writerow(row)

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def flush(self):
        for f in self._files.values():
            try:
                f.flush()
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        for f in self._files.values():
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        self._closed = True

    def generate_final_report(
        self,
        active_models: list[str] | None = None,
        max_marker_labels: int = 60,
    ) -> Path | None:
        if self._report_generated:
            path = self.session_folder / "session_timeline_summary.png"
            return path if path.exists() else None

        self.flush()

        # NUEVO v1.4: rutas actualizadas a la estructura raw/processed.
        metrics_path = self.session_folder / "processed" / "metrics_timeseries.csv"
        markers_path = self.session_folder / "markers.csv"
        output_path  = self.session_folder / "session_timeline_summary.png"

        if not metrics_path.exists():
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[SessionRecorder] No se pudo importar matplotlib: {exc}")
            return None

        metrics_rows = self._read_csv_dicts(metrics_path)
        marker_rows  = self._read_csv_dicts(markers_path)

        if not metrics_rows:
            return None

        active_models = active_models or ["Trust", "Workload", "Fatigue"]
        metric_specs = []
        if "Trust"    in active_models:
            metric_specs.append(("trust_value",    "Trust"))
        if "Workload" in active_models:
            metric_specs.append(("workload_value", "Workload"))
        if "Fatigue"  in active_models:
            metric_specs.append(("fatigue_value",  "Fatigue"))

        clean_specs = [
            (col, label) for col, label in metric_specs
            if any(self._to_float(r.get(col)) is not None for r in metrics_rows)
        ]

        if not clean_specs:
            return None

        t = []
        for row in metrics_rows:
            elapsed = self._to_float(row.get("elapsed_s"))
            if elapsed is not None:
                t.append(elapsed)
            else:
                ts = self._to_float(row.get("timestamp_lsl"))
                t.append(0.0 if ts is None else
                         (ts - self._metrics_t0 if self._metrics_t0 else ts))

        n_plots = len(clean_specs)
        fig, axes = plt.subplots(n_plots, 1, figsize=(18, 3.0 * n_plots + 2.2), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, clean_specs):
            y = [self._to_float(r.get(col)) for r in metrics_rows]
            t_valid = [ti for ti, yi in zip(t, y) if yi is not None]
            y_valid = [yi for yi in y if yi is not None]
            ax.plot(t_valid, y_valid, linewidth=1.6)
            ax.set_ylabel(label)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)

        t0_abs = self._to_float(metrics_rows[0].get("timestamp_lsl")) if t else None

        for label_count, marker_row in enumerate(marker_rows):
            if label_count >= max_marker_labels:
                break
            marker_ts = self._to_float(marker_row.get("timestamp_lsl"))
            if marker_ts is None:
                continue
            marker_x = (marker_ts - t0_abs) if t0_abs is not None else marker_ts
            marker_name = marker_row.get("marker", "")
            block = marker_row.get("block", "")
            trial = marker_row.get("trial", "")
            label = (f"T{trial}_B{block}" if block not in ("", "0") and trial not in ("", "0")
                     else self._short_marker_label(marker_name))
            for ax in axes:
                ax.axvline(marker_x, linewidth=0.7, alpha=0.30)
            axes[0].text(marker_x, 102, label, rotation=90, fontsize=7,
                         ha="center", va="bottom")

        axes[-1].set_xlabel("Elapsed time (s)")
        summary_lines = self._build_summary_lines(metrics_rows, clean_specs)
        fig.suptitle(
            f"ATCo Trust Project — participant {self.participant}, session {self.session}",
            fontsize=14, fontweight="bold",
        )
        fig.text(0.01, 0.02, "\n".join(summary_lines), ha="left", va="bottom",
                 fontsize=10, family="monospace")
        fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.20, hspace=0.25)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

        self._report_generated = True
        print(f"[SessionRecorder] Figura resumen: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _read_csv_dicts(self, path: Path) -> list[dict]:
        rows = []
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append(row)
        except Exception as exc:
            print(f"[SessionRecorder] Error leyendo {path}: {exc}")
        return rows

    def _to_float(self, value) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    def _short_marker_label(self, marker: str, max_len: int = 24) -> str:
        marker = str(marker or "")
        return marker if len(marker) <= max_len else marker[:max_len - 3] + "..."

    def _build_summary_lines(
        self, metrics_rows: list[dict], metric_specs: list[tuple[str, str]],
    ) -> list[str]:
        lines = ["Summary of relevant points:"]
        for col, label in metric_specs:
            values = [
                (idx, v) for idx, row in enumerate(metrics_rows)
                if (v := self._to_float(row.get(col))) is not None
            ]
            if not values:
                continue
            idx_max, vmax = max(values, key=lambda x: x[1])
            row_max = metrics_rows[idx_max]
            lines.append(
                f"- Max {label:<8}: {vmax:6.2f} | "
                f"t={row_max.get('elapsed_s', '')} s | "
                f"phase={row_max.get('phase', '')} | "
                f"block={row_max.get('block', '')} | "
                f"trial={row_max.get('trial', '')}"
            )
            if label == "Trust":
                idx_min, vmin = min(values, key=lambda x: x[1])
                row_min = metrics_rows[idx_min]
                lines.append(
                    f"- Min {label:<8}: {vmin:6.2f} | "
                    f"t={row_min.get('elapsed_s', '')} s | "
                    f"phase={row_min.get('phase', '')} | "
                    f"block={row_min.get('block', '')} | "
                    f"trial={row_min.get('trial', '')}"
                )
        return lines
