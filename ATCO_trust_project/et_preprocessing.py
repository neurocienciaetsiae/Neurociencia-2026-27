"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: et_preprocessing.py
Autora: Ana Ribote Gutierrez
Version: 1.9
Fecha: 06/06/2026

Pipeline online de eye-tracking.

Cambios v1.9:
- Normalizacion pupilar segun Mathot et al. (2018): tres modos disponibles
  ("zscore", "percent", "subtract"). Por defecto "zscore" para permitir
  comparacion entre sujetos con pupilas basales distintas.
- Baseline pupilar ahora guarda tanto la tendencia central (mean o mediana)
  como la dispersion (std o MAD) segun config.baseline_robust_stats.
- pupil_norm_mean en extract_trial_features usa el modo configurado.
- Mantenida compatibilidad: pupil_norm_mean sigue siendo un escalar.

Cambios v1.8:
- Deteccion de parpadeos restringida a NaN simultaneos en ambos ojos.
- Un parpadeo valido requiere entre 2 y 15 muestras consecutivas NaN simultaneas.
- Las perdidas largas de pupila (> 15 muestras) no se cuentan como blink.

Cambios v1.7:
- Separacion entre ruta rapida y ruta lenta:
  1. Ruta rapida de parpadeos:
     - get_blink_line_fast()
     - get_fast_blink_metrics()
     - usa solo mascara cruda de invalidez pupilar;
     - no interpola;
     - no calcula gaze;
     - no limpia toda la pupila;
     - pensada para refresco de UI cada 80-100 ms.
  2. Ruta lenta de preprocesado:
     - get_slow_metrics_cached()
     - limpia pupila;
     - interpola huecos cortos;
     - calcula pupil recent;
     - calcula gaze stability;
     - pensada para refresco cada 1-2 s.
  3. get_recent_metrics_cached() combina ambas rutas:
     - blinks rapidos;
     - pupila y gaze lentos.

Este modulo preprocesa la senal de eye-tracking:
- limpia pupila izquierda y derecha;
- invalida NaN, inf, ceros y valores fuera de rango fisiologico;
- interpola huecos cortos;
- calcula pupila media;
- calcula baseline local por fixation cross;
- detecta blinks;
- calcula features de pupila, blinks y estabilidad de mirada.

No preprocesa EEG. El EEG se procesa en eeg_preprocessing.py.
"""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BlinkEvent:
    start_time: float
    end_time: float
    duration_s: float
    kind: str = "blink"


class ETPipelineConfig:
    """
    Config flexible.
    Acepta claves extra para no romper si ET_PIPELINE contiene parametros
    antiguos o nuevos.
    """

    def __init__(
        self,
        fs: float = 50.0,
        max_buffer_seconds: float = 120.0,
        pupil_min_mm: float = 2.0,
        pupil_max_mm: float = 5.5,
        interpolation_max_gap_s: float = 0.35,
        default_blink_min_duration_s: float = 0.06,
        default_blink_max_duration_s: float = 0.80,
        artifact_min_duration_s: float = 1.20,
        gaze_center_x: float = 0.5,
        gaze_center_y: float = 0.5,
        gaze_stability_scale: float = 0.25,
        fast_blink_window_s: float = 10.0,
        fast_blink_cache_interval_s: float = 0.08,
        slow_metrics_cache_interval_s: float = 2.0,
        blink_min_consecutive_nan: int = 2,
        blink_max_consecutive_nan: int = 15,
        pupil_norm_mode: str = "zscore",
        baseline_robust_stats: bool = True,
        **kwargs: Any,
    ):
        self.fs = float(fs)
        self.max_buffer_seconds = float(max_buffer_seconds)

        self.pupil_min_mm = float(pupil_min_mm)
        self.pupil_max_mm = float(pupil_max_mm)
        self.interpolation_max_gap_s = float(interpolation_max_gap_s)

        self.default_blink_min_duration_s = float(default_blink_min_duration_s)
        self.default_blink_max_duration_s = float(default_blink_max_duration_s)
        self.artifact_min_duration_s = float(artifact_min_duration_s)

        self.gaze_center_x = float(gaze_center_x)
        self.gaze_center_y = float(gaze_center_y)
        self.gaze_stability_scale = float(gaze_stability_scale)

        self.fast_blink_window_s = float(fast_blink_window_s)
        self.fast_blink_cache_interval_s = float(fast_blink_cache_interval_s)
        self.slow_metrics_cache_interval_s = float(slow_metrics_cache_interval_s)

        self.blink_min_consecutive_nan = int(blink_min_consecutive_nan)
        self.blink_max_consecutive_nan = int(blink_max_consecutive_nan)

        # Normalizacion pupilar (Mathot 2018):
        #   "zscore":   (pupil - mean) / std        <- recomendado entre sujetos
        #   "percent":  (pupil - mean) / mean * 100 <- variacion porcentual
        #   "subtract": pupil - mean                <- comportamiento legacy
        norm_mode = str(pupil_norm_mode).lower()
        if norm_mode not in ("zscore", "percent", "subtract"):
            norm_mode = "zscore"
        self.pupil_norm_mode = norm_mode

        # Si True, baseline usa mediana y MAD (robustas a outliers).
        self.baseline_robust_stats = bool(baseline_robust_stats)

        for key, value in kwargs.items():
            setattr(self, key, value)


class ETPipeline:
    def __init__(self, config: ETPipelineConfig | None = None):
        self.config = config or ETPipelineConfig()

        self.timestamps: list[float] = []
        self.pupil_l: list[float] = []
        self.pupil_r: list[float] = []
        self.gaze_x: list[float] = []
        self.gaze_y: list[float] = []

        # Baseline pupilar: tendencia central + dispersion
        # (Mathot 2018 recomienda z-score, que requiere ambos)
        self.baseline_pupil_mean: float | None = None
        self.baseline_pupil_std: float | None = None
        self.baseline_window: tuple[float, float] | None = None

        # Baseline VISUAL: solo para la UI, se resetea en cada fixation cross.
        # No afecta a los datos grabados ni a pupil_norm_mean.
        self.pupil_visual_baseline: float | None = None

        self.blink_zero_calibration: dict[str, float] | None = None

        self._max_samples = int(
            max(1000, self.config.max_buffer_seconds * self.config.fs * 2)
        )

        # Cache para ruta rapida de parpadeos.
        self._fast_blink_cache: dict[str, Any] | None = None
        self._fast_blink_cache_time: float = 0.0
        self._fast_blink_cache_last_timestamp: float | None = None
        self._fast_blink_cache_window: float | None = None

        # Cache para ruta lenta de pupila y gaze.
        self._slow_metrics_cache: dict[str, float] | None = None
        self._slow_metrics_cache_time: float = 0.0
        self._slow_metrics_cache_last_timestamp: float | None = None
        self._slow_metrics_cache_window: float | None = None

    # ------------------------------------------------------------------
    # Calibracion de parpadeos
    # ------------------------------------------------------------------

    def set_blink_zero_calibration(self, calibration: dict[str, float]):
        """
        Recibe la calibracion calculada en processor.py tras la fase de
        ojos abiertos. Actualiza los limites de duracion del blink detector.
        """
        if not calibration:
            return

        self.blink_zero_calibration = dict(calibration)

        try:
            print(
                f"[ETPipeline] Calibracion blink aplicada: "
                f"min={calibration.get('min_duration_s', '?'):.3f}s "
                f"max={calibration.get('max_duration_s', '?'):.3f}s "
                f"n={calibration.get('n_blinks_used', '?')}"
            )
        except Exception:
            print("[ETPipeline] Calibracion blink aplicada.")

    # ------------------------------------------------------------------
    # Baseline visual para la UI
    # ------------------------------------------------------------------

    def set_visual_baseline(self, t_start: float, t_end: float):
        """
        Calcula la media de pupila en el intervalo dado y la guarda como
        pupil_visual_baseline. La UI resta este valor a la grafica para
        que la linea arranque en 0 en cada fixation cross.

        Llamar desde processor._on_baseline_start con los timestamps del
        FIXATION_CROSS_START.
        """
        t, pl, pr, _, _ = self._slice_between(t_start, t_end)

        if len(t) == 0:
            # No hay datos en ese intervalo todavia. Usar media reciente.
            if len(self.timestamps) >= 3:
                t_now = float(self.timestamps[-1])
                t_prev = t_now - 1.0
                t2, pl2, pr2, _, _ = self._slice_between(t_prev, t_now)

                if len(t2) > 0:
                    avg = self._clean_pupil_average(t2, pl2, pr2)
                    valid = avg[np.isfinite(avg)]

                    if len(valid) >= 2:
                        self.pupil_visual_baseline = float(np.nanmedian(valid))
            return

        avg = self._clean_pupil_average(t, pl, pr)
        valid = avg[np.isfinite(avg)]

        if len(valid) >= 2:
            self.pupil_visual_baseline = float(np.nanmedian(valid))

    # ------------------------------------------------------------------
    # Baseline de datos para pupil_norm_mean
    # ------------------------------------------------------------------

    def set_baseline_window(self, t_start: float, t_end: float):
        """
        Define baseline local para normalizacion de pupil en los datos.
        Llamar con el intervalo del fixation cross del trial.

        Computa tendencia central (media o mediana) y dispersion (std o MAD)
        segun config.baseline_robust_stats. Ambos se usan para z-score en
        extract_trial_features (Mathot 2018).
        """
        self.baseline_window = (float(t_start), float(t_end))

        t, pl, pr, _, _ = self._slice_between(t_start, t_end)

        if len(t) == 0:
            return

        pupil_clean = self._clean_pupil_average(t, pl, pr)
        valid = pupil_clean[np.isfinite(pupil_clean)]

        if len(valid) < 3:
            return

        if self.config.baseline_robust_stats:
            # Mediana + MAD: robustas a outliers (parpadeos residuales)
            center = float(np.nanmedian(valid))
            mad = float(np.median(np.abs(valid - center)))
            spread = 1.4826 * mad  # MAD -> std equivalente
        else:
            center = float(np.nanmean(valid))
            spread = float(np.nanstd(valid))

        # Evitar division por cero en z-score
        if not np.isfinite(spread) or spread < 1e-6:
            spread = 1e-6

        self.baseline_pupil_mean = center
        self.baseline_pupil_std = spread

    # ------------------------------------------------------------------
    # Entrada de datos
    # ------------------------------------------------------------------

    def push_chunk(
        self,
        pupil_l: np.ndarray,
        pupil_r: np.ndarray,
        gaze_x: np.ndarray,
        gaze_y: np.ndarray,
        timestamps: list[float] | np.ndarray,
    ):
        """
        Inserta un chunk de datos de eye-tracking en el buffer.

        Esta version usa extend en bloque, no append muestra a muestra.
        Asume que los timestamps llegan ordenados, que es lo normal en LSL.
        """
        pupil_l = np.asarray(pupil_l, dtype=float)
        pupil_r = np.asarray(pupil_r, dtype=float)
        gaze_x = np.asarray(gaze_x, dtype=float)
        gaze_y = np.asarray(gaze_y, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)

        n = min(
            len(pupil_l),
            len(pupil_r),
            len(gaze_x),
            len(gaze_y),
            len(timestamps),
        )

        if n <= 0:
            return

        self.timestamps.extend(map(float, timestamps[:n]))
        self.pupil_l.extend(map(float, pupil_l[:n]))
        self.pupil_r.extend(map(float, pupil_r[:n]))
        self.gaze_x.extend(map(float, gaze_x[:n]))
        self.gaze_y.extend(map(float, gaze_y[:n]))

        self._trim_buffers()

    # ------------------------------------------------------------------
    # NUEVO v1.9: saneado streaming sample a sample.
    # ------------------------------------------------------------------

    def clean_sample_streaming(
        self,
        pupil_l: float,
        pupil_r: float,
        gaze_x: float,
        gaze_y: float,
    ) -> tuple[float, float, float, float, float]:
        """
        Saneado local de UNA muestra sin contexto temporal.

        Aplica los criterios locales del pipeline ET (sin interpolar):
          - NaN/inf/cero -> NaN
          - Pupila fuera de [pupil_min_mm, pupil_max_mm] -> NaN
          - Gaze fuera de [0, 1] -> NaN
          - Media binocular: si un ojo es valido, usa ese; si ambos validos,
            la media; si ninguno, NaN.

        Adecuado para grabar pupil_preprocessed.csv y gaze2d_preprocessed.csv
        streaming. La interpolacion de huecos cortos requiere ver muestras
        vecinas y solo se aplica en extract_trial_features.

        Devuelve (pupil_left_clean, pupil_right_clean, pupil_mean_clean,
                  gaze_x_clean, gaze_y_clean) con NaN donde haya invalidacion.
        """
        cfg = self.config

        def _clean_pupil(v: float) -> float:
            try:
                v = float(v)
            except (TypeError, ValueError):
                return float("nan")
            if not np.isfinite(v) or v == 0.0:
                return float("nan")
            if v < cfg.pupil_min_mm or v > cfg.pupil_max_mm:
                return float("nan")
            return v

        def _clean_gaze(v: float) -> float:
            try:
                v = float(v)
            except (TypeError, ValueError):
                return float("nan")
            if not np.isfinite(v) or v < 0.0 or v > 1.0:
                return float("nan")
            return v

        pl_c = _clean_pupil(pupil_l)
        pr_c = _clean_pupil(pupil_r)
        gx_c = _clean_gaze(gaze_x)
        gy_c = _clean_gaze(gaze_y)

        if np.isnan(pl_c) and np.isnan(pr_c):
            pm_c = float("nan")
        elif np.isnan(pl_c):
            pm_c = pr_c
        elif np.isnan(pr_c):
            pm_c = pl_c
        else:
            pm_c = (pl_c + pr_c) / 2.0

        return pl_c, pr_c, pm_c, gx_c, gy_c

    # ------------------------------------------------------------------
    # Ruta rapida: linea de parpadeos para UI
    # ------------------------------------------------------------------

    def get_blink_line_fast(
        self,
        last_n_seconds: float | None = None,
        min_interval_s: float | None = None,
    ) -> dict[str, Any]:
        """
        Devuelve una linea rapida de parpadeos para pintar en la UI.

        Esta funcion NO hace el preprocesado completo:
        - no interpola;
        - no calcula baseline;
        - no calcula gaze;
        - no calcula pupila limpia.

        Solo detecta NaN simultaneos en ambos ojos.
        Es la ruta adecuada para una grafica 0/1 de parpadeos con baja latencia.

        Devuelve:
        - t: timestamps relativos, empezando aproximadamente en -last_n_seconds;
        - blink_line: array 0/1, 1 solo cuando hay blink valido;
        - blink_active: 1 si el ultimo punto esta en blink;
        - blink_count_fast: numero de blinks en la ventana;
        - blink_rate_per_min_fast: tasa de blink en la ventana;
        - last_blink_time: timestamp absoluto del ultimo blink detectado.
        """
        if last_n_seconds is None:
            last_n_seconds = self.config.fast_blink_window_s

        if min_interval_s is None:
            min_interval_s = self.config.fast_blink_cache_interval_s

        if len(self.timestamps) == 0:
            return self._empty_fast_blink_line()

        now = time.perf_counter()
        last_timestamp = float(self.timestamps[-1])
        window = float(last_n_seconds)

        interval_valid = (
            self._fast_blink_cache is not None
            and (now - self._fast_blink_cache_time) < float(min_interval_s)
            and self._fast_blink_cache_window == window
        )

        if interval_valid:
            return self._copy_fast_blink_cache(self._fast_blink_cache)

        t_end = last_timestamp
        t_start = t_end - window

        t, pl, pr, _, _ = self._slice_between(t_start, t_end)

        if len(t) == 0:
            result = self._empty_fast_blink_line()
            self._store_fast_blink_cache(result, now, last_timestamp, window)
            return result

        raw_mask = self._raw_blink_mask(pl, pr)

        # Para pintar la serie se usan tiempos relativos. Esto evita que pyqtgraph
        # tenga que manejar timestamps absolutos muy grandes.
        t_rel = t - t[-1]

        events = self._detect_blink_events_fast_from_mask(t, raw_mask)

        blink_line = np.zeros_like(raw_mask, dtype=float)
        for event in events:
            event_mask = (t >= event.start_time) & (t <= event.end_time)
            blink_line[event_mask] = 1.0

        blink_count = len(events)

        duration_s = max(1e-6, float(t[-1] - t[0]))
        blink_rate = 60.0 * blink_count / duration_s

        blink_active = bool(blink_line[-1]) if len(blink_line) > 0 else False
        last_blink_time = events[-1].end_time if events else None

        result = {
            "t": t_rel,
            "blink_line": blink_line,
            "blink_active": int(blink_active),
            "blink_count_fast": int(blink_count),
            "blink_rate_per_min_fast": float(blink_rate),
            "last_blink_time": last_blink_time,
            "pupil_visual_baseline": self.pupil_visual_baseline,
        }

        self._store_fast_blink_cache(result, now, last_timestamp, window)
        return result

    def get_fast_blink_metrics(
        self,
        last_n_seconds: float | None = None,
        min_interval_s: float | None = None,
    ) -> dict[str, float]:
        """
        Devuelve solo escalares rapidos de blink, sin arrays para grafica.
        Util para labels, tarjetas o modelos que necesiten blink rapido.
        """
        line = self.get_blink_line_fast(
            last_n_seconds=last_n_seconds,
            min_interval_s=min_interval_s,
        )

        return {
            "blink_count_total": int(line["blink_count_fast"]),
            "blink_rate_per_min": float(line["blink_rate_per_min_fast"]),
            "blink_active": int(line["blink_active"]),
            "last_blink_time": line["last_blink_time"],
            "pupil_visual_baseline": self.pupil_visual_baseline,
        }

    # ------------------------------------------------------------------
    # Ruta lenta: pupila limpia y gaze para UI
    # ------------------------------------------------------------------

    def get_slow_metrics_cached(
        self,
        last_n_seconds: float = 10.0,
        min_interval_s: float | None = None,
    ) -> dict[str, float]:
        """
        Ruta lenta para UI.

        Calcula pupila limpia y estabilidad de mirada, pero solo cada
        min_interval_s. No recalcula esto en cada frame.
        """
        if min_interval_s is None:
            min_interval_s = self.config.slow_metrics_cache_interval_s

        if len(self.timestamps) == 0:
            return self._empty_slow_metrics()

        now = time.perf_counter()
        last_timestamp = float(self.timestamps[-1])
        window = float(last_n_seconds)

        interval_valid = (
            self._slow_metrics_cache is not None
            and (now - self._slow_metrics_cache_time) < float(min_interval_s)
            and self._slow_metrics_cache_window == window
        )

        if interval_valid:
            return dict(self._slow_metrics_cache)

        metrics = self.get_slow_metrics(last_n_seconds=window)

        self._slow_metrics_cache = dict(metrics)
        self._slow_metrics_cache_time = now
        self._slow_metrics_cache_last_timestamp = last_timestamp
        self._slow_metrics_cache_window = window

        return metrics

    def get_slow_metrics(self, last_n_seconds: float = 10.0) -> dict[str, float]:
        """
        Calcula metricas lentas recientes:
        - pupila izquierda limpia reciente;
        - pupila derecha limpia reciente;
        - estabilidad de mirada.

        No calcula blink, porque blink tiene su propia ruta rapida.
        """
        empty = self._empty_slow_metrics()

        if len(self.timestamps) == 0:
            return empty

        t_end = float(self.timestamps[-1])
        t_start = t_end - float(last_n_seconds)

        t, pl, pr, gx, gy = self._slice_between(t_start, t_end)

        if len(t) == 0:
            return empty

        pupil_left_clean = self._clean_single_pupil(t, pl)
        pupil_right_clean = self._clean_single_pupil(t, pr)

        return {
            "pupil_left_recent": self._safe_nanmean(pupil_left_clean),
            "pupil_right_recent": self._safe_nanmean(pupil_right_clean),
            "gaze_stability": self._compute_gaze_stability(gx, gy),
            "pupil_visual_baseline": self.pupil_visual_baseline,
        }

    def get_recent_metrics_cached(
        self,
        last_n_seconds: float = 10.0,
        fast_interval_s: float | None = None,
        slow_interval_s: float | None = None,
    ) -> dict[str, float]:
        """
        Metodo recomendado para la UI.

        Combina:
        - blink rapido cada fast_interval_s;
        - pupila/gaze lento cada slow_interval_s.
        """
        fast = self.get_fast_blink_metrics(
            last_n_seconds=last_n_seconds,
            min_interval_s=fast_interval_s,
        )

        slow = self.get_slow_metrics_cached(
            last_n_seconds=last_n_seconds,
            min_interval_s=slow_interval_s,
        )

        return {
            "blink_count_total": int(fast["blink_count_total"]),
            "blink_rate_per_min": float(fast["blink_rate_per_min"]),
            "blink_active": int(fast["blink_active"]),
            "last_blink_time": fast["last_blink_time"],
            "pupil_left_recent": float(slow["pupil_left_recent"]),
            "pupil_right_recent": float(slow["pupil_right_recent"]),
            "gaze_stability": float(slow["gaze_stability"]),
            "pupil_visual_baseline": self.pupil_visual_baseline,
        }

    def get_recent_metrics(self, last_n_seconds: float = 10.0) -> dict[str, float]:
        """
        Compatibilidad con codigo antiguo.

        Si la UI sigue llamando a get_recent_metrics(), tambien se beneficia
        de la separacion rapida/lenta, aunque lo ideal es llamar a
        get_recent_metrics_cached().
        """
        return self.get_recent_metrics_cached(
            last_n_seconds=last_n_seconds,
            fast_interval_s=self.config.fast_blink_cache_interval_s,
            slow_interval_s=self.config.slow_metrics_cache_interval_s,
        )

    # ------------------------------------------------------------------
    # Extraccion de features por trial
    # ------------------------------------------------------------------

    def extract_trial_features(self, t_start: float, t_end: float) -> dict[str, float]:
        """
        Extrae features de eye-tracking para un intervalo temporal.

        Aqui SI se usa el preprocesado completo, porque es para guardar/analizar
        el trial, no para pintar una linea rapida de UI.
        """
        t, pl, pr, gx, gy = self._slice_between(t_start, t_end)

        if len(t) == 0:
            return self._empty_features()

        duration_s = max(1e-6, float(t[-1] - t[0]))

        pupil_left_clean = self._clean_single_pupil(t, pl)
        pupil_right_clean = self._clean_single_pupil(t, pr)
        pupil_avg_clean = self._clean_pupil_average(t, pl, pr)

        pupil_mean = self._safe_nanmean(pupil_avg_clean)
        pupil_std = self._safe_nanstd(pupil_avg_clean)

        pupil_left_mean = self._safe_nanmean(pupil_left_clean)
        pupil_right_mean = self._safe_nanmean(pupil_right_clean)

        # Normalizacion pupilar segun Mathot 2018:
        #   "zscore":   (pupil - baseline_mean) / baseline_std
        #   "percent":  (pupil - baseline_mean) / baseline_mean * 100
        #   "subtract": pupil - baseline_mean (legacy)
        pupil_norm_mean = 0.0
        if (
            self.baseline_pupil_mean is not None
            and np.isfinite(self.baseline_pupil_mean)
            and np.isfinite(pupil_mean)
        ):
            mode = self.config.pupil_norm_mode

            if mode == "zscore" and self.baseline_pupil_std is not None:
                std = max(1e-6, float(self.baseline_pupil_std))
                pupil_norm_mean = (pupil_mean - self.baseline_pupil_mean) / std
            elif mode == "percent":
                base = max(1e-6, float(self.baseline_pupil_mean))
                pupil_norm_mean = (pupil_mean - self.baseline_pupil_mean) / base * 100.0
            else:  # subtract (legacy) o zscore sin std disponible
                pupil_norm_mean = pupil_mean - self.baseline_pupil_mean

        all_events = self._detect_blink_events(t, pl, pr, include_artifacts=True)
        blink_count = len([e for e in all_events if e.kind == "blink"])
        artifact_count = len([e for e in all_events if e.kind == "artifact"])

        blink_rate_per_min = 60.0 * blink_count / duration_s

        gaze_stability = self._compute_gaze_stability(gx, gy)
        gaze_distance_center = self._compute_gaze_distance_center(gx, gy)

        return {
            "duration_s": duration_s,
            "pupil_mean_mm": pupil_mean,
            "pupil_std_mm": pupil_std,
            "pupil_left_mean": pupil_left_mean,
            "pupil_right_mean": pupil_right_mean,
            "pupil_norm_mean": pupil_norm_mean,
            "blink_count": float(blink_count),
            "blink_rate_per_min": float(blink_rate_per_min),
            "blink_artifact_count": float(artifact_count),
            "gaze_stability": gaze_stability,
            "gaze_distance_center": gaze_distance_center,
        }

    # ------------------------------------------------------------------
    # Internos: buffer
    # ------------------------------------------------------------------

    def _trim_buffers(self):
        if len(self.timestamps) <= self._max_samples:
            return

        excess = len(self.timestamps) - self._max_samples

        del self.timestamps[:excess]
        del self.pupil_l[:excess]
        del self.pupil_r[:excess]
        del self.gaze_x[:excess]
        del self.gaze_y[:excess]

        # Al recortar, las caches pueden contener arrays ya no representativos.
        self._fast_blink_cache = None
        self._slow_metrics_cache = None

    def _slice_indices_between(self, t_start: float, t_end: float) -> tuple[int, int]:
        if len(self.timestamps) == 0:
            return 0, 0

        t_start = float(t_start)
        t_end = float(t_end)

        if t_end < t_start:
            return 0, 0

        i0 = bisect.bisect_left(self.timestamps, t_start)
        i1 = bisect.bisect_right(self.timestamps, t_end)

        return i0, i1

    def _slice_between(
        self,
        t_start: float,
        t_end: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Devuelve el tramo temporal solicitado.

        Version optimizada:
        - no convierte todo el buffer a arrays;
        - usa bisect sobre timestamps;
        - solo convierte a NumPy el fragmento necesario.
        """
        if len(self.timestamps) == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty, empty

        i0, i1 = self._slice_indices_between(t_start, t_end)

        if i1 <= i0:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty, empty

        return (
            np.asarray(self.timestamps[i0:i1], dtype=float),
            np.asarray(self.pupil_l[i0:i1], dtype=float),
            np.asarray(self.pupil_r[i0:i1], dtype=float),
            np.asarray(self.gaze_x[i0:i1], dtype=float),
            np.asarray(self.gaze_y[i0:i1], dtype=float),
        )

    # ------------------------------------------------------------------
    # Internos: ruta rapida blink
    # ------------------------------------------------------------------

    def _raw_blink_mask(self, pupil_l: np.ndarray, pupil_r: np.ndarray) -> np.ndarray:
        """
        Mascara cruda de parpadeo.

        Un punto solo es candidato a blink si hay NaN simultaneo en ambos ojos.
        Los ceros, inf y valores fuera de rango se siguen limpiando en la ruta
        lenta de pupila, pero no se usan para detectar parpadeos.
        """
        pupil_l = np.asarray(pupil_l, dtype=float)
        pupil_r = np.asarray(pupil_r, dtype=float)

        return np.isnan(pupil_l) & np.isnan(pupil_r)

    def _raw_invalid_pupil_mask(self, pupil: np.ndarray) -> np.ndarray:
        pupil = np.asarray(pupil, dtype=float)

        return (
            ~np.isfinite(pupil)
            | (pupil <= 0.0)
            | (pupil < self.config.pupil_min_mm)
            | (pupil > self.config.pupil_max_mm)
        )

    def _detect_blink_events_fast_from_mask(
        self,
        t: np.ndarray,
        raw_mask: np.ndarray,
    ) -> list[BlinkEvent]:
        """
        Detecta blinks desde una mascara booleana ya restringida a NaN
        simultaneos en ambos ojos.

        Criterio aplicado:
        - minimo 2 muestras consecutivas NaN simultaneas;
        - maximo 15 muestras consecutivas NaN simultaneas;
        - si la racha tiene 1 muestra, se ignora;
        - si la racha tiene mas de 15 muestras, se considera perdida larga
          de tracking y no se cuenta como blink.
        """
        if len(t) < 2 or len(raw_mask) == 0:
            return []

        if not np.any(raw_mask):
            return []

        dt = np.diff(t)
        dt_valid = dt[np.isfinite(dt) & (dt > 0)]

        if len(dt_valid) > 0:
            sample_period = float(np.median(dt_valid))
        else:
            sample_period = 1.0 / max(1.0, self.config.fs)

        raw_runs = self._mask_runs(raw_mask)

        min_consecutive_nan = int(self.config.blink_min_consecutive_nan)
        max_consecutive_nan = int(self.config.blink_max_consecutive_nan)

        events: list[BlinkEvent] = []

        for start_idx, end_idx in raw_runs:
            n_consecutive_nan = int(end_idx - start_idx + 1)

            if n_consecutive_nan < min_consecutive_nan:
                continue

            if n_consecutive_nan > max_consecutive_nan:
                continue

            start_t = float(t[start_idx])
            end_t = float(t[end_idx])
            duration_s = max(sample_period, end_t - start_t + sample_period)

            events.append(
                BlinkEvent(
                    start_time=start_t,
                    end_time=end_t,
                    duration_s=duration_s,
                    kind="blink",
                )
            )

        return events

    def _empty_fast_blink_line(self) -> dict[str, Any]:
        return {
            "t": np.array([], dtype=float),
            "blink_line": np.array([], dtype=float),
            "blink_active": 0,
            "blink_count_fast": 0,
            "blink_rate_per_min_fast": 0.0,
            "last_blink_time": None,
            "pupil_visual_baseline": self.pupil_visual_baseline,
        }

    def _store_fast_blink_cache(
        self,
        result: dict[str, Any],
        now: float,
        last_timestamp: float,
        window: float,
    ):
        self._fast_blink_cache = result
        self._fast_blink_cache_time = now
        self._fast_blink_cache_last_timestamp = last_timestamp
        self._fast_blink_cache_window = window

    def _copy_fast_blink_cache(self, cache: dict[str, Any]) -> dict[str, Any]:
        result = dict(cache)

        if isinstance(result.get("t"), np.ndarray):
            result["t"] = result["t"].copy()

        if isinstance(result.get("blink_line"), np.ndarray):
            result["blink_line"] = result["blink_line"].copy()

        return result

    # ------------------------------------------------------------------
    # Internos: limpieza de pupila
    # ------------------------------------------------------------------

    def _clean_single_pupil(self, t: np.ndarray, pupil: np.ndarray) -> np.ndarray:
        pupil = np.asarray(pupil, dtype=float).copy()

        # Invalida: NaN, inf, <= 0 y fuera de rango fisiologico.
        invalid = (
            ~np.isfinite(pupil)
            | (pupil <= 0.0)
            | (pupil < self.config.pupil_min_mm)
            | (pupil > self.config.pupil_max_mm)
        )

        pupil[invalid] = np.nan
        pupil = self._interpolate_short_gaps(t, pupil)

        return pupil

    def _clean_pupil_average(
        self,
        t: np.ndarray,
        pupil_l: np.ndarray,
        pupil_r: np.ndarray,
    ) -> np.ndarray:
        """
        Limpia ambos ojos y calcula pupila media.

        Evita np.vstack y np.nanmean para reducir memoria y warnings.
        """
        left = self._clean_single_pupil(t, pupil_l)
        right = self._clean_single_pupil(t, pupil_r)

        avg = np.full_like(left, np.nan, dtype=float)

        left_valid = np.isfinite(left)
        right_valid = np.isfinite(right)

        both = left_valid & right_valid
        only_left = left_valid & ~right_valid
        only_right = ~left_valid & right_valid

        avg[both] = 0.5 * (left[both] + right[both])
        avg[only_left] = left[only_left]
        avg[only_right] = right[only_right]

        return avg

    def _interpolate_short_gaps(self, t: np.ndarray, values: np.ndarray) -> np.ndarray:
        if len(values) < 3:
            return values

        values = values.copy()
        valid = np.isfinite(values)

        if np.sum(valid) < 2:
            return values

        invalid_runs = self._mask_runs(~valid)

        for start_idx, end_idx in invalid_runs:
            start_t = float(t[start_idx])
            end_t = float(t[end_idx])
            gap_duration = max(0.0, end_t - start_t)

            if gap_duration > self.config.interpolation_max_gap_s:
                continue

            left_idx = start_idx - 1
            right_idx = end_idx + 1

            if left_idx < 0 or right_idx >= len(values):
                continue

            if not np.isfinite(values[left_idx]) or not np.isfinite(values[right_idx]):
                continue

            values[start_idx:right_idx] = np.interp(
                t[start_idx:right_idx],
                [t[left_idx], t[right_idx]],
                [values[left_idx], values[right_idx]],
            )

        return values

    # ------------------------------------------------------------------
    # Internos: deteccion de parpadeos completa para features de trial
    # ------------------------------------------------------------------

    def _detect_blink_events(
        self,
        t: np.ndarray,
        pupil_l: np.ndarray,
        pupil_r: np.ndarray,
        include_artifacts: bool = False,
    ) -> list[BlinkEvent]:
        """
        Detecta parpadeos como rachas de NaN simultaneos en ambos ojos.

        Estrategia:
        1. Mascara AND: blink si ambos ojos son NaN en la misma muestra.
        2. Minimo 2 y maximo 15 muestras consecutivas para contar blink.
        3. Las rachas largas pueden etiquetarse como artifact si include_artifacts=True.
        """
        if len(t) < 2:
            return []

        raw_mask = self._raw_blink_mask(pupil_l, pupil_r)

        if not np.any(raw_mask):
            return []

        events = self._detect_blink_events_fast_from_mask(t, raw_mask)

        if not include_artifacts:
            return events

        # Si tambien se piden artifacts, se vuelven a analizar las rachas largas.
        # Esto se usa para features de trial, no para UI rapida.
        dt = np.diff(t)
        dt_valid = dt[np.isfinite(dt) & (dt > 0)]

        if len(dt_valid) > 0:
            sample_period = float(np.median(dt_valid))
        else:
            sample_period = 1.0 / max(1.0, self.config.fs)

        raw_runs = self._mask_runs(raw_mask)
        max_consecutive_nan = int(self.config.blink_max_consecutive_nan)
        artifact_min = float(self.config.artifact_min_duration_s)

        all_events: list[BlinkEvent] = list(events)

        for start_idx, end_idx in raw_runs:
            n_consecutive_nan = int(end_idx - start_idx + 1)

            if n_consecutive_nan <= max_consecutive_nan:
                continue

            start_t = float(t[start_idx])
            end_t = float(t[end_idx])
            duration_s = max(sample_period, end_t - start_t + sample_period)

            if duration_s >= artifact_min:
                all_events.append(
                    BlinkEvent(
                        start_time=start_t,
                        end_time=end_t,
                        duration_s=duration_s,
                        kind="artifact",
                    )
                )

        return all_events

    def _blink_duration_limits(self) -> tuple[float, float]:
        if self.blink_zero_calibration:
            min_dur = float(
                self.blink_zero_calibration.get(
                    "min_duration_s",
                    self.config.default_blink_min_duration_s,
                )
            )
            max_dur = float(
                self.blink_zero_calibration.get(
                    "max_duration_s",
                    self.config.default_blink_max_duration_s,
                )
            )
            sample_period = float(
                self.blink_zero_calibration.get(
                    "sample_period_s",
                    1.0 / max(1.0, self.config.fs),
                )
            )

            min_dur = max(2.0 * sample_period, min_dur)
            max_dur = max(min_dur + sample_period, max_dur)

            return min_dur, max_dur

        return (
            float(self.config.default_blink_min_duration_s),
            float(self.config.default_blink_max_duration_s),
        )

    # ------------------------------------------------------------------
    # Internos: utilidades
    # ------------------------------------------------------------------

    def _mask_runs(self, mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start = None

        for i, value in enumerate(mask):
            if bool(value) and start is None:
                start = i
            elif not bool(value) and start is not None:
                runs.append((start, i - 1))
                start = None

        if start is not None:
            runs.append((start, len(mask) - 1))

        return runs

    def _compute_gaze_stability(self, gaze_x: np.ndarray, gaze_y: np.ndarray) -> float:
        gaze_x = np.asarray(gaze_x, dtype=float)
        gaze_y = np.asarray(gaze_y, dtype=float)

        valid = np.isfinite(gaze_x) & np.isfinite(gaze_y)

        if np.sum(valid) < 3:
            return 0.0

        gx = gaze_x[valid]
        gy = gaze_y[valid]

        dispersion = float(np.sqrt(np.nanvar(gx) + np.nanvar(gy)))
        stability = 1.0 - dispersion / max(1e-6, self.config.gaze_stability_scale)

        return float(np.clip(stability, 0.0, 1.0))

    def _compute_gaze_distance_center(self, gaze_x: np.ndarray, gaze_y: np.ndarray) -> float:
        gaze_x = np.asarray(gaze_x, dtype=float)
        gaze_y = np.asarray(gaze_y, dtype=float)

        valid = np.isfinite(gaze_x) & np.isfinite(gaze_y)

        if np.sum(valid) < 3:
            return 0.0

        dx = gaze_x[valid] - self.config.gaze_center_x
        dy = gaze_y[valid] - self.config.gaze_center_y
        distance = np.sqrt(dx * dx + dy * dy)

        return self._safe_nanmean(distance)

    def _safe_nanmean(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        valid = values[np.isfinite(values)]

        if len(valid) == 0:
            return 0.0

        return float(np.nanmean(valid))

    def _safe_nanstd(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        valid = values[np.isfinite(values)]

        if len(valid) == 0:
            return 0.0

        return float(np.nanstd(valid))

    def _empty_slow_metrics(self) -> dict[str, float]:
        return {
            "pupil_left_recent": 0.0,
            "pupil_right_recent": 0.0,
            "gaze_stability": 0.0,
            "pupil_visual_baseline": self.pupil_visual_baseline,
        }

    def _empty_features(self) -> dict[str, float]:
        return {
            "duration_s": 0.0,
            "pupil_mean_mm": 0.0,
            "pupil_std_mm": 0.0,
            "pupil_left_mean": 0.0,
            "pupil_right_mean": 0.0,
            "pupil_norm_mean": 0.0,
            "blink_count": 0.0,
            "blink_rate_per_min": 0.0,
            "blink_artifact_count": 0.0,
            "gaze_stability": 0.0,
            "gaze_distance_center": 0.0,
        }
