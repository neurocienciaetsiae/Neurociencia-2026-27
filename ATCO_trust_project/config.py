"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project / Arquitectura multimodal para la inferencia de la confianza en la
          automatizacion en tiempo real mediante la fusion de senales neurofisiologicas en entornos ATC
Codigo: config.py
Autora: Ana Ribote Gutierrez
Estado: Version preliminar
Version: 2.8
Fecha: 06/06/2026

Configuracion central del proyecto.

Cambios v2.8:
- Anadido ICA_PIPELINE: parametros para deteccion y limpieza de parpadeos
  por FastICA del EEG (Halder 2007, Castermans 2014).
- ET_PIPELINE: anadido pupil_norm_mode ("zscore" | "subtract" | "percent")
  para normalizacion pupilar segun Mathot 2018.
- ET_PIPELINE: anadido baseline_robust_stats (True/False) para usar mediana
  y MAD en lugar de media y std en el baseline pupilar.

Cambios v2.7:
- BUGFIX: ET_FS corregido de 50 Hz a 100 Hz. Los datos reales muestran ~99.7 Hz.
  Usar 50 Hz causaba que sample_period = 20 ms (en vez de 10 ms), afectando
  al calculo de duracion de parpadeos y al buffer del pipeline ET.
- ET_PIPELINE fs actualizado a 100.0 coherentemente.
- max_buflen_s añadido: controla el buffer LSL en StreamInlet. Se reduce de 120 s
  a 2 s para eliminar el delay de hasta 120 s observado en los datos.
- UPDATE_INTERVAL_MS mantenido en 50 ms.
"""

from __future__ import annotations

PLAYBACK_SPEED = 2.0

STREAM_NAMES = {
    "eeg": "102111-1035_eeg",
    "gaze2d": "Gaze2d",
    "pupil": "Pupil",
    "markers": "Markers",
    "gaze3d": "Gaze3d",
    "gaze_direction": "GazeDirection",
    "gaze_origin": "GazeOrigin",
    "sync": "SyncInOut",
    "magnetometer": "Magnetometer",
    "accelerometer": "Accelerometer",
    "gyroscope": "Gyroscope",
    "bat": "102111-1035_bat",
}

EEG_CHANNEL_NAMES = [
    "AF7",
    "AF3",
    "AFZ",
    "AF4",
    "AF8",
    "P3",
    "PZ",
    "P4",
]

DISPLAY_SECONDS = 10

# Ventana de procesado para EEG/modelos.
PROCESS_WINDOW_SECONDS = 2.0

UPDATE_INTERVAL_MS = 50

EEG_FS = 125.0

# CORREGIDO v2.7: el eye-tracker transmite a ~100 Hz (confirmado en datos reales:
# mediana de dt entre muestras = 10.03 ms → 99.7 Hz). El valor anterior (50.0 Hz)
# causaba que sample_period = 1/50 = 20 ms, haciendo que blink_min_duration =
# 2 * 20 ms = 40 ms en lugar de los 20 ms reales, y que el buffer del pipeline
# ET fuera el doble del necesario.
ET_FS = 100.0

# Buffer LSL máximo (segundos). Se reducirá al abrir cada StreamInlet.
# El valor anterior era 120 s, lo que permitia a LSL acumular hasta 2 minutos
# de datos antes de entregarlos, causando el delay de 9-10 s observado.
# Con 2 s la latencia máxima desde que llega el dato al bus hasta que la
# aplicacion lo procesa es <= 2 s (en practica << 100 ms).
LSL_MAX_BUFLEN_S = 2

EEG_PIPELINE = {
    "fs_in": 125.0,
    "fs_out": 125.0,

    "hp": 1.0,
    "lp": 45.0,
    "notch": 50.0,
    "notch_q": 30.0,
    "ref_channel": "PZ",
    "window_s": 0.500,
    "overlap": 0.50,
    "buffer_s": 30.0,
    "asr_threshold": 10.0,
    "filter_warmup_s": 5.0,

    # La diadema ya aplica paso alto/paso bajo internamente.
    "apply_bandpass": False,

    # Notch a 50 Hz para ruido de red.
    "apply_notch": True,

    # Mindtooth referencia por hardware. PZ se mantiene como canal EEG valido.
    "apply_rereference": False,

    "flat_threshold": 0.1,
    "noise_threshold": 10.0,

    # A 125 Hz, 16 muestras equivalen a ~128 ms.
    "chunk_min_samples": 16,

    # 1 s a 125 Hz.
    "bad_ch_min_samples": 125,
}

ET_PIPELINE = {
    # CORREGIDO v2.7: 100.0 en lugar de 50.0
    "fs": 100.0,

    "savgol_window_s": 0.150,
    "savgol_poly": 2,
    "buffer_s": 30.0,
    "baseline_s": 0.500,

    "max_buffer_seconds": 120.0,
    "pupil_min_mm": 2.0,
    "pupil_max_mm": 8.0,         # Ampliado: algunos sujetos superan 5.5 mm en pupila dilatada.
    "interpolation_max_gap_s": 0.35,
    "default_blink_min_duration_s": 0.06,
    "default_blink_max_duration_s": 0.80,
    "artifact_min_duration_s": 1.20,
    "gaze_center_x": 0.5,
    "gaze_center_y": 0.5,
    "gaze_stability_scale": 0.25,

    # Ruta rapida/lenta para UI.
    "fast_blink_window_s": 10.0,
    "fast_blink_cache_interval_s": 0.08,
    "slow_metrics_cache_interval_s": 2.0,

    # NUEVO v2.8: Modo de normalizacion pupilar segun Mathot (2018).
    #   "zscore":   (pupil - baseline_mean) / baseline_std   <- recomendado
    #   "percent":  (pupil - baseline_mean) / baseline_mean * 100
    #   "subtract": pupil - baseline_mean                    <- comportamiento legacy
    # Mathot recomienda zscore para comparacion entre sujetos.
    "pupil_norm_mode": "zscore",

    # NUEVO v2.8: si True, baseline usa mediana y MAD (robustas a outliers).
    # Si False, usa media y std (clasicas).
    "baseline_robust_stats": True,
}

# NUEVO v2.8: ICA para deteccion y limpieza de parpadeos en EEG.
# Referencia: Halder et al. 2007, "ICA-based BCI for online use".
# La calibracion se ajusta UNA vez sobre datos de ojos abiertos (~30 s con
# parpadeos visibles). Despues se aplica la matriz de unmixing aprendida a
# cada chunk nuevo en tiempo real. La matriz de mezclado es estable durante
# la sesion mientras los electrodos no se muevan.
ICA_PIPELINE = {
    # Si False, ICA se inicializa pero no se ajusta ni se aplica.
    # El sistema funciona como antes (solo ASR + bad channels + filtros).
    "enabled": True,

    # Segundos minimos de datos limpios necesarios para ajustar ICA.
    # Por debajo de este valor, el ajuste se cancela.
    "min_calibration_samples_s": 15.0,

    # Umbral de deteccion de parpadeos en n veces la desviacion robusta
    # (MAD * 1.4826) sobre la activacion de la componente del parpadeo.
    "threshold_n_std": 3.0,

    # Limites de duracion fisiologica del parpadeo (s).
    # Eventos fuera de estos limites se descartan como artefacto.
    "min_blink_duration_s": 0.05,
    "max_blink_duration_s": 0.80,

    # Si True, ICA limpia el EEG en cada chunk (elimina la componente del
    # parpadeo). Si False, ICA solo detecta blinks pero no modifica la
    # senyal EEG entregada al resto del pipeline.
    "clean_eeg_inline": True,

    # Umbral de curtosis para considerar una componente como candidata a
    # parpadeo. Las distribuciones gaussianas tienen curtosis=0 (Fisher).
    # Los parpadeos son picos breves de gran amplitud → curtosis muy positiva.
    "min_kurtosis_for_blink": 1.0,
}

EEG_BANDS = {
    "Theta":    (4.0,  8.0),
    "Alpha":    (8.0,  12.0),
    "LowBeta":  (12.0, 20.0),
    "HighBeta": (20.0, 30.0),
    "Gamma":    (30.0, 45.0),
}

DEFAULT_EEG_PLOT_COLOURS = [
    "#B3C8E6",
    "#FF8A7A",
    "#7DD3FC",
    "#4ADE80",
    "#D8B4FE",
    "#F9A8D4",
    "#FDE68A",
    "#86EFAC",
]

EEG_SENSOR_LAYOUT = [
    ("AF7", -0.62, -0.78, "#64748B"),
    ("AF3", -0.28, -0.86, "#22C55E"),
    ("AFZ",  0.00, -0.92, "#1FD43A"),
    ("AF4",  0.28, -0.86, "#EF4444"),
    ("AF8",  0.62, -0.78, "#F97316"),
    ("P3",  -0.22, -0.08, "#22C55E"),
    ("PZ",   0.00, -0.02, "#94A3B8"),
    ("P4",   0.22, -0.08, "#EF4444"),
]
