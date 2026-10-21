"""
Entidad: Universidad Politécnica de Madrid
Proyecto: ATC Trust Project / Arquitectura multimodal para la inferencia de la confianza
en la automatización en tiempo real mediante la fusión de señales neurofisiológicas
en entornos ATC

Código: lsl_bus.py
Autora: Ana Ribote Gutiérrez
Estado: Versión preliminar
Versión: 1.5
Fecha: 06/06/2026

Capa de acceso a LSL y estado compartido.

Cambios v1.5:
- Añadidos campos para parpadeos detectados por ICA del EEG:
  ica_active, ica_blink_count, ica_blink_rate_per_min, ica_blink_events.
- Añadido et_blink_events para que la UI dibuje los parpadeos del ET.
  Antes existian blink_line_time y blink_line pero nadie los rellenaba.
"""

from __future__ import annotations

import os
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

LSL_KNOWN_PEERS = ""
LSL_SESSION_ID = ""


def _apply_lsl_network_config() -> None:
    cfg_lines = ["[lab streaming layer]\n"]

    if LSL_KNOWN_PEERS:
        cfg_lines.append(f"KnownPeers = {LSL_KNOWN_PEERS}\n")

    if LSL_SESSION_ID:
        cfg_lines.append(f"SessionID = {LSL_SESSION_ID}\n")

    cfg_content = "".join(cfg_lines)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".cfg",
        prefix="lsl_api_",
        delete=False,
    )
    tmp.write(cfg_content)
    tmp.flush()
    tmp.close()

    os.environ["LSLAPICFG"] = tmp.name

    print("[lsl_bus] Configuración LSL aplicada:")
    print(cfg_content)


_apply_lsl_network_config()

try:
    from pylsl import StreamInlet, resolve_byprop, resolve_streams
except ImportError:
    StreamInlet = None
    resolve_byprop = None
    resolve_streams = None


@dataclass
class SharedState:
    # EEG principal
    eeg_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    eeg_data: Dict[str, Deque[float]] = field(default_factory=dict)

    # ET usado actualmente
    eye_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    gaze_value: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    pupil_left: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    pupil_right: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))

    # Linea visual de parpadeos para la UI.
    # El processor la rellena con eventos aceptados como blink, no con cualquier NaN.
    blink_line_time: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    blink_line: Deque[float] = field(default_factory=lambda: deque(maxlen=2000))

    # Streams adicionales preparados
    gaze3d_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    gaze3d_x: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    gaze3d_y: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    gaze3d_z: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))

    gaze_origin_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    gaze_origin_values: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5000))

    gaze_direction_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    gaze_direction_values: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5000))

    accel_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    accel_values: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5000))

    gyro_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    gyro_values: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5000))

    magnetometer_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    magnetometer_values: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5000))

    sync_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    sync_values: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5000))

    bat_time: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    bat_values: Deque[list[float]] = field(default_factory=lambda: deque(maxlen=5000))

    # Features
    band_powers: Dict[str, float] = field(default_factory=dict)
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    # Baseline visual de pupila para la UI (mm). Se actualiza en cada fixation cross.
    # La UI resta este valor a la grafica para centrarla en 0.
    pupil_visual_baseline: float | None = None

    # ICA blinks (NUEVO v1.5): parpadeos detectados por la componente ICA del EEG.
    # Solo se rellenan despues de que ICA se haya entrenado (al final de la
    # calibracion con ojos abiertos). Antes de eso, todas las colas estan vacias
    # y ica_active = False.
    ica_active: bool = False
    ica_blink_count: int = 0
    ica_blink_rate_per_min: float = 0.0
    # Lista de (start_ts_lsl, end_ts_lsl) de blinks ICA recientes.
    # Maxlen pequeno porque se usa para dibujar la ventana visible (10 s).
    ica_blink_events: Deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=200)
    )

    # ET blinks como linea continua 0/1 (NUEVO v1.5: ahora realmente populated).
    # El processor.py la rellena en _record_new_blinks con los eventos detectados
    # por et_pipeline.get_blink_line_fast(). Antes esta cola existia pero nadie
    # la escribia, por lo que la UI mostraba siempre la curva vacia.
    et_blink_events: Deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=200)
    )

    # NUEVO v1.6: True cuando el marcador END ha cerrado la sesion. La UI lo
    # usa para mostrar "Session ended" en lugar del estado de grabacion activa.
    session_finalized: bool = False

    # Modelos
    trust_value: float = 50.0
    workload_value: float = 50.0
    fatigue_value: float = 50.0

    # Frecuencias reales
    eeg_fs_real: float = 0.0
    eye_fs_real: float = 0.0

    # Marcadores
    recent_markers: Deque[str] = field(default_factory=lambda: deque(maxlen=8))

    # Calidad de señal EEG
    # bad_channels: lista de nombres de canales confirmados como malos
    # tras la calibración (ojos abiertos). Se actualiza una vez por sesión.
    bad_channels: list = field(default_factory=list)

    # calibration_done: True cuando finish_calibration() ha completado.
    # Permite al popup saber si ya hay información fiable de canales malos.
    calibration_done: bool = False


class LSLStreamReader:
    def __init__(self, stream_name: str, timeout: float = 5.0):
        if resolve_byprop is None or StreamInlet is None:
            raise RuntimeError("pylsl no está instalado. Instala pylsl primero.")

        results = resolve_byprop("name", stream_name, timeout=timeout)
        if not results:
            raise RuntimeError(f"No se encontró stream LSL con nombre: {stream_name}")

        self.inlet = StreamInlet(results[0], max_buflen=120)

    def pull_sample(self, timeout: float = 0.0):
        return self.inlet.pull_sample(timeout=timeout)