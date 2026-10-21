"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: processor.py
Autora: Ana Ribote Gutierrez
Version: 3.5
Fecha: 22/06/2026

Procesado online desde LSL.

Cambios v3.5 (arranque secuencial + reconexion de streams):
------------------------------------------------------------

[ARRANQUE SECUENCIAL DE STREAMS]
run_from_lsl ahora conecta en dos fases ordenadas:
  1. Primero solo se busca el stream de markers (bloqueante, obligatorio).
     El sistema no avanza hasta tenerlo: garantiza que ningun dato EEG/ET
     llega antes de poder recibir el WELCOME.
  2. Solo cuando markers esta conectado se lanzan en paralelo los hilos
     de EEG / gaze2d / pupil (opcionales, no bloqueantes).
Razon: evita la ventana de riesgo anterior donde EEG/ET podian conectar
antes que markers, acumulando datos sin contexto de fase.

[RECONEXION AUTOMATICA DE STREAMS PERDIDOS]
El loop principal ahora supervisa continuamente cada inlet. Si un stream
(EEG, gaze2d o pupil) deja de emitir datos durante mas de
_STREAM_TIMEOUT_S segundos (default: 5 s), se considera perdido:
  - Se descarta el inlet roto (inlets[key] = None).
  - Se lanza un hilo daemon que llama a _open_inlet con required=False
    y drain_on_connect=True para reconectar en segundo plano.
  - Mientras el inlet esta None, _pull_eeg / _pull_et lo omiten sin
    crashear.
  - Al reconectar, el hilo escribe directamente en inlets[key] y el
    loop principal lo retoma en la siguiente iteracion.
El stream de markers NO participa en la reconexion automatica: es
obligatorio y su perdida es un error critico que detiene el sistema.

Cambios v3.4 (autoguardado por END + gating de interpolacion):
--------------------------------------------------------------

[AUTOGUARDADO POR MARCADOR END]
TrialEngine ya emitia el callback on_experiment_end cuando llegaba el
marcador "END" de PsychoPy, pero no estaba conectado a nada. Ahora:
  - on_experiment_end -> self._on_experiment_end -> set _finalize_requested.
  - El loop principal comprueba el flag al final de cada iteracion. Si
    esta activo, llama a self.stop(generate_report=True) que hace flush,
    genera el PNG resumen y cierra los CSVs.
  - El propio marcador END queda escrito en markers.csv antes del cierre.
  - SharedState.session_finalized = True para que la UI lo refleje.

[GATING DE INTERPOLACION POST-CALIBRACION]
Las metricas que aplican interpolacion de huecos pupilares
(get_recent_metrics, get_slow_metrics, extract_trial_features) ya no se
ejecutan hasta que termina la calibracion completa (CALIBRATION_END_EYES_OPEN).
Razon: durante la calibracion no hay baseline establecido ni datos en
tarea, e interpolar antes de tener referencias es perder informacion
sin ganar nada.
  - Flag _full_calibration_done = True al final de EYES_OPEN.
  - _update_recent_et_metrics() retorna temprano si False.
  - Streaming preprocessing (saneado local sin interpolacion) sigue activo
    siempre, asi que pupil_preprocessed.csv y gaze2d_preprocessed.csv se
    siguen escribiendo coherentes desde el principio.
  - Las features de trial (extract_trial_features) solo se llaman desde
    _on_confidence_response, que naturalmente ocurre post-calibracion.

Cambios v3.3 (alineacion marcador-senyal):
-------------------------------------------

[TIMELINE DE CONTEXTO]
Introducido _TrialTimeline: cronologia ordenada de transiciones (timestamp,
context) que se construye marcador a marcador conforme PsychoPy los envia.
Para etiquetar cada muestra de senyal (EEG/gaze/pupil/blink) se busca con
bisect.bisect_right la transicion vigente en el timestamp de la muestra
(O(log n)). Sustituye al snapshot del TrialEngine, que tenia dos problemas:

  1. Race condition por orden del loop: si markers se procesaban DESPUES
     de EEG/ET en la misma iteracion, las muestras quedaban etiquetadas
     con la fase ANTERIOR al marcador.
  2. Etiquetado uniforme por chunk: todas las muestras de un chunk EEG
     recibian el mismo contexto aunque una transicion ocurriera en mitad
     del chunk.

[REORDENADO DEL LOOP]
markers PRIMERO, luego EEG, ET, blinks, metricas. Asi en cada iteracion la
timeline esta actualizada antes de procesar muestras. Combinado con el
contexto-por-timestamp, las transiciones quedan correctamente reflejadas
incluso cuando varios streams pulan datos al mismo tiempo.

[CONTEXTO POR MUESTRA EN _pull_eeg Y _sync_et_by_timestamp]
Donde antes habia un solo `context = self._get_recorder_context()` por
batch, ahora se llama a `self._context_at(ts)` por cada muestra
individualmente. Coste adicional: una busqueda binaria por muestra
(despreciable a 125 Hz EEG y 100 Hz ET).

[CONTEXTO POR EVENTO EN BLINKS]
_record_new_blinks y _consume_ica_blinks (ET-NaN e ICA-EEG) etiquetan
ahora cada blink con el contexto en el instante de su inicio.

[ESTRUCTURA DE CARPETAS RAW / PROCESSED]
SessionRecorder v1.4 reorganiza la salida:
  session_folder/
    markers.csv                    <- nexo raw-processed
    raw/  eeg_raw.csv, pupil.csv, gaze2d.csv
    processed/  eeg_preprocessed.csv, pupil_preprocessed.csv,
                gaze2d_preprocessed.csv, blink_events.csv,
                trial_summary.csv, metrics_timeseries.csv
    session_timeline_summary.png

[ET PREPROCESADO STREAMING]
Nuevos pupil_preprocessed.csv y gaze2d_preprocessed.csv se generan en
tiempo real desde el processor llamando a et_pipeline.clean_sample_streaming()
para cada par sincronizado. Aplica saneado local (out-of-range / NaN / cero
/ media binocular) sin interpolacion temporal. La interpolacion sigue
ocurriendo solo en extract_trial_features() para los modelos.

Cambios v3.2 (integracion ICA):
-------------------------------

[ICA - DETECCION Y LIMPIEZA DE PARPADEOS EN EEG]
Se integra blink_ica.BlinkICADetector siguiendo el paradigma offline-fit +
online-apply (Halder 2007, Castermans 2014):
  - Durante CALIBRATION_START_EYES_OPEN se pide al EEGPipeline que acumule
    los chunks procesados (post-filtrado, post-bad-channels, post-rereferencia,
    pre-ASR). El sujeto parpadea a proposito durante ~30 s.
  - Al recibir CALIBRATION_END_EYES_OPEN se recuperan los datos acumulados
    y se ajusta FastICA. Si la calibracion tiene exito, el detector se
    inyecta en el pipeline EEG, que aplica la limpieza en linea a cada
    chunk antes del ASR.
  - Si el ajuste falla (datos insuficientes, sklearn ausente, ninguna
    componente identificada como parpadeo), el sistema sigue funcionando
    sin ICA (solo ASR + bad channels).

Orden de procesado EEG resultante:
  filtros -> bad channels -> rereferencia -> ICA clean -> ASR -> buffer
ICA va ANTES de ASR porque necesita ver los parpadeos con amplitud completa.

Cambios v3.1 (filtro pre-WELCOME):
------------------------------------

[DESCARTE PRE-WELCOME]
Todo dato de EEG y ET que llega antes del marcador WELCOME se descarta
y no se escribe en ningun CSV ni se introduce en ningun pipeline.
Motivacion: antes del WELCOME el participante aun no esta en la tarea,
los streams pueden estar iniciandose o en estado inconstante, y los datos
de ese periodo contaminarian los buffers de calibracion y el baseline.
Implementacion:
  - Flag _discard_until_welcome = True al inicio.
  - on_welcome callback conectado al TrialEngine.
  - Cuando llega WELCOME: se pone el flag a False, se limpian todos los
    buffers internos (EEG pipeline, ET pipeline, colas pendientes,
    SharedState) y se drena de nuevo el inlet LSL.
  - Hasta ese momento _pull_eeg y _pull_et leen y descartan sin escribir.
  - El marcador WELCOME si se graba (es el primer evento valido).

Cambios v3.0 (correcciones criticas):
--------------------------------------

[DELAY - CAUSA RAIZ CONFIRMADA]
El max_buflen=120 en StreamInlet permitia que LSL acumulara hasta 120 segundos
de datos en buffer antes de entregarlos. Los datos reales confirman offsets gaze-
pupil de hasta 120.2 s, exactamente el valor del buffer. Solucion:
  - LSL_MAX_BUFLEN_S = 2 (importado de config.py).
  - Tras abrir cada inlet, se drena el buffer con pull_chunk(timeout=0.0) para
    descartar datos acumulados antes de que la aplicacion conectara.
  - Se usa pull_chunk en lugar de pull_sample para EEG y ET: en un solo ciclo
    se procesan todos los datos disponibles, evitando que el buffer crezca.

[ET SYNC - MATCHING POR TIMESTAMP]
_sync_et_samples usaba n = min(len(gaze), len(pupil)), asumiendo llegada 1:1.
Con cualquier jitter de red o diferencia de tasa, una cola crece sin control
y los pares se desalinean en el tiempo. Solucion:
  - _sync_et_by_timestamp: empareja muestras por timestamp con tolerancia 50 ms.
  - Descarta muestras huerfanas (sin pareja en la otra cola) para que las colas
    no crezcan indefinidamente.

[CALIBRACION - SEPARACION OJOS ABIERTOS / CERRADOS]
_on_calib_start iniciaba la calibracion EEG para AMBOS tipos de calibracion.
Los datos de ojos abiertos contienen artefactos de parpadeo y movimientos
oculares que contaminan el baseline del ASR y la deteccion de canales malos.
Solucion:
  - La calibracion EEG (ASR + bad channels) solo se ejecuta durante
    CALIBRATION_EYES_CLOSED.
  - CALIBRATION_EYES_OPEN se reserva para: calibracion de parpadeos (blink_zero),
    baseline de pupila, y NUEVO v3.2: ajuste de ICA.
  - Se usa el calib_type recibido en el callback para discriminar.

[BLINKS EN ET - GRABACION DE EVENTOS]
Los eventos de parpadeo detectados por ETPipeline se graban en blink_events.csv
con timestamp y contexto experimental.

[TRIAL SUMMARY - TIMESTAMP DE FIN]
_compute_trial_end ahora pasa timestamp_end a write_trial_summary (antes
siempre se escribia 0.0 para t_end porque el campo no existia en TrialInfo).

[FEATURES - PERCLOS Y BLINK DURATION]
Se extraen perclos y blink_duration_mean del pipeline ET y se pasan a
build_feature_bundle para alimentar el modelo de fatiga.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict

import numpy as np
from pylsl import StreamInlet, resolve_byprop, proc_clocksync, proc_dejitter, local_clock

from config import (
    EEG_BANDS,
    EEG_FS,
    EEG_PIPELINE,
    ET_PIPELINE,
    ICA_PIPELINE,
    LSL_MAX_BUFLEN_S,
    PROCESS_WINDOW_SECONDS,
    STREAM_NAMES,
)
from eeg_preprocessing import EEGPipeline, EEGPipelineConfig
from et_preprocessing import ETPipeline, ETPipelineConfig
from features_extraction import (
    FeatureBundle,
    build_feature_bundle,
    compute_blink_intensity,
    compute_perclos,
)
from lsl_bus import SharedState
from markers import TrialEngine, TrialInfo
from trust_model import SyntheticTrustModel
from workload_model import WorkloadModel
from fatigue_model import FatigueModel
from session_recorder import SessionRecorder, RecorderContext

# Import opcional: si sklearn no esta disponible, BlinkICADetector no funcionara
# pero el resto del sistema sigue operativo.
try:
    from blink_ica import BlinkICADetector
    _BLINK_ICA_AVAILABLE = True
except Exception as _ica_exc:
    BlinkICADetector = None
    _BLINK_ICA_AVAILABLE = False
    print(f"[Processor] BlinkICADetector no disponible: {_ica_exc}")

import bisect


@dataclass
class RuntimeFeatures:
    all_features: Dict[str, float] = field(default_factory=dict)
    selected_features: Dict[str, float] = field(default_factory=dict)


class _TrialTimeline:
    """
    Cronologia ordenada de transiciones de contexto experimental.

    Cada vez que llega un marcador de PsychoPy y el TrialEngine cambia de
    fase, se registra una nueva entrada (timestamp_lsl, context). Para
    etiquetar una muestra de senyal (EEG, gaze, pupil), se busca por
    bisect la transicion mas reciente con timestamp <= sample_ts.

    Sustituye al enfoque "snapshot" anterior, que devolvia el contexto
    actual del TrialEngine en el momento de escribir. El snapshot tenia
    dos problemas:
      1. Race condition en el loop: si markers se procesaban DESPUES de
         EEG/ET en la misma iteracion, las muestras EEG/ET quedaban
         etiquetadas con la fase ANTERIOR al marcador.
      2. Etiquetado uniforme por chunk: todas las muestras de un mismo
         chunk obtenian el mismo contexto aunque una transicion ocurriera
         en mitad del chunk.

    Con timeline, cada muestra recibe el contexto que estaba activo en
    su propio timestamp LSL, independientemente del orden de procesado.
    """

    def __init__(self, default_context: RecorderContext):
        self._default = default_context
        self._ts: list[float] = []
        self._ctx: list[RecorderContext] = []

    def add(self, timestamp_lsl: float, context: RecorderContext) -> None:
        """Anyade una transicion. Los timestamps deben llegar en orden."""
        ts = float(timestamp_lsl)
        if self._ts and ts < self._ts[-1]:
            # Marcador llegado fuera de orden (no deberia pasar con LSL).
            # Lo descartamos para no romper la busqueda binaria.
            print(
                f"[Timeline] WARN: marcador con ts fuera de orden ignorado: "
                f"{ts:.3f} < {self._ts[-1]:.3f}"
            )
            return
        self._ts.append(ts)
        self._ctx.append(context)

    def context_at(self, timestamp_lsl: float) -> RecorderContext:
        """
        Devuelve el contexto vigente en el instante dado.
        Busqueda binaria O(log n). Si el timestamp es anterior al primer
        marcador, devuelve el contexto por defecto (IDLE).
        """
        if not self._ts:
            return self._default
        idx = bisect.bisect_right(self._ts, float(timestamp_lsl)) - 1
        if idx < 0:
            return self._default
        return self._ctx[idx]

    def clear(self) -> None:
        """Vacia la timeline (uso: reset al recibir WELCOME)."""
        self._ts.clear()
        self._ctx.clear()

    def __len__(self) -> int:
        return len(self._ts)


class OnlineProcessor:
    def __init__(
        self,
        state: SharedState,
        eeg_channel_names: list[str],
        fs: float = EEG_FS,
        active_models: list[str] | None = None,
        active_modalities: dict[str, bool] | None = None,
        participant: str = "-",
        session: str = "-",
        recording_base_folder: str | Path = "recordings",
    ):
        self.state = state
        self.eeg_channel_names = eeg_channel_names
        self.fs = fs
        self._stop = False
        self._already_stopped = False

        self.active_models = active_models or ["Trust"]
        self.active_modalities = active_modalities or {
            "eeg": True,
            "eye_tracking": True,
        }

        self.use_eeg = bool(self.active_modalities.get("eeg", True))
        self.use_et = bool(self.active_modalities.get("eye_tracking", True))

        eeg_cfg = EEGPipelineConfig(**EEG_PIPELINE)
        et_cfg = ETPipelineConfig(**ET_PIPELINE)

        self.eeg_pipeline = (
            EEGPipeline(config=eeg_cfg, channel_names=eeg_channel_names)
            if self.use_eeg else None
        )
        self.et_pipeline = (
            ETPipeline(config=et_cfg)
            if self.use_et else None
        )

        self.runtime_features = RuntimeFeatures()

        self.trust_model = (
            SyntheticTrustModel()
            if "Trust" in self.active_models else None
        )
        self.workload_model = (
            WorkloadModel(fs=eeg_cfg.fs_out, channel_names=eeg_channel_names)
            if "Workload" in self.active_models and self.use_eeg else None
        )
        self.fatigue_model = (
            FatigueModel()
            if "Fatigue" in self.active_models else None
        )

        # NUEVO v3.2: detector ICA de parpadeos (Halder 2007).
        # Se inicializa pero no se ajusta aqui; el ajuste se hace al
        # terminar la calibracion con ojos abiertos en _on_calib_end.
        self.blink_ica = None
        if (
            self.use_eeg
            and _BLINK_ICA_AVAILABLE
            and ICA_PIPELINE.get("enabled", True)
        ):
            try:
                self.blink_ica = BlinkICADetector(
                    channel_names=eeg_channel_names,
                    fs=eeg_cfg.fs_out,
                    threshold_n_std=ICA_PIPELINE.get("threshold_n_std", 3.0),
                    min_blink_duration_s=ICA_PIPELINE.get("min_blink_duration_s", 0.05),
                    max_blink_duration_s=ICA_PIPELINE.get("max_blink_duration_s", 0.80),
                )
                print("[Processor] BlinkICADetector inicializado (pendiente de calibracion).")
            except Exception as exc:
                print(f"[Processor] No se pudo inicializar BlinkICADetector: {exc}")
                self.blink_ica = None

        self._eeg_ts_window: Deque[float] = deque(maxlen=100)
        self._eye_ts_window: Deque[float] = deque(maxlen=100)

        # Colas pendientes para sincronizacion ET por timestamp
        self._pending_gaze_samples: Deque[list[float]] = deque()
        self._pending_gaze_ts: Deque[float] = deque()
        self._pending_pupil_samples: Deque[list[float]] = deque()
        self._pending_pupil_ts: Deque[float] = deque()

        # Tolerancia de emparejamiento entre gaze y pupil (s)
        # A 100 Hz, dt = 10 ms; tolerancia = 50 ms = 5 muestras
        self._et_sync_tol: float = 0.050
        # Descarta muestras huerfanas con mas de 2 s de antiguedad
        self._et_max_pending_age: float = 2.0

        # ------------------------------------------------------------------
        # Flag de descarte pre-WELCOME.
        # Mientras sea True, _pull_eeg y _pull_et leen LSL pero no escriben
        # en ningun CSV ni introducen datos en los pipelines.
        # Se pone a False en _on_welcome, que tambien limpia todos los buffers.
        # ------------------------------------------------------------------
        self._discard_until_welcome: bool = True

        self.trial_engine = TrialEngine()
        self.trial_engine.on_welcome             = self._on_welcome
        self.trial_engine.on_calib_start         = self._on_calib_start
        self.trial_engine.on_calib_end           = self._on_calib_end
        self.trial_engine.on_baseline_start      = self._on_baseline_start
        self.trial_engine.on_confidence_response = self._on_confidence_response
        # NUEVO v3.4: cuando PsychoPy envia el marcador END, autoguardar
        # la sesion (flush + reporte + close) sin necesidad de pulsar Stop.
        self.trial_engine.on_experiment_end      = self._on_experiment_end

        # Flag que se activa al recibir END. La finalizacion efectiva se
        # ejecuta al final de la iteracion del loop, asi se asegura que el
        # propio marcador END y los datos contiguos quedan grabados antes
        # de cerrar los CSVs.
        self._finalize_requested: bool = False

        # NUEVO v3.4: flag indicando que toda la calibracion ha terminado
        # (ojos cerrados Y ojos abiertos). Mientras sea False, no se computan
        # las metricas que requieren interpolacion de huecos (no tiene sentido
        # interpolar antes de tener baseline ni datos validos del sujeto en
        # tarea). Se activa al final de CALIBRATION_END_EYES_OPEN.
        self._full_calibration_done: bool = False

        # NUEVO v3.5: umbral de antiguedad para descartar markers obsoletos.
        # El stream Markers se conserva al conectar (drain_on_connect=False)
        # para no perder un WELCOME ya enviado. Pero si PsychoPy estuvo
        # corriendo en una sesion anterior, el buffer arrastra markers
        # viejos (potencialmente horas). Procesarlos en rafaga arruina las
        # calibraciones porque entre marker y marker apenas pasa EEG.
        # Cualquier marker con edad > este umbral se descarta como obsoleto.
        self._stale_marker_threshold_s: float = 60.0

        self.last_marker_text: str = ""

        # NUEVO v3.3: timeline de transiciones de contexto experimental.
        # Cada marcador procesado por TrialEngine deja una entrada
        # (timestamp_lsl, contexto). Para etiquetar una muestra de senyal
        # se busca la entrada activa en su timestamp via bisect, en lugar
        # del contexto actual del TrialEngine (que causaba etiquetado
        # incorrecto en chunks que cruzaban transiciones).
        self._trial_timeline = _TrialTimeline(default_context=RecorderContext())

        self.recorder = SessionRecorder(
            base_folder=recording_base_folder,
            participant=participant,
            session=session,
            eeg_channel_names=eeg_channel_names,
        )
        print(f"[Processor] Grabacion en: {self.recorder.session_folder}")

        self._last_metrics_record_ts: float = 0.0
        self._metrics_record_period_s: float = 0.25

        # Calibracion
        self._current_calibration_kind: str | None = None
        self._calib_open_pupil_l: list[float] = []
        self._calib_open_pupil_r: list[float] = []
        self._calib_open_ts: list[float] = []

        self._calib_workload_open_values: list[float] = []
        self._calib_workload_closed_values: list[float] = []
        self.workload_zero_reference: float | None = None
        self.blink_zero_calibration: dict[str, float] | None = None

        # Seguimiento de blinks para no registrar el mismo dos veces
        self._last_blink_end_ts: float = 0.0

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def request_stop(self):
        self._stop = True

    def stop(self, generate_report: bool = True):
        if self._already_stopped:
            if generate_report:
                return self.recorder.generate_final_report(active_models=self.active_models)
            return None

        self._stop = True
        self._already_stopped = True
        report_path = None

        try:
            self.recorder.flush()
            if generate_report:
                report_path = self.recorder.generate_final_report(
                    active_models=self.active_models
                )
        finally:
            self.recorder.close()

        return report_path

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _get_recorder_context(self) -> RecorderContext:
        phase = self.trial_engine.current_phase.name
        current_trial = self.trial_engine.state.current_trial

        return RecorderContext(
            phase=phase,
            block=getattr(current_trial, "block", 0),
            trial=getattr(current_trial, "trial", 0),
            condition=getattr(current_trial, "condition", ""),
            current_marker=self.last_marker_text,
        )

    def _context_at(self, timestamp_lsl: float) -> RecorderContext:
        """
        NUEVO v3.3: devuelve el contexto experimental vigente en el
        timestamp LSL dado, usando la timeline de transiciones.

        Esto sustituye a _get_recorder_context() para todas las muestras
        de senyal (EEG, gaze, pupil, blinks) que tengan timestamp LSL
        propio. Asi cada muestra queda etiquetada con la fase, trial y
        condicion que efectivamente estaban activos cuando se adquirio,
        no las del momento en que se procesa.

        Se sigue usando _get_recorder_context() para eventos sin
        timestamp claro (e.g., escritura del marcador mismo, en cuyo
        caso el contexto post-marcador es el correcto).
        """
        return self._trial_timeline.context_at(timestamp_lsl)

    def _open_inlet(
        self,
        stream_name: str,
        timeout: float = 10.0,
        retry_interval: float = 3.0,
        required: bool = True,
        drain_on_connect: bool = True,
        max_buflen: int | None = None,
    ) -> StreamInlet | None:
        while not self._stop:
            print(f"[Processor] Buscando stream LSL: '{stream_name}' ...")
            results = resolve_byprop("name", stream_name, timeout=timeout)

            if results:
                _buf = max_buflen if max_buflen is not None else LSL_MAX_BUFLEN_S
                # processing_flags activa sincronizacion automatica de relojes
                # (proc_clocksync) y suavizado de timestamps (proc_dejitter).
                # Sin esto, cada stream usa su reloj local y los timestamps no
                # son comparables entre dispositivos distintos.
                inlet = StreamInlet(
                    results[0],
                    max_buflen=_buf,
                    processing_flags=proc_clocksync | proc_dejitter,
                )

                # Drenar buffer solo para streams continuos (EEG, ET).
                # Para marcadores NO se drena: cada marcador es un evento
                # unico irrecuperable. Si PsychoPy envio WELCOME antes de
                # que la app conectara, debe poder recibirlo del buffer.
                if drain_on_connect:
                    try:
                        inlet.pull_chunk(max_samples=10000, timeout=0.0)
                        print(f"[Processor] Stream '{stream_name}' conectado (buffer drenado).")
                    except Exception:
                        print(f"[Processor] Stream '{stream_name}' conectado.")
                else:
                    print(f"[Processor] Stream '{stream_name}' conectado (buffer conservado).")

                return inlet

            if not required:
                print(f"[Processor] Stream '{stream_name}' no disponible, continuando sin el.")
                return None

            print(f"[Processor] Stream '{stream_name}' no encontrado. Reintentando en {retry_interval}s...")
            time.sleep(retry_interval)

        return None

    def _estimate_real_fs(self, ts_window: Deque[float]) -> float:
        if len(ts_window) < 3:
            return 0.0
        ts = np.asarray(ts_window, dtype=float)
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            return 0.0
        mean_dt = float(np.mean(dt))
        return float(1.0 / mean_dt) if mean_dt > 0 else 0.0

    # ------------------------------------------------------------------
    # ET sincronizacion por timestamp
    # CORREGIDO v3.0: emparejar gaze y pupil por timestamp en lugar de
    # por posicion en cola (el metodo anterior causaba desalineacion con
    # cualquier jitter de red o diferencia de tasa entre streams).
    # ------------------------------------------------------------------

    def _sync_et_by_timestamp(self):
        """
        Empareja muestras de gaze y pupil por proximidad temporal.

        Por cada muestra de gaze busca la muestra de pupil mas cercana
        dentro de self._et_sync_tol segundos (50 ms por defecto).
        Los pares validos se envian al ETPipeline y se graban.
        Las muestras sin pareja se descartan si son mas antiguas de
        self._et_max_pending_age segundos.
        """
        if not self.use_et or self.et_pipeline is None:
            return

        if not self._pending_gaze_ts or not self._pending_pupil_ts:
            return

        now_ts = max(
            self._pending_gaze_ts[-1] if self._pending_gaze_ts else 0.0,
            self._pending_pupil_ts[-1] if self._pending_pupil_ts else 0.0,
        )

        gaze_x_list:  list[float] = []
        gaze_y_list:  list[float] = []
        pupil_l_list: list[float] = []
        pupil_r_list: list[float] = []
        ts_list:      list[float] = []

        while self._pending_gaze_ts and self._pending_pupil_ts:
            gts = self._pending_gaze_ts[0]

            # Descarta gaze muy antiguo
            if now_ts - gts > self._et_max_pending_age:
                self._pending_gaze_samples.popleft()
                self._pending_gaze_ts.popleft()
                continue

            pts = self._pending_pupil_ts[0]

            # Descarta pupil muy antigua
            if now_ts - pts > self._et_max_pending_age:
                self._pending_pupil_samples.popleft()
                self._pending_pupil_ts.popleft()
                continue

            delta = abs(gts - pts)

            if delta <= self._et_sync_tol:
                # Par valido: consumir ambas colas
                g = self._pending_gaze_samples.popleft()
                self._pending_gaze_ts.popleft()
                p = self._pending_pupil_samples.popleft()
                self._pending_pupil_ts.popleft()

                gaze_x_val = float(g[0]) if len(g) > 0 else np.nan
                gaze_y_val = float(g[1]) if len(g) > 1 else np.nan
                pupil_l_val = float(p[0]) if len(p) > 0 else np.nan
                pupil_r_val = float(p[1]) if len(p) > 1 else np.nan

                # Usar timestamp medio para el par
                avg_ts = (gts + pts) / 2.0

                gaze_x_list.append(gaze_x_val)
                gaze_y_list.append(gaze_y_val)
                pupil_l_list.append(pupil_l_val)
                pupil_r_list.append(pupil_r_val)
                ts_list.append(avg_ts)

                # CORREGIDO v3.3: contexto por muestra via timeline.
                ctx_gaze  = self._context_at(gts)
                ctx_pupil = self._context_at(pts)
                ctx_avg   = self._context_at(avg_ts)

                # Grabar brutos en raw/
                self.recorder.write_gaze2d(
                    timestamp_lsl=gts,
                    gaze_x=gaze_x_val,
                    gaze_y=gaze_y_val,
                    context=ctx_gaze,
                )
                self.recorder.write_pupil(
                    timestamp_lsl=pts,
                    pupil_left=pupil_l_val,
                    pupil_right=pupil_r_val,
                    context=ctx_pupil,
                )

                # NUEVO v3.3: grabar preprocesado streaming en processed/.
                # Aplica solo saneado local (sin interpolacion temporal),
                # que ya es informativo: invalida out-of-range, NaN/inf/cero,
                # y calcula media binocular. La interpolacion completa
                # se hace en extract_trial_features.
                (
                    pl_c, pr_c, pm_c, gx_c, gy_c,
                ) = self.et_pipeline.clean_sample_streaming(
                    pupil_l_val, pupil_r_val, gaze_x_val, gaze_y_val,
                )
                self.recorder.write_pupil_preprocessed(
                    timestamp_lsl=pts,
                    pupil_left_clean=pl_c,
                    pupil_right_clean=pr_c,
                    pupil_mean_clean=pm_c,
                    context=ctx_pupil,
                )
                self.recorder.write_gaze2d_preprocessed(
                    timestamp_lsl=gts,
                    gaze_x_clean=gx_c,
                    gaze_y_clean=gy_c,
                    context=ctx_gaze,
                )

            elif gts < pts:
                # Gaze mas antiguo, sin pareja → descartar gaze
                self._pending_gaze_samples.popleft()
                self._pending_gaze_ts.popleft()
            else:
                # Pupil mas antigua, sin pareja → descartar pupil
                self._pending_pupil_samples.popleft()
                self._pending_pupil_ts.popleft()

        if ts_list:
            self.et_pipeline.push_chunk(
                pupil_l=np.asarray(pupil_l_list, dtype=float),
                pupil_r=np.asarray(pupil_r_list, dtype=float),
                gaze_x=np.asarray(gaze_x_list, dtype=float),
                gaze_y=np.asarray(gaze_y_list, dtype=float),
                timestamps=ts_list,
            )

    # ------------------------------------------------------------------
    # Grabacion de blinks nuevos
    # ------------------------------------------------------------------

    def _record_new_blinks(self):
        """
        Compara los eventos de parpadeo recientes del pipeline ET con el
        ultimo parpadeo grabado y registra los nuevos en blink_events.csv.

        Tambien actualiza state.et_blink_events para que la UI dibuje la
        linea de parpadeos ET (antes no se rellenaba en ningun sitio).
        """
        if self.et_pipeline is None:
            return

        now_ts = (
            float(self.state.eye_time[-1]) if self.state.eye_time else 0.0
        )
        if now_ts <= 0:
            return

        window_s = min(10.0, now_ts - self._last_blink_end_ts + 1.0)
        blink_data = self.et_pipeline.get_blink_line_fast(
            last_n_seconds=window_s,
            min_interval_s=0.0,  # sin cache para deteccion inmediata
        )

        # get_blink_line_fast trabaja con tiempos relativos; necesitamos absolutos
        t_rel = blink_data.get("t", np.array([]))
        if len(t_rel) == 0:
            return

        # t_rel[-1] = 0 (mas reciente); t_rel[0] = -window_s (mas antiguo)
        t_abs_end = now_ts
        t_abs = t_rel + t_abs_end  # t_abs[-1] = now_ts

        # Detectar events individuales con timestamps absolutos
        blink_line = blink_data.get("blink_line", np.array([]))
        if len(blink_line) == 0:
            return

        # Re-detectar eventos con timestamps absolutos para grabarlos
        in_blink = False
        blink_start = 0.0

        for i, (ts_abs, is_blink) in enumerate(zip(t_abs, blink_line)):
            if is_blink and not in_blink:
                in_blink = True
                blink_start = float(ts_abs)
            elif not is_blink and in_blink:
                in_blink = False
                blink_end = float(t_abs[i - 1]) if i > 0 else blink_start
                blink_dur = blink_end - blink_start

                if blink_start > self._last_blink_end_ts:
                    # CORREGIDO v3.3: contexto en el instante del blink,
                    # no en el momento de procesar.
                    self.recorder.write_blink_event(
                        timestamp_start_lsl=blink_start,
                        timestamp_end_lsl=blink_end,
                        duration_s=blink_dur,
                        kind="blink",
                        context=self._context_at(blink_start),
                    )
                    # NUEVO v3.2: publicar tambien en SharedState para la UI.
                    self.state.et_blink_events.append((blink_start, blink_end))
                    self._last_blink_end_ts = blink_end

    # ------------------------------------------------------------------
    # NUEVO v3.2: Consumo de blinks detectados por ICA
    # ------------------------------------------------------------------

    def _consume_ica_blinks(self):
        """
        Recupera los blinks detectados por ICA en el pipeline EEG desde la
        ultima llamada, los graba en blink_events.csv como kind="blink_ica"
        y los publica en SharedState para la UI.

        Solo se ejecuta si ICA esta entrenado (despues de la calibracion
        con ojos abiertos). state.ica_active se mantiene True desde ese
        momento.
        """
        if self.eeg_pipeline is None or not self.eeg_pipeline.has_ica:
            return

        # Marcar ICA como activo (la UI lo usa para mostrar/ocultar el panel)
        if not self.state.ica_active:
            self.state.ica_active = True

        events = self.eeg_pipeline.consume_ica_blink_events()
        if not events:
            return

        for ev in events:
            start_ts = float(getattr(ev, "timestamp_start", 0.0))
            end_ts = float(getattr(ev, "timestamp_end", start_ts))
            duration = float(getattr(ev, "duration_s", end_ts - start_ts))

            # CORREGIDO v3.3: contexto en el instante del blink (timeline).
            self.recorder.write_blink_event(
                timestamp_start_lsl=start_ts,
                timestamp_end_lsl=end_ts,
                duration_s=duration,
                kind="blink_ica",
                context=self._context_at(start_ts),
            )

            # Publicar para la UI
            self.state.ica_blink_events.append((start_ts, end_ts))
            self.state.ica_blink_count += 1

        # Actualizar tasa por minuto sobre los blinks ICA en los ultimos 60 s
        if self.state.ica_blink_events:
            now_ts = max(end_ts, float(self.state.eeg_time[-1]) if self.state.eeg_time else end_ts)
            window_s = 60.0
            recent = [
                (s, e) for (s, e) in self.state.ica_blink_events
                if now_ts - e <= window_s
            ]
            duration_min = window_s / 60.0
            self.state.ica_blink_rate_per_min = float(len(recent)) / duration_min

    # ------------------------------------------------------------------
    # Callback WELCOME: descarta todo lo previo y limpia buffers
    # ------------------------------------------------------------------

    def _on_welcome(self, timestamp: float):
        # Se dispara cuando llega el marcador WELCOME de PsychoPy.
        # Activa la grabacion y limpia todos los buffers internos
        # para que no quede ningun dato previo al inicio del experimento.
        # El marcador WELCOME si se graba: write_marker se llama DESPUES
        # de process_marker (que dispara este callback), por lo que WELCOME
        # es el primer evento registrado en markers.csv.
        print(f"[Processor] WELCOME recibido (t={timestamp:.3f}). "
              "Activando grabacion y limpiando buffers pre-experimento.")

        # 1. Activar grabacion
        self._discard_until_welcome = False

        # 2. Limpiar pipeline EEG (buffer circular + warm-up)
        if self.eeg_pipeline is not None:
            self.eeg_pipeline.reset()

        # 3. Limpiar pipeline ET
        if self.et_pipeline is not None:
            self.et_pipeline.timestamps.clear()
            self.et_pipeline.pupil_l.clear()
            self.et_pipeline.pupil_r.clear()
            self.et_pipeline.gaze_x.clear()
            self.et_pipeline.gaze_y.clear()
            self.et_pipeline._fast_blink_cache = None
            self.et_pipeline._slow_metrics_cache = None

        # 4. Limpiar colas de sincronizacion ET
        self._pending_gaze_samples.clear()
        self._pending_gaze_ts.clear()
        self._pending_pupil_samples.clear()
        self._pending_pupil_ts.clear()

        # 5. Limpiar SharedState de senyal
        self.state.eeg_time.clear()
        self.state.eye_time.clear()
        self.state.pupil_left.clear()
        self.state.pupil_right.clear()
        self.state.gaze_value.clear()
        for ch in self.state.eeg_data:
            self.state.eeg_data[ch].clear()

        # 6. Resetear ventanas de estimacion de FS
        self._eeg_ts_window.clear()
        self._eye_ts_window.clear()

        # 7. Resetear datos de calibracion previos al WELCOME
        self._calib_open_pupil_l.clear()
        self._calib_open_pupil_r.clear()
        self._calib_open_ts.clear()
        self._calib_workload_open_values.clear()
        self._calib_workload_closed_values.clear()

        # 8. Resetear t0 de metricas (elapsed_s empieza desde WELCOME)
        self.recorder._metrics_t0 = None
        self._last_metrics_record_ts = 0.0
        self._last_blink_end_ts = 0.0

        # 9. NUEVO v3.2: limpiar buffers de blinks ET e ICA en SharedState.
        # NO se resetea ica_active porque si el ICA ya estaba entrenado en
        # una sesion previa al WELCOME (caso raro), deberia conservarse.
        # En el flujo normal, ica_active sera False aqui porque la
        # calibracion de ojos abiertos sucede DESPUES del WELCOME.
        self.state.et_blink_events.clear()
        self.state.ica_blink_events.clear()
        self.state.ica_blink_count = 0
        self.state.ica_blink_rate_per_min = 0.0

        # 10. NUEVO v3.3: limpiar la timeline de marcadores. Pre-WELCOME no
        # deberia haber marcadores reales, pero por si acaso, descartamos.
        # El propio WELCOME se anyadira justo despues, al volver _pull_markers
        # de procesarlo, como primera entrada valida de la timeline.
        self._trial_timeline.clear()

        print("[Processor] Buffers limpiados. Grabacion activa desde WELCOME.")

    # ------------------------------------------------------------------
    # NUEVO v3.4: Callback de fin de experimento
    # ------------------------------------------------------------------

    def _on_experiment_end(self, timestamp: float):
        """
        Disparado por el TrialEngine cuando llega el marcador "END" de PsychoPy.

        No cierra el recorder ni para el loop inmediatamente: se limita a marcar
        _finalize_requested = True. El cierre real ocurre al final de la
        iteracion actual del loop principal, asegurando que:
          1. El propio marcador END queda escrito en markers.csv.
          2. Cualquier muestra de EEG/ET pendiente en colas se procesa y graba.
          3. La ultima fila de metrics_timeseries se escribe con el timestamp
             del END.
        Tras todo eso, run_loop() llama a self.stop() que hace flush + reporte
        + close. La UI puede seguir funcionando y mostrara el reporte generado.
        """
        print(
            f"[Processor] Marcador END recibido (t={timestamp:.3f}). "
            "Sesion se cerrara al final de esta iteracion."
        )
        self._finalize_requested = True

    # ------------------------------------------------------------------
    # Callbacks de calibracion y trial
    # ------------------------------------------------------------------

    def _on_calib_start(self, calib_type: str, timestamp: float):
        print(f"[Processor] Calibracion iniciada: {calib_type}")

        # CORREGIDO v3.0: solo iniciar calibracion EEG con ojos CERRADOS.
        # Los datos de ojos abiertos contienen parpadeos y movimientos
        # oculares que contaminan el baseline del ASR.
        calib_upper = calib_type.upper()
        is_closed = "CLOSED" in calib_upper or "CERRAD" in calib_upper
        is_open   = "OPEN"   in calib_upper or "ABIERT" in calib_upper

        if is_closed and self.eeg_pipeline is not None:
            self.eeg_pipeline.start_calibration()
            print("[Processor] Calibracion EEG iniciada (solo ojos cerrados).")

        # NUEVO v3.2: durante ojos abiertos, acumular datos para entrenar ICA.
        # Requiere que la calibracion de ojos cerrados ya haya terminado
        # (ASR calibrado, bad channels detectados) para que los datos
        # acumulados sean utiles.
        if (
            is_open
            and self.eeg_pipeline is not None
            and self.blink_ica is not None
        ):
            self.eeg_pipeline.start_ica_accumulation()
            print("[Processor] Inicio acumulacion datos para ICA (ojos abiertos).")

    def _on_calib_end(self, calib_type: str, timestamp: float):
        print(f"[Processor] Calibracion finalizada: {calib_type}")

        calib_upper = calib_type.upper()
        is_closed = "CLOSED" in calib_upper or "CERRAD" in calib_upper
        is_open   = "OPEN"   in calib_upper or "ABIERT" in calib_upper

        if is_closed and self.eeg_pipeline is not None:
            bad = self.eeg_pipeline.finish_calibration()
            self.eeg_pipeline.confirm_bad_channels(bad)
            self.state.bad_channels = list(bad)
            self.state.calibration_done = True
            print(f"[Processor] Canales malos publicados: {bad}")

        # NUEVO v3.2: ajustar ICA al terminar ojos abiertos.
        if (
            is_open
            and self.eeg_pipeline is not None
            and self.blink_ica is not None
        ):
            data = self.eeg_pipeline.finish_ica_accumulation()
            n_samples = int(len(data))
            min_required = int(
                ICA_PIPELINE.get("min_calibration_samples_s", 15.0)
                * self.eeg_pipeline.fs_out
            )

            if n_samples < min_required:
                print(
                    f"[Processor] ICA NO ajustado: datos insuficientes "
                    f"({n_samples} < {min_required} muestras)."
                )
                return

            print(f"[Processor] Ajustando ICA sobre {n_samples} muestras...")
            fit_result = self.blink_ica.fit(data)

            if fit_result.get("success", False):
                self.eeg_pipeline.set_ica_detector(
                    self.blink_ica,
                    clean_inline=bool(ICA_PIPELINE.get("clean_eeg_inline", True)),
                )
                print(
                    f"[Processor] ICA ajustado y conectado al pipeline EEG. "
                    f"Componente blink: #{fit_result['blink_component_idx']}, "
                    f"frontal_ratio={fit_result['frontal_ratio']:.3f}, "
                    f"kurt={fit_result['kurtosis']:.2f}, "
                    f"sign_coh={fit_result['sign_coherence']:.2f}"
                )
            else:
                print(
                    f"[Processor] ICA NO ajustado: {fit_result.get('reason', 'unknown')}. "
                    "El pipeline EEG continuara sin limpieza ICA."
                )

        # NUEVO v3.4: tras CALIBRATION_END_EYES_OPEN (con ICA o sin el),
        # toda la calibracion esta completa. A partir de aqui se permiten
        # las metricas con interpolacion de huecos.
        if is_open:
            self._full_calibration_done = True
            print(
                "[Processor] Calibracion completa. Metricas con interpolacion "
                "(slow metrics, trial features) habilitadas."
            )

    def _on_baseline_start(self, info: TrialInfo, timestamp: float):
        if self.et_pipeline is not None and info.t_baseline_start > 0:
            self.et_pipeline.set_baseline_window(info.t_baseline_start, timestamp)

    def _on_confidence_response(
        self, info: TrialInfo, condition: str, confidence: float, timestamp: float
    ):
        self._compute_trial_end(info, timestamp)

    def _compute_trial_end(self, info: TrialInfo, timestamp: float):
        band_dict = {name: 0.0 for name in EEG_BANDS}
        eeg_epoch = np.empty((0, len(self.eeg_channel_names)))

        if self.eeg_pipeline is not None:
            eeg_epoch = self.eeg_pipeline.extract_epoch(
                info.t_baseline_start, timestamp
            )
            if len(eeg_epoch) >= 128:
                band_dict = self.eeg_pipeline.band_powers(eeg_epoch, EEG_BANDS)

            self.state.band_powers = {
                "Theta":    band_dict.get("Theta",    0.0),
                "Alpha":    band_dict.get("Alpha",    0.0),
                "LowBeta":  band_dict.get("LowBeta",  0.0),
                "HighBeta": band_dict.get("HighBeta", 0.0),
                "Gamma":    band_dict.get("Gamma",    0.0),
            }

        et_features: dict = {}
        if self.et_pipeline is not None:
            et_features = self.et_pipeline.extract_trial_features(
                info.t_baseline_start, timestamp
            )

        # --- Features ET para modelos ---
        low_beta    = float(band_dict.get("LowBeta",  0.0))
        high_beta   = float(band_dict.get("HighBeta", 0.0))
        pupil_mean  = float(et_features.get("pupil_mean_mm",   0.0))
        pupil_norm  = float(et_features.get("pupil_norm_mean", 0.0))
        gaze_stab   = float(et_features.get("gaze_stability",  0.0))
        blink_rate  = float(et_features.get("blink_rate_per_min", 0.0))
        blink_dur   = float(et_features.get("blink_duration_mean", 0.0))

        # PERCLOS: % de muestras invalidas en el epoch ET
        if self.et_pipeline is not None and len(self.et_pipeline.pupil_l) > 0:
            # Extraer rango temporal del trial para PERCLOS
            try:
                from et_preprocessing import ETPipeline as _ETP
                t_slice, pl_slice, pr_slice, _, _ = self.et_pipeline._slice_between(
                    info.t_baseline_start, timestamp
                )
                perclos = compute_perclos(pl_slice, pr_slice,
                                          pupil_min_mm=self.et_pipeline.config.pupil_min_mm)
            except Exception:
                perclos = 0.0
        else:
            perclos = 0.0

        blink_intensity = compute_blink_intensity(
            float(et_features.get("pupil_left_mean",  0.0)),
            float(et_features.get("pupil_right_mean", 0.0)),
        )

        bundle: FeatureBundle = build_feature_bundle(
            band_powers=self.state.band_powers,
            low_beta=low_beta,
            high_beta=high_beta,
            blink_rate=blink_rate,
            blink_intensity=blink_intensity,
            pupil_mean=pupil_mean,
            pupil_norm_mean=pupil_norm,
            gaze_stability=gaze_stab,
            perclos=perclos,
            blink_duration_mean=blink_dur,
        )

        self.runtime_features.all_features = bundle.all_features
        self.runtime_features.selected_features = bundle.selected_features

        if self.trust_model is not None:
            self.state.trust_value = self.trust_model.predict(bundle.selected_features)

        if self.fatigue_model is not None:
            self.state.fatigue_value = self.fatigue_model.predict(bundle.selected_features)

        if self.workload_model is not None and len(eeg_epoch) >= 128:
            raw_workload = self.workload_model.predict(eeg_epoch)
            self.state.workload_value = self._apply_workload_baseline(raw_workload)

        # CORREGIDO v3.0: pasar timestamp_end para que trial_summary tenga t_response
        self.recorder.write_trial_summary(
            trial_info=info,
            trust_value=float(self.state.trust_value),
            workload_value=float(self.state.workload_value),
            fatigue_value=float(self.state.fatigue_value),
            timestamp_end=timestamp,
        )
        self._write_metrics(timestamp)
        self.recorder.flush()

    # ------------------------------------------------------------------
    # Loop principal LSL
    # ------------------------------------------------------------------

    # Segundos sin datos antes de considerar un inlet perdido (EEG / ET).
    _STREAM_TIMEOUT_S: float = 5.0

    def run_from_lsl(self):
        import threading as _th

        # ------------------------------------------------------------------
        # FASE 1: conectar SOLO el stream de markers (bloqueante).
        # No avanzamos hasta tenerlo: garantiza que el WELCOME puede
        # llegar antes de que ningun dato EEG/ET entre en el pipeline.
        # ------------------------------------------------------------------
        print("[Processor] Esperando stream de markers...")
        inlets: dict[str, StreamInlet | None] = {"markers": None}
        inlets["markers"] = self._open_inlet(
            STREAM_NAMES["markers"],
            required=True,
            drain_on_connect=False,   # conservar buffer: WELCOME puede estar ahi
            max_buflen=360,
        )
        if self._stop:
            return

        print("[Processor] Stream de markers conectado. Buscando EEG / ET...")

        # ------------------------------------------------------------------
        # FASE 2: conectar EEG y ET en paralelo (opcionales, no bloqueantes).
        # ------------------------------------------------------------------
        signal_keys: list[str] = []
        if self.use_eeg:
            signal_keys.append("eeg")
            inlets["eeg"] = None
        if self.use_et:
            signal_keys.extend(["gaze2d", "pupil"])
            inlets["gaze2d"] = None
            inlets["pupil"]  = None

        def _open_signal(key: str):
            inlets[key] = self._open_inlet(
                STREAM_NAMES[key],
                required=False,
                drain_on_connect=True,
            )

        sig_threads = [
            _th.Thread(target=_open_signal, args=(k,), daemon=True)
            for k in signal_keys
        ]
        for t in sig_threads:
            t.start()
        for t in sig_threads:
            t.join()

        if self._stop:
            return

        # ------------------------------------------------------------------
        # Timestamps de ultima muestra recibida por cada stream de señal.
        # Se usan para detectar perdida de stream durante el experimento.
        # ------------------------------------------------------------------
        _last_data_ts: dict[str, float] = {k: time.time() for k in signal_keys}
        # Flags: True = hay un hilo de reconexion activo para ese stream.
        _reconnecting: dict[str, bool] = {k: False for k in signal_keys}

        def _reconnect(key: str):
            """Hilo daemon: intenta reconectar un inlet perdido."""
            print(f"[Processor] Reconectando stream '{key}'...")
            new_inlet = self._open_inlet(
                STREAM_NAMES[key],
                required=False,
                drain_on_connect=True,
            )
            inlets[key] = new_inlet
            _reconnecting[key] = False
            if new_inlet is not None:
                _last_data_ts[key] = time.time()
                print(f"[Processor] Stream '{key}' reconectado.")
            else:
                print(f"[Processor] Stream '{key}' no disponible tras reconexion.")

        # El sistema espera indefinidamente hasta recibir el WELCOME real.
        # Mientras tanto los datos siguen llegando pero se descartan
        # (modo _discard_until_welcome). El usuario es responsable de
        # iniciar PsychoPy. Para una sesion de prueba sin PsychoPy, envia
        # un marcador WELCOME manualmente por otro outlet de LSL.

        while not self._stop:
            # v3.3: markers PRIMERO para que la timeline este actualizada
            # antes de procesar EEG y ET en esta iteracion.
            self._pull_markers(inlets)

            # ------------------------------------------------------------------
            # NUEVO v3.5: watchdog de reconexion para streams de señal perdidos.
            # _last_data_ts se actualiza en _pull_eeg / _pull_et cuando llegan
            # datos reales. Si un stream lleva mas de _STREAM_TIMEOUT_S sin
            # datos (inlet None o inlet vivo pero silencioso), se lanza un hilo
            # de reconexion en background.
            # El stream de markers NO participa: su perdida es critica y
            # detiene el sistema.
            # ------------------------------------------------------------------
            now = time.time()
            for key in signal_keys:
                silent = (now - _last_data_ts.get(key, now)) > self._STREAM_TIMEOUT_S
                already = _reconnecting.get(key, False)
                if silent and not already:
                    if inlets.get(key) is not None:
                        print(
                            f"[Processor] Stream '{key}' sin datos durante "
                            f"{self._STREAM_TIMEOUT_S:.0f}s. Reconectando..."
                        )
                        inlets[key] = None
                    _reconnecting[key] = True
                    _th.Thread(
                        target=_reconnect, args=(key,), daemon=True
                    ).start()

            # ------------------------------------------------------------------

            self._pull_eeg(inlets, _last_data_ts)
            self._pull_et(inlets, _last_data_ts)
            self._consume_ica_blinks()
            self._update_recent_et_metrics()
            self._record_new_blinks()
            self._update_online_workload()
            self._write_metrics_if_needed()

            # v3.4: si llego END en esta iteracion, cerrar la sesion.
            if self._finalize_requested and not self._already_stopped:
                print("[Processor] Finalizando sesion tras marcador END...")
                report_path = self.stop(generate_report=True)
                if report_path:
                    print(f"[Processor] Reporte de sesion generado: {report_path}")
                self.state.session_finalized = True
                break

            time.sleep(0.001)

    # ------------------------------------------------------------------
    # Pull EEG
    # CORREGIDO v3.0: usar pull_chunk en lugar de pull_sample para
    # procesar todas las muestras disponibles en cada ciclo.
    # ------------------------------------------------------------------

    def _pull_eeg(self, inlets: dict, _last_data_ts: dict | None = None):
        if not self.use_eeg or self.eeg_pipeline is None:
            return

        eeg_inlet = inlets.get("eeg")
        if eeg_inlet is None:
            return

        # pull_chunk drena todas las muestras disponibles en una sola llamada
        chunk, timestamps = eeg_inlet.pull_chunk(max_samples=64, timeout=0.0)

        if not chunk:
            return

        # v3.5: anotar que este stream sigue vivo.
        if _last_data_ts is not None:
            _last_data_ts["eeg"] = time.time()

        # Antes del WELCOME: leer y descartar sin escribir ni procesar.
        # Asi el buffer LSL no se acumula pero los datos no contaminan
        # el pipeline ni los CSVs.
        if self._discard_until_welcome:
            return

        chunk = np.asarray(chunk, dtype=float)      # (n, n_ch)
        timestamps = list(timestamps)               # (n,)

        # CORREGIDO v3.3: contexto por muestra via timeline (no por batch).
        for i, (sample, ts) in enumerate(zip(chunk, timestamps)):
            ts = float(ts)
            self.recorder.write_eeg_raw(
                timestamp_lsl=ts,
                values=sample.tolist(),
                context=self._context_at(ts),
            )
            self._eeg_ts_window.append(ts)

        self.state.eeg_fs_real = self._estimate_real_fs(self._eeg_ts_window)

        # Procesar el chunk completo de una vez
        clean_block = self.eeg_pipeline.push_chunk(
            chunk,
            timestamps=timestamps,
        )

        if len(clean_block) == 0:
            return

        if len(self.eeg_pipeline._buffer_ts) >= len(clean_block):
            out_ts = list(self.eeg_pipeline._buffer_ts)[-len(clean_block):]

            for i in range(len(clean_block)):
                out_ts_i = float(out_ts[i])
                self.state.eeg_time.append(out_ts_i)

                n_use = min(len(self.eeg_channel_names), clean_block.shape[1])
                clean_row = []

                for ch_idx in range(n_use):
                    ch_name = self.eeg_channel_names[ch_idx]
                    val = float(clean_block[i, ch_idx])
                    self.state.eeg_data[ch_name].append(val)
                    clean_row.append(val)

                # CORREGIDO v3.3: contexto por muestra (cada timestamp puede
                # caer en una fase distinta dentro del mismo chunk).
                self.recorder.write_eeg_preprocessed(
                    timestamp_lsl=out_ts_i,
                    values=clean_row,
                    context=self._context_at(out_ts_i),
                )

    # ------------------------------------------------------------------
    # Pull ET
    # CORREGIDO v3.0: usar pull_chunk + sincronizacion por timestamp.
    # ------------------------------------------------------------------

    def _pull_et(self, inlets: dict, _last_data_ts: dict | None = None):
        if not self.use_et or self.et_pipeline is None:
            return

        gaze_inlet  = inlets.get("gaze2d")
        pupil_inlet = inlets.get("pupil")

        # Antes del WELCOME: drenar los inlets sin procesar ni escribir.
        if self._discard_until_welcome:
            if gaze_inlet is not None:
                gaze_inlet.pull_chunk(max_samples=32, timeout=0.0)
            if pupil_inlet is not None:
                pupil_inlet.pull_chunk(max_samples=32, timeout=0.0)
            return

        if gaze_inlet is not None:
            gaze_chunk, gaze_timestamps = gaze_inlet.pull_chunk(
                max_samples=32, timeout=0.0
            )
            if gaze_chunk and _last_data_ts is not None:
                _last_data_ts["gaze2d"] = time.time()
            for sample, gts in zip(gaze_chunk, gaze_timestamps):
                vals = np.asarray(sample, dtype=float)
                self._pending_gaze_samples.append(vals.tolist())
                self._pending_gaze_ts.append(float(gts))

                if len(vals) >= 2 and np.isfinite(vals[0]) and np.isfinite(vals[1]):
                    dist_center = float(np.sqrt((vals[0] - 0.5)**2 + (vals[1] - 0.5)**2))
                else:
                    dist_center = np.nan
                self.state.gaze_value.append(dist_center)

        if pupil_inlet is not None:
            pupil_chunk, pupil_timestamps = pupil_inlet.pull_chunk(
                max_samples=32, timeout=0.0
            )
            if pupil_chunk and _last_data_ts is not None:
                _last_data_ts["pupil"] = time.time()
            for sample, pts in zip(pupil_chunk, pupil_timestamps):
                vals = np.asarray(sample, dtype=float)
                pts = float(pts)

                self._pending_pupil_samples.append(vals.tolist())
                self._pending_pupil_ts.append(pts)

                self.state.eye_time.append(pts)
                self._eye_ts_window.append(pts)
                self.state.eye_fs_real = self._estimate_real_fs(self._eye_ts_window)

                if len(vals) >= 2:
                    self.state.pupil_left.append(float(vals[0]))
                    self.state.pupil_right.append(float(vals[1]))

                    if self._current_calibration_kind == "open":
                        self._calib_open_pupil_l.append(float(vals[0]))
                        self._calib_open_pupil_r.append(float(vals[1]))
                        self._calib_open_ts.append(pts)

        # CORREGIDO v3.0: emparejar por timestamp en lugar de por conteo
        self._sync_et_by_timestamp()

    # ------------------------------------------------------------------
    # Pull markers
    # ------------------------------------------------------------------

    def _pull_markers(self, inlets: dict):
        marker_inlet = inlets.get("markers")
        if marker_inlet is None:
            return

        marker_sample, marker_ts = marker_inlet.pull_sample(timeout=0.01)
        if marker_sample is None:
            return

        marker = str(marker_sample[0]) if len(marker_sample) > 0 else ""
        marker_ts = float(marker_ts)

        # NUEVO v3.5: descartar markers obsoletos.
        # Si PsychoPy corrio en una sesion anterior, su outlet de LSL pudo
        # haber dejado markers en el buffer del Markers stream. Al conectar
        # con drain_on_connect=False (para no perder WELCOME), arrastramos
        # esos viejos. Procesarlos genera dos problemas:
        #   1. Se ejecutan en rafaga (~1ms entre ellos) → calibraciones vacias.
        #   2. La timeline acumula entradas con timestamps de hace horas →
        #      etiquetado de muestras actuales con fases obsoletas.
        # Solucion: comprobar la edad respecto al reloj LSL local y descartar
        # si supera _stale_marker_threshold_s. local_clock() devuelve el
        # mismo time base que los marker_ts si todo paso por proc_clocksync.
        try:
            now_lsl = local_clock()
            age = now_lsl - marker_ts
            if age > self._stale_marker_threshold_s:
                print(
                    f"[Processor] Marker obsoleto descartado: '{marker}' "
                    f"(edad={age:.1f}s, ts={marker_ts:.3f})"
                )
                return
        except Exception as exc:
            # Si local_clock falla por algun motivo, procesar el marker
            # como antes en vez de bloquear el flujo.
            print(f"[Processor] WARN: no se pudo comprobar edad del marker: {exc}")

        self.last_marker_text = marker
        text = f"{marker_ts:.3f}  {marker}"
        self.state.recent_markers.appendleft(text)

        self._handle_calibration_marker(marker, marker_ts)

        try:
            self.trial_engine.process_marker(marker, marker_ts)
        except Exception as exc:
            print(f"[Processor] ERROR en trial_engine.process_marker: {exc}")

        # NUEVO v3.3: registrar la transicion en la timeline.
        # Tras process_marker, el TrialEngine ya ha actualizado su estado
        # interno. El contexto que devuelve _get_recorder_context() es el
        # contexto POST-marcador, que es el que debe estar activo a partir
        # de marker_ts. Todas las muestras de senyal con timestamp >= marker_ts
        # heredaran este contexto via _context_at().
        post_marker_context = self._get_recorder_context()
        self._trial_timeline.add(marker_ts, post_marker_context)

        self.recorder.write_marker(
            timestamp_lsl=marker_ts,
            marker=marker,
            context=post_marker_context,
        )
        self._write_metrics(marker_ts)

    # ------------------------------------------------------------------
    # Manejo de markers de calibracion
    # ------------------------------------------------------------------

    def _handle_calibration_marker(self, marker: str, marker_ts: float):
        m = marker.upper()

        is_calib = "CALIB" in m or "CALIBRATION" in m or "BASELINE" in m
        if not is_calib:
            return

        is_open   = "OPEN"  in m or "ABIERT" in m or "EYES_OPEN"   in m
        is_closed = "CLOSE" in m or "CERRAD" in m or "EYES_CLOSED" in m
        is_start  = "START" in m or "INICIO" in m or "BEGIN" in m
        is_end    = "END"   in m or "FIN"    in m or "STOP"  in m

        if is_start and is_open:
            self._current_calibration_kind = "open"
            self._calib_open_pupil_l.clear()
            self._calib_open_pupil_r.clear()
            self._calib_open_ts.clear()
            self._calib_workload_open_values.clear()
            print("[Processor] Inicio calibracion ojos abiertos.")
            return

        if is_start and is_closed:
            self._current_calibration_kind = "closed"
            self._calib_workload_closed_values.clear()
            print("[Processor] Inicio calibracion ojos cerrados.")
            return

        if is_end and self._current_calibration_kind == "open":
            self._finish_open_eye_calibration()
            self._finish_workload_baseline(kind="open")
            self._current_calibration_kind = None
            print("[Processor] Fin calibracion ojos abiertos.")
            return

        if is_end and self._current_calibration_kind == "closed":
            self._finish_workload_baseline(kind="closed")
            self._current_calibration_kind = None
            print("[Processor] Fin calibracion ojos cerrados.")
            return

    # ------------------------------------------------------------------
    # Calibracion de parpadeos con ojos abiertos
    # ------------------------------------------------------------------

    def _finish_open_eye_calibration(self):
        if not self._calib_open_ts:
            return

        pupil_l = np.asarray(self._calib_open_pupil_l, dtype=float)
        pupil_r = np.asarray(self._calib_open_pupil_r, dtype=float)
        ts      = np.asarray(self._calib_open_ts,      dtype=float)

        if len(ts) < 3:
            return

        pupil_avg = np.nanmean(np.vstack([pupil_l, pupil_r]), axis=0)
        zero_mask = np.isfinite(pupil_avg) & (pupil_avg <= 0.0)
        nan_mask  = ~np.isfinite(pupil_avg)
        invalid_mask = zero_mask | nan_mask

        run_lengths = self._zero_run_lengths(invalid_mask)

        if not run_lengths:
            print("[Processor] No se detectaron parpadeos durante calibracion (ningun cierre ocular).")
            return

        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0)]

        if len(dt) == 0:
            return

        sample_period = float(np.median(dt))
        durations = np.asarray(run_lengths, dtype=float) * sample_period
        # Filtrar rachas muy cortas (ruido de 1 muestra)
        durations = durations[durations >= 2.0 * sample_period]

        if len(durations) == 0:
            print("[Processor] Ningun parpadeo de duracion suficiente en calibracion.")
            return

        mean_dur = float(np.mean(durations))
        std_dur  = float(np.std(durations))
        min_dur  = max(2.0 * sample_period, mean_dur - 2.0 * std_dur)
        max_dur  = mean_dur + 2.0 * std_dur

        self.blink_zero_calibration = {
            "mean_duration_s":   mean_dur,
            "std_duration_s":    std_dur,
            "min_duration_s":    min_dur,
            "max_duration_s":    max_dur,
            "sample_period_s":   sample_period,
            "n_blinks_used":     int(len(durations)),
        }

        self._apply_blink_zero_calibration_to_et_pipeline()
        print(f"[Processor] Calibracion blink: {self.blink_zero_calibration}")

    def _zero_run_lengths(self, mask: np.ndarray) -> list[int]:
        lengths = []
        current = 0
        for value in mask:
            if bool(value):
                current += 1
            else:
                if current > 0:
                    lengths.append(current)
                current = 0
        if current > 0:
            lengths.append(current)
        return lengths

    def _apply_blink_zero_calibration_to_et_pipeline(self):
        if self.et_pipeline is None or self.blink_zero_calibration is None:
            return

        calib = self.blink_zero_calibration

        if hasattr(self.et_pipeline, "set_blink_zero_calibration"):
            self.et_pipeline.set_blink_zero_calibration(calib)
            return

        for attr, key in [
            ("blink_zero_mean_duration_s", "mean_duration_s"),
            ("blink_zero_min_duration_s",  "min_duration_s"),
            ("blink_zero_max_duration_s",  "max_duration_s"),
        ]:
            if hasattr(self.et_pipeline, attr):
                setattr(self.et_pipeline, attr, calib[key])

    # ------------------------------------------------------------------
    # Metricas ET recientes
    # ------------------------------------------------------------------

    def _update_recent_et_metrics(self):
        if self.et_pipeline is None:
            return

        # NUEVO v3.4: durante la calibracion no tiene sentido calcular las
        # metricas lentas que aplican interpolacion de huecos en la senyal
        # pupilar. La calibracion es el periodo donde se establecen las
        # referencias (ASR baseline, ICA mixing, pupil baseline); interpolar
        # antes de tenerlas es ruido en pantalla. Una vez la calibracion
        # con ojos abiertos termina, _full_calibration_done = True y se
        # habilitan estas metricas.
        if not self._full_calibration_done:
            return

        recent_et = self.et_pipeline.get_recent_metrics(last_n_seconds=60.0)
        self.state.blink_rate_per_min = float(recent_et.get("blink_rate_per_min", 0.0))
        self.state.blink_count = int(recent_et.get("blink_count_total", 0))

    # ------------------------------------------------------------------
    # Workload online
    # ------------------------------------------------------------------

    def _update_online_workload(self):
        if (
            self.eeg_pipeline is None
            or not self.eeg_pipeline._warmup_done
            or self.eeg_pipeline.buffer_duration < PROCESS_WINDOW_SECONDS
        ):
            return

        t_end   = float(self.eeg_pipeline._buffer_ts[-1]) if self.eeg_pipeline._buffer_ts else 0.0
        t_start = max(0.0, t_end - PROCESS_WINDOW_SECONDS)
        eeg_epoch = self.eeg_pipeline.extract_epoch(t_start, t_end)

        if len(eeg_epoch) < 128:
            return

        band_dict = self.eeg_pipeline.band_powers(eeg_epoch, EEG_BANDS)
        self.state.band_powers = {
            "Theta":    band_dict.get("Theta",    0.0),
            "Alpha":    band_dict.get("Alpha",    0.0),
            "LowBeta":  band_dict.get("LowBeta",  0.0),
            "HighBeta": band_dict.get("HighBeta", 0.0),
            "Gamma":    band_dict.get("Gamma",    0.0),
        }

        if self.workload_model is not None:
            raw_workload = self.workload_model.predict(eeg_epoch)

            if self._current_calibration_kind == "open":
                self._calib_workload_open_values.append(float(raw_workload))
            elif self._current_calibration_kind == "closed":
                self._calib_workload_closed_values.append(float(raw_workload))

            self.state.workload_value = self._apply_workload_baseline(raw_workload)

    def _finish_workload_baseline(self, kind: str):
        values = (
            self._calib_workload_open_values if kind == "open"
            else self._calib_workload_closed_values
        )
        if not values:
            return

        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return

        baseline = float(np.median(arr))

        if kind == "open":
            self.workload_zero_reference = baseline
        elif self.workload_zero_reference is None:
            self.workload_zero_reference = baseline

        print(
            f"[Processor] Baseline workload {kind}: {baseline:.2f}. "
            f"Ref cero: {self.workload_zero_reference:.2f}"
        )

    def _apply_workload_baseline(self, raw_value: float) -> float:
        raw_value = float(raw_value)
        if self.workload_zero_reference is None:
            return float(np.clip(raw_value, 0.0, 100.0))
        adjusted = raw_value - self.workload_zero_reference
        return float(np.clip(adjusted, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Grabacion de metricas temporales
    # ------------------------------------------------------------------

    def _write_metrics_if_needed(self):
        now_ts = self._get_current_lsl_time_for_metrics()
        if now_ts <= 0:
            return
        if now_ts - self._last_metrics_record_ts >= self._metrics_record_period_s:
            self._write_metrics(now_ts)
            self._last_metrics_record_ts = now_ts

    def _write_metrics(self, timestamp_lsl: float):
        context = self._get_recorder_context()

        trust    = float(self.state.trust_value)    if "Trust"    in self.active_models else None
        workload = float(self.state.workload_value) if "Workload" in self.active_models else None
        fatigue  = float(self.state.fatigue_value)  if "Fatigue"  in self.active_models else None

        self.recorder.write_metrics(
            timestamp_lsl=float(timestamp_lsl),
            trust_value=trust,
            workload_value=workload,
            fatigue_value=fatigue,
            context=context,
        )

    def _get_current_lsl_time_for_metrics(self) -> float:
        candidates = []
        if self.use_eeg and len(self.state.eeg_time) > 0:
            candidates.append(float(self.state.eeg_time[-1]))
        if self.use_et and len(self.state.eye_time) > 0:
            candidates.append(float(self.state.eye_time[-1]))
        return max(candidates) if candidates else 0.0