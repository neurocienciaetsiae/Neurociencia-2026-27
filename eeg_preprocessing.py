"""
Entidad: Universidad Politécnica de Madrid
Proyecto: ATC Trust Project / Arquitectura multimodal para la inferencia de la confianza en la
          automatización en tiempo real mediante la fusión de señales neurofisiológicas en entornos ATC
Código: eeg_preprocessing.py
Autora: Ana Ribote Gutiérrez
Estado: Versión preliminar
Versión: 2.5
Fecha: 06/06/2026

Pipeline EEG online avanzado.

Pasos:
1. Acumulación de muestras en buffer de entrada (robusto a jitter LSL)
2. Downsampling opcional (por defecto desactivado — diadema ya va a 125 Hz)
3. Bandpass 1-45 Hz con filter state persistente (activable/desactivable)
4. Notch 50 Hz con filter state persistente (activable/desactivable)
5. Interpolación simple de canales malos
6. Re-referencia a PZ o CAR
7. NUEVO v2.5: limpieza de parpadeos mediante ICA (si esta entrenado)
8. ASR simplificado online
9. Buffer circular para epoching y ventanas deslizantes

Cambios v2.5:
- Hook de ICA: set_ica_detector(), aplicacion in-line antes de ASR.
- Acumulacion de datos limpios para entrenar ICA durante calibracion
  con ojos abiertos: start_ica_accumulation(), get_ica_accumulated_data(),
  finish_ica_accumulation().
- BUGFIX: timestamps de salida ahora usan los timestamps LSL reales en
  lugar de generarlos sinteticamente con dt=1/fs. Esto permite alinear
  eeg_raw.csv con eeg_preprocessed.csv muestra a muestra.
- Nuevo metodo band_powers_per_channel() que devuelve potencias por canal
  ademas del promedio. Necesario para asimetria alfa frontal (Davidson 1992).

Cambios v2.4:
- Integración con TrialEngine (markers.py v2.0):
  on_calib_start/on_calib_end conectados a start_calibration/finish_calibration.
  on_baseline_end extrae epoch y computa band_powers automáticamente.
  on_confidence_response registra la condición HIGH/LOW del trial.

Cambios v2.3 mantenidos íntegramente:
- fs_in/fs_out actualizados a 125 Hz.
- apply_bandpass=False, apply_notch=True, apply_rereference=False por defecto.
- band_powers: mínimo bajado de 256 a 128 muestras (1s a 125 Hz).
- Umbrales más permisivos: flat_threshold=0.1, noise_threshold=10.0.
- chunk_min_samples=16, bad_ch_min_samples=125.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, welch
from scipy.signal import tf2sos


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class EEGPipelineConfig:
    fs_in: float = 125.0
    fs_out: float = 125.0
    hp: float = 1.0
    lp: float = 45.0
    notch: float = 50.0
    notch_q: float = 30.0
    ref_channel: str = "PZ"
    window_s: float = 0.500
    overlap: float = 0.50
    buffer_s: float = 30.0
    asr_threshold: float = 10.0
    filter_warmup_s: float = 5.0

    apply_bandpass: bool = False
    apply_notch: bool = True
    apply_rereference: bool = False

    flat_threshold: float = 0.1
    noise_threshold: float = 10.0

    chunk_min_samples: int = 16
    bad_ch_min_samples: int = 125


# ---------------------------------------------------------------------------
# Downsampler
# ---------------------------------------------------------------------------

class Downsampler:
    def __init__(self, fs_in: float, fs_out: float):
        self.factor = int(round(fs_in / fs_out))
        if self.factor < 1:
            self.factor = 1

        nyq = fs_in / 2.0
        cutoff = (fs_out / 2.0 * 0.9) / nyq
        cutoff = min(0.99, max(0.01, cutoff))
        self._sos = butter(8, cutoff, btype="low", output="sos")
        self._zi: Optional[np.ndarray] = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return chunk

        if self._zi is None:
            zi_1d = sosfilt_zi(self._sos)
            self._zi = np.repeat(zi_1d[:, :, np.newaxis], chunk.shape[1], axis=2)

        filtered, self._zi = sosfilt(self._sos, chunk, axis=0, zi=self._zi)
        return filtered[::self.factor]


# ---------------------------------------------------------------------------
# Filtro causal genérico
# ---------------------------------------------------------------------------

class CausalFilter:
    def __init__(self, sos: np.ndarray):
        self._sos = sos
        self._zi: Optional[np.ndarray] = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return chunk

        if self._zi is None:
            zi_1d = sosfilt_zi(self._sos)
            self._zi = np.repeat(zi_1d[:, :, np.newaxis], chunk.shape[1], axis=2)

        out, self._zi = sosfilt(self._sos, chunk, axis=0, zi=self._zi)
        return out


def make_bandpass_filter(fs: float, hp: float, lp: float, order: int = 4) -> CausalFilter:
    nyq = fs / 2.0
    low = max(1e-4, hp / nyq)
    high = min(0.999, lp / nyq)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return CausalFilter(sos)


def make_notch_filter(fs: float, freq: float, q: float) -> CausalFilter:
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, q)
    sos = tf2sos(b, a)
    return CausalFilter(sos)


# ---------------------------------------------------------------------------
# ASR simplificado
# ---------------------------------------------------------------------------

class ASRProcessor:
    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._calibrated = False

    def calibrate(self, clean_data: np.ndarray):
        if clean_data.size == 0:
            return

        self._mean = np.nanmean(clean_data, axis=0)
        self._std = np.nanstd(clean_data, axis=0)
        self._std[self._std < 1e-6] = 1e-6
        self._calibrated = True
        print(f"[ASR] Calibrado. Std media por canal: {np.round(self._std, 3)}")

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not self._calibrated or self._mean is None or self._std is None:
            return chunk

        limit = self.threshold * self._std
        return np.clip(chunk, self._mean - limit, self._mean + limit)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


# ---------------------------------------------------------------------------
# Detector e interpolador de canales malos
# ---------------------------------------------------------------------------

class BadChannelDetector:
    def __init__(self, flat_threshold: float = 0.1, noise_threshold: float = 10.0):
        self.flat_threshold = flat_threshold
        self.noise_threshold = noise_threshold
        self.bad_channels: List[int] = []
        self.bad_channel_names: List[str] = []

    def detect(
        self,
        data: np.ndarray,
        channel_names: List[str],
        min_samples: int = 128,
    ) -> List[str]:
        if data.size == 0 or len(data) < min_samples:
            return self.bad_channel_names

        std_per_ch = np.nanstd(data, axis=0)
        median_std = np.nanmedian(std_per_ch)

        bad_idx = []
        for i, std in enumerate(std_per_ch):
            if std < self.flat_threshold:
                print(f"[BadCh] Canal plano: {channel_names[i]} (std={std:.3f})")
                bad_idx.append(i)
            elif median_std > 0 and std > self.noise_threshold * median_std:
                print(
                    f"[BadCh] Canal ruidoso: {channel_names[i]} "
                    f"(std={std:.3f}, mediana={median_std:.3f})"
                )
                bad_idx.append(i)

        self.bad_channels = bad_idx
        self.bad_channel_names = [channel_names[i] for i in bad_idx]
        return self.bad_channel_names

    def interpolate(self, chunk: np.ndarray) -> np.ndarray:
        if not self.bad_channels or chunk.size == 0:
            return chunk

        result = chunk.copy()
        n_ch = chunk.shape[1]

        for bad_idx in self.bad_channels:
            neighbors = [
                i for i in range(n_ch)
                if i != bad_idx and i not in self.bad_channels
            ]
            if neighbors:
                result[:, bad_idx] = np.nanmean(chunk[:, neighbors], axis=1)

        return result


# ---------------------------------------------------------------------------
# Re-referencia
# ---------------------------------------------------------------------------

def apply_rereference(chunk: np.ndarray, ref_idx: int) -> np.ndarray:
    ref = chunk[:, ref_idx:ref_idx + 1]
    return chunk - ref


def apply_car(chunk: np.ndarray, exclude_idx: List[int] | None = None) -> np.ndarray:
    n_ch = chunk.shape[1]
    exclude_idx = exclude_idx or []
    include = [i for i in range(n_ch) if i not in exclude_idx]
    if not include:
        return chunk
    avg = np.nanmean(chunk[:, include], axis=1, keepdims=True)
    return chunk - avg


# ---------------------------------------------------------------------------
# Ventanas deslizantes
# ---------------------------------------------------------------------------

def sliding_windows(
    data: np.ndarray,
    fs: float,
    window_s: float = 0.500,
    overlap: float = 0.50,
) -> List[np.ndarray]:
    win_samples = int(window_s * fs)
    hop_samples = int(win_samples * (1.0 - overlap))

    if hop_samples <= 0:
        hop_samples = 1

    n_samples = len(data)
    if n_samples < win_samples:
        return []

    windows = []
    start = 0
    while start + win_samples <= n_samples:
        windows.append(data[start:start + win_samples])
        start += hop_samples

    return windows


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class EEGPipeline:
    def __init__(self, config: EEGPipelineConfig, channel_names: List[str]):
        self.config = config
        self.channel_names = channel_names
        self.fs_out = config.fs_out
        self.n_ch = len(channel_names)

        if config.ref_channel in channel_names:
            self._ref_idx = channel_names.index(config.ref_channel)
        else:
            self._ref_idx = None
            print(
                f"[EEGPipeline] Canal de referencia '{config.ref_channel}' "
                "no encontrado. Usando CAR."
            )

        self._non_eeg = [
            i for i, ch in enumerate(channel_names) if ch in ("GND", "REF")
        ]

        self._downsampler = Downsampler(config.fs_in, config.fs_out)
        self._bp_filter = make_bandpass_filter(config.fs_out, config.hp, config.lp)
        self._notch = make_notch_filter(config.fs_out, config.notch, config.notch_q)
        self._asr = ASRProcessor(threshold=config.asr_threshold)
        self._bad_ch = BadChannelDetector(
            flat_threshold=config.flat_threshold,
            noise_threshold=config.noise_threshold,
        )

        buf_samples = int(config.buffer_s * config.fs_out)
        self._buffer_data: Deque[np.ndarray] = deque(maxlen=buf_samples)
        self._buffer_ts: Deque[float] = deque(maxlen=buf_samples)

        self._input_buf: List[np.ndarray] = []
        self._input_ts_buf: List[float] = []
        self._chunk_min_samples: int = max(1, config.chunk_min_samples)

        self._calib_data: List[np.ndarray] = []
        self._is_calibrating = False
        self._calibrated = False
        self._warmup_done = False
        self._samples_seen = 0
        self._warmup_samples = int(config.filter_warmup_s * config.fs_out)

        self._t_last_lsl: float = 0.0
        self._t_next_out: Optional[float] = None

        self.confirmed_bad_channels: List[str] = []
        self._confirmed_bad_idx: List[int] = []

        # Bandas por defecto para band_powers en baseline
        self.default_bands: Dict[str, tuple[float, float]] = {
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta":  (13.0, 30.0),
        }

        # Callback opcional: se llama al final de cada baseline con
        # (trial_info, band_powers_dict)
        self.on_baseline_powers: Callable | None = None

        # NUEVO v2.5: ICA externo para limpieza de parpadeos.
        # Se ajusta fuera del pipeline (en processor.py al final de la
        # calibracion con ojos abiertos) y se inyecta aqui. Si esta
        # entrenado y self._ica_clean_inline es True, cada chunk se
        # limpia tras la re-referencia y antes del ASR.
        self._ica_detector = None
        self._ica_clean_inline: bool = True

        # Acumulador de datos para entrenar ICA fuera del pipeline.
        # Cuando _is_accumulating_ica es True, cada chunk procesado se
        # guarda. processor.py lo activa al iniciar la calibracion con
        # ojos abiertos y recupera los datos al terminarla.
        self._ica_accum: List[np.ndarray] = []
        self._is_accumulating_ica: bool = False

        # Cola de eventos de parpadeo detectados por ICA. Se llena en
        # _process_block tras la limpieza ICA y se vacia desde processor.py
        # con consume_ica_blink_events().
        self._pending_ica_blinks: list = []

    # ------------------------------------------------------------------
    # Integración con TrialEngine (markers.py v2.0)
    # ------------------------------------------------------------------

    def bind_to_trial_engine(self, engine) -> None:
        """
        Conecta los callbacks del TrialEngine al pipeline EEG.

        Uso:
            pipeline.bind_to_trial_engine(engine)

        Equivale a:
            engine.on_calib_start   → pipeline.start_calibration()
            engine.on_calib_end     → pipeline.finish_calibration()
            engine.on_baseline_end  → pipeline extrae epoch y computa band_powers
            engine.on_confidence_response → pipeline registra condición HIGH/LOW
        """
        engine.on_calib_start = self._on_calib_start
        engine.on_calib_end   = self._on_calib_end
        engine.on_baseline_end = self._on_baseline_end
        engine.on_confidence_response = self._on_confidence_response

    def _on_calib_start(self, calib_type: str, timestamp: float) -> None:
        print(f"[EEGPipeline] Calibración iniciada ({calib_type}).")
        self.start_calibration()

    def _on_calib_end(self, calib_type: str, timestamp: float) -> None:
        print(f"[EEGPipeline] Calibración finalizada ({calib_type}).")
        bad = self.finish_calibration()
        if bad:
            print(f"[EEGPipeline] Canales malos detectados: {bad}")

    def _on_baseline_end(self, trial_info, timestamp: float) -> None:
        """
        Al recibir FIXATION_CROSS_END extrae el epoch del baseline
        y computa las potencias de banda.
        """
        epoch = self.extract_epoch(
            trial_info.t_baseline_start,
            trial_info.t_baseline_end,
        )

        if len(epoch) == 0:
            print(
                f"[EEGPipeline] Epoch vacío — "
                f"B{trial_info.block} T{trial_info.trial}"
            )
            return

        powers = self.band_powers(epoch, self.default_bands)
        print(
            f"[EEGPipeline] Band powers — "
            f"B{trial_info.block} T{trial_info.trial}: {powers}"
        )

        if self.on_baseline_powers:
            self.on_baseline_powers(trial_info, powers)

    def _on_confidence_response(
        self,
        trial_info,
        condition: str,
        confidence: float,
        timestamp: float,
    ) -> None:
        """
        Registra la condición y el valor de confianza del trial.
        """
        print(
            f"[EEGPipeline] Confianza {condition} — "
            f"B{trial_info.block} T{trial_info.trial}: {confidence}"
        )

    # ------------------------------------------------------------------
    # Calibración
    # ------------------------------------------------------------------

    def start_calibration(self):
        self._is_calibrating = True
        self._calib_data = []
        print("[EEGPipeline] Calibración iniciada.")

    def finish_calibration(self) -> List[str]:
        self._is_calibrating = False

        if not self._calib_data:
            print("[EEGPipeline] Sin datos de calibración.")
            return []

        calib_arr = np.concatenate(self._calib_data, axis=0)
        print(f"[EEGPipeline] Calibración: {len(calib_arr)} muestras a {self.fs_out} Hz.")

        bad = self._bad_ch.detect(
            calib_arr,
            self.channel_names,
            min_samples=self.config.bad_ch_min_samples,
        )

        self._asr.calibrate(calib_arr)
        self._calibrated = True
        print(f"[EEGPipeline] Canales malos detectados: {bad}")
        return bad

    def confirm_bad_channels(self, bad_names: List[str]):
        self.confirmed_bad_channels = bad_names
        self._confirmed_bad_idx = [
            i for i, ch in enumerate(self.channel_names) if ch in bad_names
        ]
        self._bad_ch.bad_channels = self._confirmed_bad_idx
        self._bad_ch.bad_channel_names = bad_names
        print(f"[EEGPipeline] Canales malos confirmados: {bad_names}")

    # ------------------------------------------------------------------
    # NUEVO v2.5: Gestion del detector ICA
    # ------------------------------------------------------------------

    def set_ica_detector(self, detector, clean_inline: bool = True) -> None:
        """
        Inyecta un detector ICA externo (BlinkICADetector) ya ajustado.

        Si clean_inline es True, cada chunk se limpia inline tras la
        re-referencia y antes del ASR (orden: filtros -> bad channels ->
        rereference -> ICA clean -> ASR -> buffer).

        Si el detector no esta entrenado (is_fitted=False), la inyeccion
        no tiene efecto.
        """
        self._ica_detector = detector
        self._ica_clean_inline = bool(clean_inline)

        if detector is not None and getattr(detector, "is_fitted", False):
            print(f"[EEGPipeline] ICA conectado (clean_inline={self._ica_clean_inline}).")
        else:
            print("[EEGPipeline] ICA conectado pero no entrenado todavia.")

    def start_ica_accumulation(self) -> None:
        """
        Inicia la acumulacion de chunks procesados (post-rereferencia,
        pre-ASR) para entrenar ICA al terminar la fase de ojos abiertos.

        Tras llamar a esto, cada chunk procesado se guarda en self._ica_accum
        hasta que se llame a finish_ica_accumulation().
        """
        self._ica_accum.clear()
        self._is_accumulating_ica = True
        print("[EEGPipeline] Inicio acumulacion ICA.")

    def finish_ica_accumulation(self) -> np.ndarray:
        """
        Termina la acumulacion y devuelve los datos acumulados como
        ndarray (n_samples, n_channels). Vacia el buffer interno.
        Devuelve un array vacio si no hay datos.
        """
        self._is_accumulating_ica = False

        if not self._ica_accum:
            print("[EEGPipeline] Acumulacion ICA terminada sin datos.")
            return np.empty((0, self.n_ch), dtype=float)

        data = np.concatenate(self._ica_accum, axis=0)
        self._ica_accum.clear()
        print(f"[EEGPipeline] Acumulacion ICA terminada: {data.shape[0]} muestras.")
        return data

    @property
    def has_ica(self) -> bool:
        return (
            self._ica_detector is not None
            and getattr(self._ica_detector, "is_fitted", False)
        )

    def consume_ica_blink_events(self) -> list:
        """
        Devuelve los eventos de parpadeo detectados por ICA desde la
        ultima llamada y vacia la cola. Cada evento es un BlinkEvent con
        atributos: timestamp_start, timestamp_end, duration_s, peak_amplitude.

        Llamar periodicamente desde processor.py (idealmente despues de
        cada push_chunk de EEG) para no acumular memoria.
        """
        if not self._pending_ica_blinks:
            return []
        events = list(self._pending_ica_blinks)
        self._pending_ica_blinks.clear()
        return events

    # ------------------------------------------------------------------
    # Entrada principal
    # ------------------------------------------------------------------

    def push_chunk(
        self,
        chunk: np.ndarray,
        timestamps: List[float] | None = None,
    ) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)

        if chunk.size == 0:
            return np.empty((0, self.n_ch), dtype=float)

        n_in = len(chunk)

        if timestamps is not None and len(timestamps) > 0:
            self._t_last_lsl = float(timestamps[-1])
            if self._t_next_out is None:
                self._t_next_out = float(timestamps[0])

        if self._t_next_out is None:
            self._t_next_out = 0.0

        # CORREGIDO v2.5: guardar timestamps LSL reales (antes se guardaban 0.0
        # y luego se generaban sinteticamente, lo que rompia la alineacion entre
        # eeg_raw.csv y eeg_preprocessed.csv).
        dt_out = 1.0 / self.fs_out
        for i in range(n_in):
            self._input_buf.append(chunk[i])
            if timestamps is not None and i < len(timestamps):
                self._input_ts_buf.append(float(timestamps[i]))
            else:
                # Fallback: extrapolar a partir del ultimo conocido
                last_ts = (
                    self._input_ts_buf[-1] if self._input_ts_buf
                    else self._t_next_out
                )
                self._input_ts_buf.append(last_ts + dt_out)

        if len(self._input_buf) < self._chunk_min_samples:
            return np.empty((0, self.n_ch), dtype=float)

        accumulated = np.array(self._input_buf, dtype=float)
        accumulated_ts = list(self._input_ts_buf)
        self._input_buf.clear()
        self._input_ts_buf.clear()

        return self._process_block(accumulated, accumulated_ts)

    def _process_block(
        self,
        block: np.ndarray,
        ts_in: List[float] | None = None,
    ) -> np.ndarray:
        # 1. Downsampling
        block_ds = self._downsampler.process(block)
        if len(block_ds) == 0:
            return block_ds

        n = len(block_ds)

        # 2. Timestamps de salida: usar los reales del downsampling
        # CORREGIDO v2.5: derivar ts_out de ts_in en lugar de generarlos
        # sinteticamente con dt=1/fs.
        dt_out = 1.0 / self.fs_out
        factor = self._downsampler.factor

        if ts_in is not None and len(ts_in) > 0:
            ts_arr = np.asarray(ts_in, dtype=float)
            ts_out_arr = ts_arr[::factor][:n]
            ts_out = list(ts_out_arr)
            # Padding por si edge effects dejan ts_out mas corto que n
            while len(ts_out) < n:
                last = ts_out[-1] if ts_out else self._t_next_out
                ts_out.append(last + dt_out)
        else:
            ts_out = [self._t_next_out + i * dt_out for i in range(n)]

        self._t_next_out = ts_out[-1] + dt_out

        # 3. Filtrado
        chunk_filt = (
            self._bp_filter.process(block_ds)
            if self.config.apply_bandpass
            else block_ds
        )
        chunk_filt = (
            self._notch.process(chunk_filt)
            if self.config.apply_notch
            else chunk_filt
        )

        # 4. Warmup
        self._samples_seen += n
        if not self._warmup_done and self._samples_seen >= self._warmup_samples:
            self._warmup_done = True
            print(f"[EEGPipeline] Warmup completado ({self.config.filter_warmup_s}s).")

        if not self._warmup_done:
            for i in range(n):
                self._buffer_data.append(chunk_filt[i])
                self._buffer_ts.append(ts_out[i])
            return chunk_filt

        # 5. Interpolación de canales malos
        chunk_interp = self._bad_ch.interpolate(chunk_filt)

        # 6. Re-referencia
        if self.config.apply_rereference:
            if self._ref_idx is not None:
                chunk_ref = apply_rereference(chunk_interp, self._ref_idx)
            else:
                chunk_ref = apply_car(
                    chunk_interp,
                    exclude_idx=self._non_eeg + self._confirmed_bad_idx,
                )
        else:
            chunk_ref = chunk_interp

        # 7. NUEVO v2.5: limpieza de parpadeos por ICA antes del ASR.
        # Razon del orden: ICA necesita ver el parpadeo con su amplitud
        # completa para separarlo bien. Si ASR ya lo recortara, ICA
        # podria no identificarlo.
        if (
            self._ica_clean_inline
            and self._ica_detector is not None
            and getattr(self._ica_detector, "is_fitted", False)
        ):
            # 7a. Detectar eventos de parpadeo (con timestamps).
            # process() devuelve lista de BlinkEvent con start/end ts.
            try:
                ica_events = self._ica_detector.process(chunk_ref, ts_out)
                if ica_events:
                    self._pending_ica_blinks.extend(ica_events)
            except Exception as exc:
                print(f"[EEGPipeline] Error detectando blinks por ICA: {exc}")

            # 7b. Limpiar la senyal (poner a 0 la componente del parpadeo).
            chunk_post_ica = self._ica_detector.clean(chunk_ref)
        else:
            chunk_post_ica = chunk_ref

        # 8. ASR (sobre senal ya limpia de parpadeos si ICA estaba activo)
        chunk_clean = self._asr.process(chunk_post_ica)

        # 9. Buffer circular
        for i in range(n):
            self._buffer_data.append(chunk_clean[i])
            self._buffer_ts.append(ts_out[i])

        # 10. Acumular para calibración ASR (ojos cerrados)
        if self._is_calibrating:
            self._calib_data.append(chunk_clean)

        # 11. NUEVO v2.5: acumular para entrenamiento ICA (ojos abiertos).
        # Se guarda chunk_ref (pre-ICA, pre-ASR) porque ICA debe aprender
        # con los parpadeos visibles en su amplitud original.
        if self._is_accumulating_ica:
            self._ica_accum.append(chunk_ref)

        return chunk_clean

    # ------------------------------------------------------------------
    # Epoching
    # ------------------------------------------------------------------

    def extract_epoch(self, t_start: float, t_end: float) -> np.ndarray:
        ts_arr = np.asarray(self._buffer_ts, dtype=float)
        data_arr = np.asarray(self._buffer_data, dtype=float)

        if len(ts_arr) == 0 or len(data_arr) == 0:
            return np.empty((0, self.n_ch), dtype=float)

        mask = (ts_arr >= t_start) & (ts_arr <= t_end)
        if not np.any(mask):
            return np.empty((0, self.n_ch), dtype=float)

        return data_arr[mask]

    def get_sliding_windows(self, t_start: float, t_end: float) -> List[np.ndarray]:
        epoch = self.extract_epoch(t_start, t_end)
        if len(epoch) == 0:
            return []
        return sliding_windows(epoch, self.fs_out, self.config.window_s, self.config.overlap)

    # ------------------------------------------------------------------
    # Potencias de banda
    # ------------------------------------------------------------------

    def band_powers(
        self,
        eeg_block: np.ndarray,
        bands: Dict[str, tuple[float, float]],
    ) -> Dict[str, float]:
        x = np.asarray(eeg_block, dtype=float)

        if x.ndim != 2 or len(x) < 128:
            return {name: 0.0 for name in bands}

        valid_idx = [
            i for i, ch in enumerate(self.channel_names)
            if ch not in ("GND", "REF")
        ]
        if not valid_idx:
            return {name: 0.0 for name in bands}

        x = x[:, valid_idx]

        freqs, psd = welch(x, fs=self.fs_out, nperseg=min(len(x), 128), axis=0)

        out: Dict[str, float] = {}
        for name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            if not np.any(mask):
                out[name] = 0.0
                continue
            band_power = np.trapz(psd[mask, :], freqs[mask], axis=0)
            out[name] = float(np.nanmean(band_power))

        return out

    def band_powers_per_channel(
        self,
        eeg_block: np.ndarray,
        bands: Dict[str, tuple[float, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        NUEVO v2.5: devuelve potencias de banda POR CANAL en lugar de promediadas.

        Necesario para asimetria alfa frontal (Davidson 1992):
            alpha_asym = ln(alpha_AF4) - ln(alpha_AF3)

        Devuelve:
            {
              "Theta":    {"AF7": 1.23, "AF3": 1.45, ...},
              "Alpha":    {"AF7": 2.10, "AF3": 1.98, ...},
              ...
            }
        Canales malos confirmados se devuelven como 0.0.
        """
        empty = {name: {ch: 0.0 for ch in self.channel_names} for name in bands}
        x = np.asarray(eeg_block, dtype=float)

        if x.ndim != 2 or len(x) < 128:
            return empty

        freqs, psd = welch(x, fs=self.fs_out, nperseg=min(len(x), 128), axis=0)

        out: Dict[str, Dict[str, float]] = {}
        for name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            if not np.any(mask):
                out[name] = {ch: 0.0 for ch in self.channel_names}
                continue
            band_power = np.trapz(psd[mask, :], freqs[mask], axis=0)  # (n_ch,)
            out[name] = {
                ch: float(band_power[i]) if i < len(band_power) and np.isfinite(band_power[i]) else 0.0
                for i, ch in enumerate(self.channel_names)
            }

        return out

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def buffer_duration(self) -> float:
        if not self._buffer_ts:
            return 0.0
        return float(self._buffer_ts[-1] - self._buffer_ts[0])

    def reset(self):
        self._input_buf.clear()
        self._input_ts_buf.clear()
        self._buffer_data.clear()
        self._buffer_ts.clear()
        self._calib_data.clear()
        self._is_calibrating = False
        self._calibrated = False
        self._warmup_done = False
        self._samples_seen = 0
        self._t_next_out = None
        # Limpiar acumulador ICA pero NO el detector entrenado:
        # tras un reset por WELCOME, queremos conservar el ICA aprendido.
        self._ica_accum.clear()
        self._is_accumulating_ica = False
        # Limpiar tambien la cola de eventos pendientes para no arrastrar
        # detecciones previas al WELCOME.
        self._pending_ica_blinks.clear()
        print("[EEGPipeline] Pipeline reiniciado.")