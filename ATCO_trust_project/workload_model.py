"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: workload_model.py
Autora: Ana Ribote Gutierrez
Version: 2.0
Fecha: 27/05/2026

Modelo de carga de trabajo cognitiva (Workload) basado en EEG.

Referencias y fundamentacion cientifica:
-----------------------------------------
El indice de referencia para workload cognitivo con EEG es el
Theta/Alpha Ratio (TAR) frontal:

  TAR = theta_frontal / alpha_parietal

Gevins et al. (1997, 1998): en tareas de memoria de trabajo (n-back),
el theta frontal aumenta y el alpha posterior disminuye de forma
proporcional a la carga cognitiva. El TAR discrimina fiablemente
niveles de carga.

Klimesch (1999, Neuroscience & Biobehavioral Reviews): el alpha
parietal refleja el estado de procesamiento cortical; se suprime
(desincronizacion) cuando aumenta el procesamiento activo.

Onton et al. (2005, NeuroImage): en controladores de trafico aereo,
el theta frontal midline (Fz) es el marcador mas robusto de workload.

Strangman et al. (2014): el TAR es el indice EEG de workload mas
replicado en entornos operacionales reales.

Formula implementada:
  W_raw = w_tar * tanh(TAR / TAR_ref) + w_lbeta * tanh(LowBeta_frontal / beta_ref)
  W_norm = clip(50 + 50 * (W_raw - W_baseline) / W_range, 0, 100)

Senales usadas:
  - EEG: theta frontal (AF7, AF3, AFZ, AF4, AF8)
  - EEG: alpha parietal (P3, PZ, P4)
  - EEG: low beta frontal (12-20 Hz) como componente secundario

Normalizacion: 0-100 (percentil relativo al baseline de calibracion).

Cambios v2.0:
- Modelo reescrito completamente con base en Gevins 1997/1998.
- Indice principal: TAR = theta_frontal / alpha_parietal.
- Componente secundario: LowBeta frontal (procesamiento activo).
- La contribucion de HighBeta se elimina: en ATC el HighBeta refleja
  mayoritariamente artefactos musculares (masticacion, tension mandibular)
  mas que carga cognitiva pura.
- Baseline con datos de calibracion para normalizacion relativa.
- Suavizado EMA ajustado a alpha=0.20 (ventana efectiva ~5 actualizaciones).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


class WorkloadModel:
    """
    Indice de workload cognitivo basado en el Theta/Alpha Ratio (TAR) frontal.

    TAR = theta_frontal_mean / alpha_parietal_mean

    Se normaliza respecto al baseline de calibracion (ojos abiertos en reposo)
    para obtener un valor relativo al estado base de cada participante.
    """

    # Canales frontales para theta y low_beta
    FRONTAL_CH = ["AF7", "AF3", "AFZ", "AF4", "AF8"]
    # Canales parietales para alpha
    PARIETAL_CH = ["P3", "PZ", "P4"]

    def __init__(self, fs: float, channel_names: list[str]):
        self.fs = float(fs)
        self.channel_names = list(channel_names)
        self._upper_names = [ch.upper() for ch in channel_names]

        self.value = 50.0

        # Suavizado EMA: alpha=0.20 → cada nuevo valor pesa un 20 %
        self._alpha_ema = 0.20

        # Baseline para normalizacion: median(TAR durante calibracion en reposo)
        self.baseline_tar: float | None = None
        # Rango esperado de variacion del TAR (para escalar 0-100)
        # Se estima durante calibracion como 3 * std(TAR) en reposo.
        self._tar_range: float = 1.0

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def set_baseline(self, baseline_value: float):
        """
        Recibe el TAR mediano durante la calibracion en reposo.
        Llamar desde processor._finish_workload_baseline.
        """
        try:
            v = float(baseline_value)
        except Exception:
            return
        if np.isfinite(v) and v > 0:
            self.baseline_tar = v

    def predict(self, eeg_epoch: np.ndarray,
                apply_internal_baseline: bool = False) -> float:
        """
        Calcula el indice de workload a partir de un epoch EEG.

        Parametros
        ----------
        eeg_epoch : ndarray (n_samples, n_channels)
            EEG ya preprocesado (filtrado, artefactos eliminados).
        apply_internal_baseline : bool
            Si True, resta el baseline almacenado internamente.
            En tiempo real normalmente se gestiona desde processor.py.

        Devuelve
        --------
        float en [0, 100]
        """
        eeg_epoch = np.asarray(eeg_epoch, dtype=float)

        if eeg_epoch.ndim != 2 or eeg_epoch.shape[0] < 16:
            return self.value

        eeg_epoch = np.nan_to_num(eeg_epoch, nan=0.0, posinf=0.0, neginf=0.0)

        powers = self._band_powers_by_channel(eeg_epoch)

        theta_frontal   = self._channel_mean(powers["theta"],    self.FRONTAL_CH)
        alpha_parietal  = self._channel_mean(powers["alpha"],    self.PARIETAL_CH)
        low_beta_frontal = self._channel_mean(powers["low_beta"], self.FRONTAL_CH)

        # TAR: indice principal (Gevins 1997)
        tar = theta_frontal / (alpha_parietal + 1e-9)

        # Componente low_beta: procesamiento cognitivo activo (peso menor)
        lb_norm = np.tanh(low_beta_frontal / (theta_frontal + 1e-9))

        # Score bruto: TAR normalizado + contribucion LowBeta
        # TAR en reposo suele ser ~0.5-1.5; en alta carga sube a 2-5
        tar_score = float(np.tanh(tar / 2.0))  # 0-1 saturado

        raw = float(np.clip(
            100.0 * (0.75 * tar_score + 0.25 * lb_norm),
            0.0, 100.0
        ))

        if apply_internal_baseline and self.baseline_tar is not None:
            tar_centered = tar - self.baseline_tar
            raw = float(np.clip(
                50.0 + 50.0 * np.tanh(tar_centered / (self._tar_range + 1e-9)),
                0.0, 100.0
            ))

        # EMA suavizante
        self.value = (1.0 - self._alpha_ema) * self.value + self._alpha_ema * raw
        self.value = float(np.clip(self.value, 0.0, 100.0))

        return self.value

    # ------------------------------------------------------------------
    # Internos: calculo de potencias
    # ------------------------------------------------------------------

    def _band_powers_by_channel(self, eeg: np.ndarray) -> dict[str, np.ndarray]:
        n = eeg.shape[0]
        # Centrado DC
        eeg = eeg - np.nanmean(eeg, axis=0, keepdims=True)
        window = np.hanning(n).reshape(-1, 1)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        fft = np.fft.rfft(eeg * window, axis=0)
        psd = (np.abs(fft) ** 2) / max(1, n)

        def bp(fmin, fmax):
            mask = (freqs >= fmin) & (freqs < fmax)
            if not np.any(mask):
                return np.zeros(psd.shape[1], dtype=float)
            power = np.nanmean(psd[mask, :], axis=0)
            return np.log1p(np.maximum(power, 0.0))

        return {
            "theta":    bp(4.0,  8.0),
            "alpha":    bp(8.0,  12.0),
            "low_beta": bp(12.0, 20.0),
        }

    def _channel_mean(self, powers: np.ndarray, preferred: list[str]) -> float:
        selected = []
        for ch in preferred:
            ch_u = ch.upper()
            if ch_u in self._upper_names:
                idx = self._upper_names.index(ch_u)
                if idx < len(powers) and np.isfinite(powers[idx]):
                    selected.append(float(powers[idx]))
        if not selected:
            valid = powers[np.isfinite(powers)]
            return float(np.mean(valid)) if len(valid) > 0 else 0.0
        return float(np.mean(selected))
