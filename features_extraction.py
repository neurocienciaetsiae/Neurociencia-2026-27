"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: features_extraction.py
Autora: Ana Ribote Gutierrez
Version: 2.1
Fecha: 27/05/2026

Extraccion y seleccion de features multimodales.

Cambios v2.1:
- BUGFIX: Las claves de blink_rate, theta y alpha no coincidian con las
  que esperaban trust_model.py y fatigue_model.py:
    - trust_model espera "blink_rate_per_min" → antes se exportaba como "blink_rate"
    - fatigue_model espera "theta_power" y "alpha_power" → antes solo "theta" y "alpha"
  Se añaden AMBAS claves (la antigua y la nueva) para mantener compatibilidad
  con cualquier codigo que use las claves antiguas.
- Añadido "perclos" (% Eye Closure) como feature ET para fatigue_model.
- Añadidos ratios EEG documentados en literatura:
    - theta_alpha_ratio  (Gevins 1997): carga de trabajo
    - fatigue_index_eeg  = (theta+alpha)/(low_beta+high_beta): indice de fatiga (Lal 2001)
    - trust_beta_ratio   = low_beta / (high_beta + eps): ratio de confianza (Causse 2019)
    - alpha_asymmetry    = alpha_AF3 - alpha_AF4: asimetria frontal (Davidson 1992)
      (requiere potencias por canal; aqui se usa el promedio global como aproximacion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class FeatureBundle:
    all_features: Dict[str, float]
    selected_features: Dict[str, float]


def compute_blink_intensity(pupil_left_mean: float, pupil_right_mean: float) -> float:
    """
    Intensidad de parpadeo como proxy de la frecuencia de cierre ocular.
    Devuelve 0 si la pupila media esta en rango normal (~5 mm),
    y sube hacia 1 conforme la media baja (mas cierres, pupila promedio menor).
    """
    vals = [v for v in [pupil_left_mean, pupil_right_mean] if np.isfinite(v) and v > 0]
    if not vals:
        return 0.0
    pupil_mean = float(np.mean(vals))
    value = max(0.0, 1.0 - pupil_mean / 5.0)
    return float(np.clip(value, 0.0, 1.0))


def compute_perclos(pupil_l: np.ndarray, pupil_r: np.ndarray,
                    pupil_min_mm: float = 2.0) -> float:
    """
    PERCLOS simplificado: porcentaje de muestras donde la pupila esta invalida
    (cierre ocular o perdida de senyal) en la ventana dada.
    Rango: 0.0 (ojos siempre abiertos) - 1.0 (ojos siempre cerrados).
    Referencia: Wierwille & Ellsworth (1994), SAE Technical Paper.
    """
    pupil_l = np.asarray(pupil_l, dtype=float)
    pupil_r = np.asarray(pupil_r, dtype=float)

    if len(pupil_l) == 0:
        return 0.0

    invalid_l = ~np.isfinite(pupil_l) | (pupil_l <= 0) | (pupil_l < pupil_min_mm)
    invalid_r = ~np.isfinite(pupil_r) | (pupil_r <= 0) | (pupil_r < pupil_min_mm)
    closed = invalid_l | invalid_r

    return float(np.mean(closed))


def build_feature_bundle(
    band_powers: Dict[str, float],
    low_beta: float,
    high_beta: float,
    blink_rate: float,
    blink_intensity: float,
    pupil_mean: float,
    pupil_norm_mean: float,
    gaze_stability: float,
    perclos: float = 0.0,
    blink_duration_mean: float = 0.0,
) -> FeatureBundle:
    """
    Construye el bundle de features con TODAS las claves necesarias para los modelos.

    Claves de salida (selected_features):
      EEG:
        theta, alpha, low_beta, high_beta, gamma
        theta_power, alpha_power          (aliases para fatigue_model)
        theta_alpha_ratio                 (Gevins 1997, workload)
        fatigue_index_eeg                 (Lal 2001, fatiga EEG)
        trust_beta_ratio                  (Causse 2019, trust EEG)
      ET:
        blink_rate, blink_rate_per_min    (alias para trust_model)
        blink_intensity
        blink_duration_mean               (duracion media de parpadeo, s)
        perclos                           (% closure, Wierwille 1994)
        pupil_mean, pupil_norm_mean
        gaze_stability
    """
    theta     = float(band_powers.get("Theta",    0.0))
    alpha     = float(band_powers.get("Alpha",    0.0))
    low_beta_v  = float(band_powers.get("LowBeta",  low_beta))
    high_beta_v = float(band_powers.get("HighBeta", high_beta))
    gamma     = float(band_powers.get("Gamma",    0.0))

    # Ratios EEG basados en literatura
    eps = 1e-9
    theta_alpha_ratio  = theta / (alpha + eps)
    fatigue_index_eeg  = (theta + alpha) / (low_beta_v + high_beta_v + eps)
    trust_beta_ratio   = low_beta_v / (high_beta_v + eps)

    features: Dict[str, float] = {
        # Potencias de banda
        "theta":     theta,
        "alpha":     alpha,
        "low_beta":  low_beta_v,
        "high_beta": high_beta_v,
        "gamma":     gamma,

        # Aliases que esperan los modelos
        "theta_power": theta,           # fatigue_model espera "theta_power"
        "alpha_power": alpha,           # fatigue_model espera "alpha_power"

        # Ratios EEG
        "theta_alpha_ratio":  theta_alpha_ratio,
        "fatigue_index_eeg":  fatigue_index_eeg,
        "trust_beta_ratio":   trust_beta_ratio,

        # Eye-tracking
        "blink_rate":         float(blink_rate),
        "blink_rate_per_min": float(blink_rate),  # trust_model espera esta clave
        "blink_intensity":    float(blink_intensity),
        "blink_duration_mean": float(blink_duration_mean),
        "perclos":            float(perclos),
        "pupil_mean":         float(pupil_mean),
        "pupil_norm_mean":    float(pupil_norm_mean),
        "gaze_stability":     float(gaze_stability),
    }

    # selected_features = subconjunto relevante para los modelos
    selected: Dict[str, float] = {k: features[k] for k in [
        "theta", "alpha", "low_beta", "high_beta",
        "theta_power", "alpha_power",
        "theta_alpha_ratio", "fatigue_index_eeg", "trust_beta_ratio",
        "blink_rate", "blink_rate_per_min",
        "blink_intensity", "blink_duration_mean", "perclos",
        "pupil_mean", "pupil_norm_mean",
        "gaze_stability",
    ]}

    return FeatureBundle(
        all_features=features,
        selected_features=selected,
    )
