"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: fatigue_model.py
Autora: Ana Ribote Gutierrez
Version: 2.0
Fecha: 27/05/2026

Modelo de fatiga mental basado en EEG y ET.

Referencias y fundamentacion cientifica:
-----------------------------------------
1. INDICE EEG DE FATIGA (θ+α)/β
   Lal & Craig (2001, Ergonomics): en conductores fatigados, la potencia
   combinada theta+alpha aumenta y la beta disminuye. El ratio (θ+α)/β es
   el indice EEG mas citado para fatiga en entornos de conduccion y vigilancia.
   Jap et al. (2009, Expert Systems with Applications): validacion del mismo
   indice en conductores con privacion de sueno.
   Formula: F_eeg = (theta + alpha) / (low_beta + high_beta + eps)

2. BLINK RATE Y DURACION
   Caffier et al. (2003, European Journal of Applied Physiology):
   la tasa de parpadeo aumenta con la fatiga en tareas de vigilancia.
   Stern et al. (1994): la duracion del parpadeo aumenta con la fatiga
   (los ojos permanecen mas tiempo cerrados).
   Combinar tasa y duracion mejora la discriminacion.

3. PERCLOS (Percentage of Eye Closure)
   Wierwille & Ellsworth (1994, SAE Technical Paper): PERCLOS es la medida
   de campo mas validada para fatiga al volante. Se define como el porcentaje
   de tiempo con >80% de cierre ocular. En nuestra implementacion simplificada
   se usa el % de muestras con senyal pupilar invalida como aproximacion.

4. PUPILA
   Lowenstein et al. (1963): la dilatacion pupilar espontanea decrece con la
   fatiga. Un pupil_norm_mean negativo (pupila mas pequenia que el baseline)
   puede indicar mayor fatiga.

Formula implementada:
  F = w1 * f(EEG_fatigue_index) + w2 * f(blink_rate) + w3 * f(blink_duration)
      + w4 * f(perclos) + w5 * f(pupil_norm)

  Pesos: EEG=0.35, blink_rate=0.20, blink_duration=0.15, perclos=0.20, pupila=0.10
  Total = 1.00

Senales usadas:
  - EEG: theta (4-8 Hz) + alpha (8-12 Hz) en todos los canales
  - EEG: low_beta (12-20 Hz) + high_beta (20-30 Hz) como denominador
  - ET:  blink_rate_per_min (tasa de parpadeo)
  - ET:  blink_duration_mean (duracion media del parpadeo, segundos)
  - ET:  perclos (% tiempo con senyal pupilar invalida)
  - ET:  pupil_norm_mean (pupila normalizada respecto a baseline)

Normalizacion: 0-100.

Cambios v2.0:
- Reescritura completa con base en Lal (2001) y Wierwille (1994).
- BUGFIX: el modelo anterior buscaba "theta_power" y "alpha_power",
  pero features_extraction exportaba "theta" y "alpha". Ahora se aceptan
  AMBAS claves ("theta_power"/"alpha_power" y "theta"/"alpha").
- Indice EEG (θ+α)/β como componente principal (w=0.35).
- PERCLOS como segundo componente (w=0.20).
- Duracion de parpadeo como tercer componente (w=0.15).
- Tasa de parpadeo como cuarto componente (w=0.20).
- Pupila normalizada como quinto componente (w=0.10).
"""

from __future__ import annotations

import numpy as np


class FatigueModel:
    """
    Modelo interpretable de fatiga mental.

    Escala de salida:
      0   → alerta maxima (bajo theta+alpha, parpadeos cortos, pupila dilatada)
      50  → estado moderado / indeterminado
      100 → fatiga severa (alto theta+alpha, PERCLOS elevado, parpadeos largos)
    """

    # Valores de referencia para normalizacion (ajustables por calibracion)
    # Obtenidos de literatura para operadores de vigilancia en reposo relajado.
    _REF_FATIGUE_INDEX_EEG = 1.5   # (theta+alpha)/beta en reposo: ~0.8-2.0
    _REF_BLINK_RATE_MIN   = 15.0   # parpadeos/min en reposo
    _REF_BLINK_DUR_S      = 0.15   # duracion media en reposo (s)
    _REF_PERCLOS_REST     = 0.05   # 5% de tiempo con cierre en reposo

    def __init__(self):
        self.value = 20.0
        self._alpha_ema = 0.12

    def predict(self, features: dict[str, float]) -> float:
        """
        Calcula el indice de fatiga a partir de features multimodales.
        """
        components: list[tuple[float, float]] = []

        # ------------------------------------------------------------------
        # 1. Indice EEG (θ+α)/β (Lal 2001, Jap 2009) - peso 0.35
        # ------------------------------------------------------------------
        fatigue_eeg = self._get(features, "fatigue_index_eeg")
        if fatigue_eeg is not None:
            # fatigue_index_eeg = (theta+alpha)/(low_beta+high_beta)
            # En reposo ~1.0-1.5; en fatiga sube a 3-5.
            eeg_score = float(100.0 * np.tanh(fatigue_eeg / self._REF_FATIGUE_INDEX_EEG / 2.0))
            components.append((float(np.clip(eeg_score, 0.0, 100.0)), 0.35))
        else:
            # Fallback: calcular con claves individuales
            theta = self._get(features, "theta_power") or self._get(features, "theta") or 0.0
            alpha = self._get(features, "alpha_power") or self._get(features, "alpha") or 0.0
            lb    = self._get(features, "low_beta") or 0.0
            hb    = self._get(features, "high_beta") or 0.0
            if theta > 0 or alpha > 0:
                idx = (theta + alpha) / (lb + hb + 1e-9)
                eeg_score = float(100.0 * np.tanh(idx / self._REF_FATIGUE_INDEX_EEG / 2.0))
                components.append((float(np.clip(eeg_score, 0.0, 100.0)), 0.35))

        # ------------------------------------------------------------------
        # 2. PERCLOS (Wierwille 1994) - peso 0.20
        #    % tiempo con cierre ocular (senyal pupilar invalida)
        # ------------------------------------------------------------------
        perclos = self._get(features, "perclos")
        if perclos is not None:
            # 5% → score~50; 20% → score~90; 0% → score~0
            perclos_score = float(100.0 * np.tanh(perclos / self._REF_PERCLOS_REST))
            components.append((float(np.clip(perclos_score, 0.0, 100.0)), 0.20))

        # ------------------------------------------------------------------
        # 3. Tasa de parpadeo (Caffier 2003) - peso 0.20
        #    Aumenta con fatiga en tareas de vigilancia.
        # ------------------------------------------------------------------
        blink_rate = self._get(features, "blink_rate_per_min")
        if blink_rate is None:
            blink_rate = self._get(features, "blink_rate")
        if blink_rate is not None:
            rate_score = float(100.0 * np.tanh(blink_rate / (self._REF_BLINK_RATE_MIN * 2.0)))
            components.append((float(np.clip(rate_score, 0.0, 100.0)), 0.20))

        # ------------------------------------------------------------------
        # 4. Duracion media del parpadeo (Stern 1994) - peso 0.15
        #    Parpadeos mas largos → mayor fatiga
        # ------------------------------------------------------------------
        blink_dur = self._get(features, "blink_duration_mean")
        if blink_dur is not None and blink_dur > 0:
            dur_score = float(100.0 * np.tanh(blink_dur / (self._REF_BLINK_DUR_S * 3.0)))
            components.append((float(np.clip(dur_score, 0.0, 100.0)), 0.15))

        # ------------------------------------------------------------------
        # 5. Pupila normalizada (Lowenstein 1963) - peso 0.10
        #    Pupila mas pequenia que baseline → mayor fatiga
        # ------------------------------------------------------------------
        pupil_norm = self._get(features, "pupil_norm_mean")
        if pupil_norm is not None:
            # pupil_norm < 0 → miosis (contraccion) → fatiga
            pupil_score = float(50.0 + 40.0 * np.tanh(-pupil_norm / 0.5))
            components.append((float(np.clip(pupil_score, 0.0, 100.0)), 0.10))

        # ------------------------------------------------------------------
        # Agregacion y suavizado EMA
        # ------------------------------------------------------------------
        if not components:
            raw = self.value
        else:
            raw = self._weighted_average(components)

        self.value = (1.0 - self._alpha_ema) * self.value + self._alpha_ema * raw
        self.value = float(np.clip(self.value, 0.0, 100.0))

        return self.value

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _get(self, features: dict[str, float], key: str) -> float | None:
        value = features.get(key)
        if value is None:
            return None
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    def _weighted_average(self, components: list[tuple[float, float]]) -> float:
        total_w = sum(w for _, w in components)
        if total_w <= 0:
            return self.value
        return float(sum(v * w for v, w in components) / total_w)
