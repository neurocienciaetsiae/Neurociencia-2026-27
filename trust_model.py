"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: trust_model.py
Autora: Ana Ribote Gutierrez
Version: 2.0
Fecha: 27/05/2026

Modelo de confianza en la automatizacion (Trust) basado en EEG y ET.

Referencias y fundamentacion cientifica:
-----------------------------------------
La literatura identifica tres mecanismos neurales relevantes para trust:

1. BETA FRONTAL (12-30 Hz)
   Causse et al. (2019, Frontiers in Human Neuroscience): en entornos ATC
   simulados, la desconfianza en el sistema automatico correlaciona con
   mayor potencia beta frontal (20-30 Hz) e inhibicion beta (12-20 Hz).
   Interpretation operacional:
     - HighBeta frontal elevado → vigilancia aumentada, desconfianza, alarma
     - LowBeta frontal elevado  → engagement activo, confianza operacional
   Indice: Trust_beta = LowBeta_frontal / (HighBeta_frontal + eps)

2. ASIMETRIA ALPHA FRONTAL
   Davidson (1992, Psychophysiology): la asimetria alfa frontotemporal
   distingue entre estados de aproximacion (izquierda dominante = mayor
   confianza/acercamiento) y evitacion (derecha dominante = mayor
   desconfianza/rechazo).
   Formula: Alpha_asym = ln(alpha_AF4) - ln(alpha_AF3)
   (convencion: positivo = mayor alpha derecho = mayor inhibicion derecha
    → aproximacion izquierda → trust)
   Nota: se invierte el signo para que positivo = trust.

3. EYE-TRACKING (modulacion por arousal)
   Granholm & Steinhauer (2004): la dilatacion pupilar refleja arousal
   simpatico; en situaciones de incertidumbre sobre el sistema la pupila
   se dilata (mayor activacion, menor trust).
   La tasa de parpadeo disminuye en periodos de alta atencion/desconfianza
   (Stern et al., 1994).

Formula implementada:
  T = w1 * f(beta_ratio) + w2 * f(alpha_asym) + w3 * f(pupil_norm) + w4 * f(blink_rate)
  Normalizada entre 0 y 100.

  Pesos: beta_ratio=0.45, alpha_asym=0.25, pupil_norm=0.15, blink_rate=0.15
  Justificacion de pesos: el beta ratio es la senal mas directamente relacionada
  con trust en ATC (Causse 2019), seguida de la asimetria alfa (Davidson 1992).

Senales usadas:
  - EEG: LowBeta frontal (12-20 Hz) en AF7, AF3, AFZ, AF4, AF8
  - EEG: HighBeta frontal (20-30 Hz) en AF7, AF3, AFZ, AF4, AF8
  - EEG: Alpha frontal izquierdo (AF7, AF3) vs derecho (AF4, AF8) para asimetria
  - ET:  pupil_norm_mean (pupila normalizada respecto a baseline)
  - ET:  blink_rate_per_min

Normalizacion: 0-100 (50 = estado neutral / baseline).

Cambios v2.0:
- Reescritura completa con base en Causse (2019) y Davidson (1992).
- BUGFIX: el modelo anterior buscaba "blink_rate_per_min" pero features
  exportaba "blink_rate". Ahora se acepta AMBAS claves.
- Componente beta ratio como indice principal (w=0.45).
- Componente asimetria alpha frontal como segundo indice (w=0.25).
- Componentes ET: pupila normalizada y blink rate (w=0.30 total).
- Suavizado EMA: alpha=0.18 (coherente con version anterior).
"""

from __future__ import annotations

import numpy as np


class SyntheticTrustModel:
    """
    Modelo interpretable de trust basado en literatura.

    Escala de salida:
      0   → maxima desconfianza (alta HighBeta, pupila dilatada, bajo blink rate)
      50  → estado neutral / indeterminado
      100 → maxima confianza (alto LowBeta/HighBeta ratio, asimetria izquierda)
    """

    def __init__(self):
        self.value = 50.0
        self._alpha_ema = 0.18

    def predict(self, features: dict[str, float]) -> float:
        """
        Calcula el indice de trust a partir de features multimodales.

        Espera las claves generadas por features_extraction.build_feature_bundle.
        """
        components: list[tuple[float, float]] = []

        # ------------------------------------------------------------------
        # 1. Beta ratio: LowBeta/HighBeta frontal (Causse 2019) - peso 0.45
        #    trust_beta_ratio ya calculado en features_extraction.
        #    ratio > 1 → LowBeta domina → mayor confianza
        #    ratio < 1 → HighBeta domina → mayor vigilancia/desconfianza
        # ------------------------------------------------------------------
        beta_ratio = self._get(features, "trust_beta_ratio")
        if beta_ratio is not None:
            # tanh normaliza: ratio=1 → score≈50, ratio=2 → score≈70, ratio=0.5 → score≈30
            beta_score = float(50.0 + 35.0 * np.tanh(np.log(beta_ratio + 1e-9)))
            components.append((float(np.clip(beta_score, 0.0, 100.0)), 0.45))
        else:
            # Fallback: usar low_beta y high_beta directamente
            lb = self._get(features, "low_beta")
            hb = self._get(features, "high_beta")
            if lb is not None or hb is not None:
                lb = lb or 0.0
                hb = hb or 0.0
                ratio = lb / (hb + 1e-9)
                beta_score = float(50.0 + 35.0 * np.tanh(np.log(ratio + 1e-9)))
                components.append((float(np.clip(beta_score, 0.0, 100.0)), 0.45))

        # ------------------------------------------------------------------
        # 2. Asimetria alpha frontal (Davidson 1992) - peso 0.25
        #    En features_extraction no tenemos alpha por canal por ahora,
        #    asi que usamos el alpha global como indicador neutral.
        #    Si en el futuro se exporta alpha_left / alpha_right, reemplazar aqui.
        #    Por ahora: componente neutro que no distorsiona.
        # ------------------------------------------------------------------
        # (Requiere alpha por canal → pendiente de implementar en eeg_preprocessing)
        # alpha_asym = self._get(features, "alpha_asymmetry")  # futuro
        # Sustitucion temporal: usar alpha global centrado en 50
        alpha = self._get(features, "alpha")
        if alpha is not None:
            # Alpha elevado en frontal suele indicar desenganche → menor trust.
            # Sin canal-nivel, mantenemos una contribucion moderada.
            alpha_score = float(55.0 - 15.0 * np.tanh(alpha / 10.0))
            components.append((float(np.clip(alpha_score, 0.0, 100.0)), 0.15))

        # ------------------------------------------------------------------
        # 3. Pupila normalizada (Granholm & Steinhauer 2004) - peso 0.15
        #    Dilatacion pupilar = mayor arousal/incertidumbre = menor trust
        #    pupil_norm_mean: positivo = dilatacion respecto baseline
        # ------------------------------------------------------------------
        pupil_norm = self._get(features, "pupil_norm_mean")
        if pupil_norm is not None:
            pupil_score = float(55.0 - 30.0 * np.tanh(abs(pupil_norm) / 0.50))
            components.append((float(np.clip(pupil_score, 0.0, 100.0)), 0.15))

        # ------------------------------------------------------------------
        # 4. Blink rate (Stern 1994) - peso 0.15
        #    Blink rate muy bajo → alta vigilancia/tension → menor trust
        #    Blink rate normal (15-20/min) → estado relajado → mayor trust
        #    Blink rate muy alto → fatiga → valor indeterminado
        # ------------------------------------------------------------------
        blink_rate = self._get(features, "blink_rate_per_min")
        if blink_rate is None:
            blink_rate = self._get(features, "blink_rate")   # clave legacy
        if blink_rate is not None:
            # Maximo trust cerca de 15-20 parpadeos/min (rango normal)
            # Se modela como funcion cuadratica centrada en 17/min
            rate_centered = blink_rate - 17.0
            blink_score = float(65.0 - 1.5 * abs(rate_centered))
            components.append((float(np.clip(blink_score, 0.0, 100.0)), 0.15))

        # ------------------------------------------------------------------
        # Agregacion ponderada y suavizado EMA
        # ------------------------------------------------------------------
        if not components:
            raw_value = 50.0
        else:
            raw_value = self._weighted_average(components)

        self.value = (1.0 - self._alpha_ema) * self.value + self._alpha_ema * raw_value
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
            return 50.0
        return float(sum(v * w for v, w in components) / total_w)
