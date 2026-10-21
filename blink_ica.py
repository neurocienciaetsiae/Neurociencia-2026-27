"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: blink_ica.py
Autora: Ana Ribote Gutierrez
Version: 1.1
Fecha: 06/06/2026

Deteccion y limpieza de parpadeos basada en ICA del EEG.

Funcionamiento general:
-----------------------
1. Durante la calibracion con ojos abiertos (>= 15 s) se acumula EEG limpio.
2. Al terminar la calibracion se ajusta FastICA sobre esos datos.
3. Se identifica automaticamente la componente del parpadeo mediante:
     - peso en canales frontales (AF7, AF3, AFZ, AF4, AF8): los parpadeos
       generan un dipolo electrico que se ve fuertemente en frontales.
     - coherencia de signo en frontales: un parpadeo real tiene el mismo
       signo en todos los canales frontales (eleva o baja toda la franja).
     - curtosis alta: los parpadeos son eventos cortos y aislados, su
       distribucion estadistica es muy puntiaguda.
4. Se calcula un umbral adaptativo a partir de la distribucion observada,
   usando mediana y MAD (robustos frente a la propia presencia de blinks).
5. En tiempo real se multiplica cada muestra nueva por la matriz de
   unmixing aprendida -> activacion de la componente del parpadeo ->
   umbral para detectar; o se reconstruye sin la componente para limpiar.

Ventajas frente al detector del eye-tracker:
- Detecta parpadeos parciales que el Tobii pierde.
- No depende de que la camara vea bien la pupila.
- Permite validacion cruzada: si ICA-EEG y Tobii coinciden, es un
  parpadeo seguro; si solo uno lo ve, puede ser artefacto.

Cambios v1.1:
- Identificacion de componente del parpadeo mejorada: ahora exige
  coherencia de signo en canales frontales (un parpadeo real es
  monopolar frontal, no dipolar).
- Umbral mas robusto: MAD en lugar de std en la calibracion.
- Anadidos diagnosticos: score de cada componente, ratio de senyal
  conservada tras la limpieza.
- fit() ahora devuelve un dict con metricas en lugar de un bool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis

try:
    from sklearn.decomposition import FastICA
    _SKLEARN_AVAILABLE = True
except ImportError:
    FastICA = None
    _SKLEARN_AVAILABLE = False


@dataclass
class BlinkEvent:
    timestamp_start: float
    timestamp_end: float
    duration_s: float
    peak_amplitude: float


class BlinkICADetector:
    """
    Detector de parpadeos en EEG mediante FastICA.

    Uso tipico:
      detector = BlinkICADetector(channel_names=['AF7','AF3','AFZ','AF4','AF8',
                                                  'P3','PZ','P4'], fs=125.0)
      detector.fit(eeg_calibration_data)           # datos de calibracion
      detector.is_fitted  # True si la calibracion ICA tuvo exito

      # En tiempo real:
      events = detector.process(chunk, timestamps)  # devuelve lista de BlinkEvent
    """

    # Canales que se consideran frontales para identificar la componente
    FRONTAL_CH_DEFAULT = ["AF7", "AF3", "AFZ", "AF4", "AF8"]

    def __init__(
        self,
        channel_names: list[str],
        fs: float,
        frontal_channels: list[str] | None = None,
        min_blink_duration_s: float = 0.05,
        max_blink_duration_s: float = 0.80,
        threshold_n_std: float = 3.0,
    ):
        self.channel_names = list(channel_names)
        self.fs = float(fs)
        self.frontal_channels = frontal_channels or self.FRONTAL_CH_DEFAULT
        self.min_blink_samples = int(min_blink_duration_s * fs)
        self.max_blink_samples = int(max_blink_duration_s * fs)
        self.threshold_n_std = float(threshold_n_std)

        # Indices de canales frontales en self.channel_names
        self._frontal_idx = [
            i for i, ch in enumerate(self.channel_names)
            if ch.upper() in [f.upper() for f in self.frontal_channels]
        ]

        self.ica: "FastICA | None" = None
        self.blink_component_idx: int | None = None
        self.threshold: float = 0.0

        # Estado para deteccion continua entre llamadas a process()
        self._in_blink: bool = False
        self._blink_start_ts: float = 0.0
        self._blink_start_idx_in_run: int = 0
        self._blink_peak: float = 0.0
        self._samples_in_blink: int = 0

    # ------------------------------------------------------------------
    # Calibracion (offline, una vez)
    # ------------------------------------------------------------------

    def fit(self, eeg_data: np.ndarray) -> dict:
        """
        Ajusta FastICA sobre el bloque de calibracion.

        Parametros
        ----------
        eeg_data : ndarray (n_samples, n_channels)
            EEG ya preprocesado (filtrado, sin canales malos) de la
            calibracion con ojos abiertos. Minimo ~10 s a 125 Hz.

        Devuelve
        --------
        dict con claves:
            - "success": bool
            - "reason":  str (motivo si success=False)
            - "blink_component_idx": int o None
            - "threshold": float
            - "frontal_ratio": float (peso frontal de la componente elegida)
            - "kurtosis":  float
            - "n_samples": int
            - "sign_coherence": float (0..1, coherencia de signo en frontales)
        """
        result = {
            "success": False,
            "reason": "",
            "blink_component_idx": None,
            "threshold": 0.0,
            "frontal_ratio": 0.0,
            "kurtosis": 0.0,
            "n_samples": 0,
            "sign_coherence": 0.0,
        }

        if not _SKLEARN_AVAILABLE:
            print("[BlinkICA] sklearn no disponible. Instala con: pip install scikit-learn")
            result["reason"] = "sklearn_not_available"
            return result

        eeg_data = np.asarray(eeg_data, dtype=float)

        if eeg_data.ndim != 2:
            print(f"[BlinkICA] Forma invalida: {eeg_data.shape}")
            result["reason"] = f"invalid_shape_{eeg_data.shape}"
            return result

        n_samples, n_ch = eeg_data.shape
        result["n_samples"] = n_samples
        min_required = int(8 * self.fs)  # 8 segundos minimo

        if n_samples < min_required:
            print(f"[BlinkICA] Insuficientes muestras: {n_samples} < {min_required}")
            result["reason"] = f"insufficient_samples_{n_samples}<{min_required}"
            return result

        if not self._frontal_idx:
            print(f"[BlinkICA] No se encontraron canales frontales en {self.channel_names}")
            result["reason"] = "no_frontal_channels"
            return result

        # Limpiar NaN/inf
        eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Ajustar FastICA con tantas componentes como canales
        try:
            self.ica = FastICA(
                n_components=n_ch,
                random_state=42,
                max_iter=500,
                tol=0.0001,
                whiten="unit-variance",
            )
            sources = self.ica.fit_transform(eeg_data)  # (n_samples, n_components)
        except Exception as exc:
            print(f"[BlinkICA] Error ajustando ICA: {exc}")
            result["reason"] = f"fastica_failed_{exc}"
            return result

        # Identificar componente del parpadeo (con diagnosticos)
        chosen, diag = self._identify_blink_component(sources)

        if chosen is None:
            print("[BlinkICA] No se pudo identificar componente del parpadeo.")
            result["reason"] = "no_blink_component_found"
            return result

        self.blink_component_idx = chosen

        # Umbral robusto sobre el modulo de la activacion
        blink_activation = np.abs(sources[:, chosen])
        median = float(np.median(blink_activation))
        mad = float(np.median(np.abs(blink_activation - median)))
        std_robust = 1.4826 * mad  # MAD -> std equivalente
        self.threshold = median + self.threshold_n_std * std_robust

        # Diagnostico
        mixing = self.ica.mixing_  # (n_channels, n_components)
        topo = mixing[:, chosen]
        print(f"[BlinkICA] Calibrado. Componente del parpadeo: #{chosen}")
        print(f"[BlinkICA] Topografia (peso por canal):")
        for ch, w in zip(self.channel_names, topo):
            marker = " <- frontal" if ch in self.frontal_channels else ""
            print(f"           {ch}: {w:+.3f}{marker}")
        print(f"[BlinkICA] Umbral de deteccion: {self.threshold:.3f}")
        print(f"[BlinkICA] Score: frontal_ratio={diag['frontal_ratio']:.3f}, "
              f"kurt={diag['kurtosis']:.2f}, sign_coh={diag['sign_coherence']:.2f}")

        result.update({
            "success": True,
            "blink_component_idx": int(chosen),
            "threshold": float(self.threshold),
            "frontal_ratio": float(diag["frontal_ratio"]),
            "kurtosis": float(diag["kurtosis"]),
            "sign_coherence": float(diag["sign_coherence"]),
        })
        return result

    def _identify_blink_component(self, sources: np.ndarray) -> tuple:
        """
        Identifica que componente de ICA corresponde al parpadeo.

        Criterios combinados:
          1. Peso frontal absoluto alto en la matriz de mezclado.
          2. Coherencia de signo en frontales: un parpadeo real es monopolar
             frontal (sube o baja toda la franja a la vez), no dipolar.
             Una componente con AF3 positivo y AF4 negativo no es parpadeo.
          3. Curtosis alta (Fisher > 1): distribucion puntiaguda, eventos raros.

        Devuelve (idx, diagnostico_dict) o (None, dict_vacio) si ninguna sirve.
        """
        if self.ica is None:
            return None, {}

        mixing = self.ica.mixing_  # (n_channels, n_components)
        n_components = sources.shape[1]

        scores = np.zeros(n_components)
        diagnostics = []

        for c in range(n_components):
            # 1. Ratio de peso frontal vs total
            total_weight = np.sum(np.abs(mixing[:, c])) + 1e-9
            frontal_weight = np.sum(np.abs(mixing[self._frontal_idx, c]))
            frontal_ratio = frontal_weight / total_weight

            # 2. Coherencia de signo en frontales: |suma(signos)| / n_frontales
            #    Si todos los frontales tienen el mismo signo -> coherencia = 1
            #    Si la mitad positiva y la mitad negativa (dipolar) -> 0
            frontal_weights = mixing[self._frontal_idx, c]
            frontal_signs = np.sign(frontal_weights)
            n_frontal = len(self._frontal_idx)
            sign_coherence = float(np.abs(np.sum(frontal_signs)) / max(1, n_frontal))

            # 3. Curtosis de la fuente (positiva = puntiaguda)
            k = float(kurtosis(sources[:, c], fisher=True))
            k_score = max(0.0, k)

            # Score combinado: requiere los tres componentes
            scores[c] = frontal_ratio * sign_coherence * k_score

            diagnostics.append({
                "component": c,
                "frontal_ratio": frontal_ratio,
                "sign_coherence": sign_coherence,
                "kurtosis": k,
                "score": scores[c],
            })

        if np.max(scores) <= 0:
            return None, {}

        best = int(np.argmax(scores))

        # Rechazar candidatos con curtosis demasiado baja: no son parpadeos
        if diagnostics[best]["kurtosis"] < 1.0:
            return None, {}

        # Rechazar candidatos con coherencia de signo baja: son dipolos
        if diagnostics[best]["sign_coherence"] < 0.4:
            return None, {}

        return best, diagnostics[best]

    # ------------------------------------------------------------------
    # Deteccion en tiempo real
    # ------------------------------------------------------------------

    def process(
        self,
        chunk: np.ndarray,
        timestamps: list[float] | np.ndarray,
    ) -> list[BlinkEvent]:
        """
        Procesa un chunk nuevo y devuelve los parpadeos completos
        detectados en el. El estado de "estoy en medio de un parpadeo"
        se mantiene entre llamadas para no perder eventos en los bordes.
        """
        if not self.is_fitted:
            return []

        chunk = np.asarray(chunk, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)

        if chunk.ndim != 2 or len(chunk) == 0:
            return []

        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

        # Aplicar matriz de unmixing aprendida → activaciones de todas las componentes
        try:
            sources = self.ica.transform(chunk)
        except Exception:
            return []

        # Solo nos interesa la componente del parpadeo
        activation = np.abs(sources[:, self.blink_component_idx])
        is_blink_sample = activation > self.threshold

        events: list[BlinkEvent] = []

        for i, (above, ts) in enumerate(zip(is_blink_sample, timestamps)):
            if above and not self._in_blink:
                # Inicio de parpadeo
                self._in_blink = True
                self._blink_start_ts = float(ts)
                self._blink_peak = float(activation[i])
                self._samples_in_blink = 1

            elif above and self._in_blink:
                # Continuacion
                self._samples_in_blink += 1
                if activation[i] > self._blink_peak:
                    self._blink_peak = float(activation[i])

            elif not above and self._in_blink:
                # Fin de parpadeo
                duration_samples = self._samples_in_blink
                end_ts = float(ts)
                duration_s = end_ts - self._blink_start_ts

                if (self.min_blink_samples <= duration_samples <= self.max_blink_samples):
                    events.append(BlinkEvent(
                        timestamp_start=self._blink_start_ts,
                        timestamp_end=end_ts,
                        duration_s=duration_s,
                        peak_amplitude=self._blink_peak,
                    ))

                self._in_blink = False
                self._samples_in_blink = 0
                self._blink_peak = 0.0

        return events

    # ------------------------------------------------------------------
    # Limpieza del EEG (elimina la componente del parpadeo)
    # ------------------------------------------------------------------

    def clean(self, chunk: np.ndarray) -> np.ndarray:
        """
        Elimina la componente del parpadeo del EEG y devuelve la senal
        reconstruida sin ese artefacto.

        Procedimiento:
          sources = chunk @ W           (descomposicion)
          sources[:, blink] = 0         (cero a la componente del parpadeo)
          chunk_limpio = sources @ A    (recomposicion sin parpadeos)

        Si el detector no esta entrenado, devuelve el chunk sin cambios.
        """
        if not self.is_fitted:
            return chunk

        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2 or len(chunk) == 0:
            return chunk

        chunk_safe = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            sources = self.ica.transform(chunk_safe)
            sources[:, self.blink_component_idx] = 0.0
            chunk_clean = self.ica.inverse_transform(sources)
            return chunk_clean
        except Exception:
            return chunk

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return (
            self.ica is not None
            and self.blink_component_idx is not None
        )

    def reset(self):
        """Reinicia el estado pero mantiene el modelo entrenado."""
        self._in_blink = False
        self._samples_in_blink = 0
        self._blink_peak = 0.0
