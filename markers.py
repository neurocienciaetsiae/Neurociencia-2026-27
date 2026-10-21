"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: markers.py 
Autora: Ana Ribote Gutierrez
Version: 2.1 (FIXED - Calibration Display)
Fecha: 17/05/2026

Motor de interpretacion de marcadores experimentales.

Marcadores del experimento Trust_ATC_neutral_100_100_50_100:

  Calibracion:
    CALIBRATION_START_EYES_CLOSED / CALIBRATION_END_EYES_CLOSED
    CALIBRATION_START_EYES_OPEN  / CALIBRATION_END_EYES_OPEN

  Fixation (baseline de trial):
    FIXATION_CROSS_START | BLOCK_{N} | Trial={N}
    FIXATION_CROSS_END | BLOCK_{N} | Trial={N}

  Confianza HIGH:
    CONF_H | B{N} | T{N} | V={val}

  Confianza LOW:
    CONF_L | B{N} | T{N} | V{val}

  Bienvenida y final:
    WELCOME
    END
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


class TrialPhase(Enum):
    IDLE = auto()
    CALIBRATION = auto()
    BASELINE = auto()
    BLOCK = auto()
    TRIAL = auto()
    RESPONSE = auto()
    ENDED = auto()


@dataclass
class TrialInfo:
    block: int = 0
    trial: int = 0
    condition: str = ""
    t_baseline_start: float = 0.0
    t_baseline_end: float = 0.0
    response: str = ""
    confidence_high: float = -1.0
    confidence_low: float = -1.0


@dataclass
class TrialEngineState:
    current_trial: TrialInfo = field(default_factory=TrialInfo)
    current_block: int = 0
    last_marker: str = ""
    welcome_received: bool = False
    calibration_type: str = ""
    calibration_active: bool = False  # NUEVO: rastrear si estamos en calibración


class TrialEngine:
    def __init__(self):
        self.current_phase = TrialPhase.IDLE
        self.state = TrialEngineState()

        self.on_welcome: Callable[[float], None] | None = None
        self.on_calib_start: Callable[[str, float], None] | None = None
        self.on_calib_end: Callable[[str, float], None] | None = None
        self.on_baseline_start: Callable[[TrialInfo, float], None] | None = None
        self.on_baseline_end: Callable[[TrialInfo, float], None] | None = None
        self.on_confidence_response: Callable[[TrialInfo, str, float, float], None] | None = None
        self.on_experiment_end: Callable[[float], None] | None = None

    def process_marker(self, marker: str, timestamp: float):
        """Procesa un marcador entrante"""
        marker = str(marker or "").strip()
        timestamp = float(timestamp)

        if not marker:
            return

        self.state.last_marker = marker
        m = marker.upper()

        # WELCOME
        if self._is_welcome(m):
            self.current_phase = TrialPhase.IDLE
            self.state.welcome_received = True
            if self.on_welcome:
                self.on_welcome(timestamp)
            return

        # CALIBRATION START
        if self._is_calibration_start(m):
            self.current_phase = TrialPhase.CALIBRATION
            calib_type = self._parse_calibration_type(marker)
            self.state.calibration_type = calib_type
            self.state.calibration_active = True  # NUEVO: marcar calibración activa
            if self.on_calib_start:
                self.on_calib_start(calib_type, timestamp)
            return

        # CALIBRATION END
        if self._is_calibration_end(m):
            self.current_phase = TrialPhase.IDLE
            calib_type = self._parse_calibration_type(marker)
            self.state.calibration_active = False  # NUEVO: marcar calibración inactiva
            if self.on_calib_end:
                self.on_calib_end(calib_type, timestamp)
            return

        # FIXATION CROSS START
        if self._is_fixation_start(m):
            self.current_phase = TrialPhase.BASELINE
            block = self._parse_block(marker)
            trial = self._parse_trial(marker)

            if block > 0:
                self.state.current_block = block

            if trial > 0 or block > 0:
                self.state.current_trial = TrialInfo(
                    block=block if block > 0 else self.state.current_block,
                    trial=trial,
                    condition=self.state.current_trial.condition,
                    t_baseline_start=timestamp,
                )

            if self.on_baseline_start:
                self.on_baseline_start(self.state.current_trial, timestamp)
            return

        # FIXATION CROSS END
        if self._is_fixation_end(m):
            self.state.current_trial.t_baseline_end = timestamp
            if self.on_baseline_end:
                self.on_baseline_end(self.state.current_trial, timestamp)
            return

        # CONFIDENCE HIGH
        if self._is_confidence_high(m):
            self.current_phase = TrialPhase.RESPONSE
            block = self._parse_block(marker)
            trial = self._parse_trial(marker)
            confidence = self._parse_confidence(marker)

            if block > 0:
                self.state.current_trial.block = block
            if trial > 0:
                self.state.current_trial.trial = trial

            self.state.current_trial.confidence_high = confidence
            self.state.current_trial.condition = "HIGH"

            if self.on_confidence_response:
                self.on_confidence_response(self.state.current_trial, "HIGH", confidence, timestamp)
            return

        # CONFIDENCE LOW
        if self._is_confidence_low(m):
            self.current_phase = TrialPhase.RESPONSE
            block = self._parse_block(marker)
            trial = self._parse_trial(marker)
            confidence = self._parse_confidence(marker)

            if block > 0:
                self.state.current_trial.block = block
            if trial > 0:
                self.state.current_trial.trial = trial

            self.state.current_trial.confidence_low = confidence
            self.state.current_trial.condition = "LOW"

            if self.on_confidence_response:
                self.on_confidence_response(self.state.current_trial, "LOW", confidence, timestamp)
            return

        # END
        if self._is_experiment_end(m):
            self.current_phase = TrialPhase.ENDED
            if self.on_experiment_end:
                self.on_experiment_end(timestamp)
            return

    # DETECTORES - VERSIÓN MEJORADA CON REGEX
    def _is_welcome(self, m: str) -> bool:
        return m == "WELCOME"

    def _is_calibration_start(self, m: str) -> bool:
        """Detecta CALIBRATION_START o CALIBRATION START (con espacios o guiones)"""
        return bool(re.search(r"CALIBRATION[_\s]*START", m, flags=re.IGNORECASE))

    def _is_calibration_end(self, m: str) -> bool:
        """Detecta CALIBRATION_END o CALIBRATION END (con espacios o guiones)"""
        return bool(re.search(r"CALIBRATION[_\s]*END", m, flags=re.IGNORECASE))

    def _is_fixation_start(self, m: str) -> bool:
        return "FIXATION_CROSS_START" in m

    def _is_fixation_end(self, m: str) -> bool:
        return "FIXATION_CROSS_END" in m

    def _is_confidence_high(self, m: str) -> bool:
        return m.startswith("CONF_H")

    def _is_confidence_low(self, m: str) -> bool:
        return m.startswith("CONF_L")

    def _is_experiment_end(self, m: str) -> bool:
        return m == "END"

    # PARSERS - VERSIÓN MEJORADA
    def _parse_block(self, marker: str) -> int:
        """Parsea BLOCK_{N} o B{N}"""
        patterns = [r"BLOCK[_=\s]*(\d+)", r"\bB(\d+)\b"]
        return self._first_int_match(marker, patterns)

    def _parse_trial(self, marker: str) -> int:
        """Parsea Trial={N} o T{N}"""
        patterns = [r"Trial[_=\s]*(\d+)", r"\bT(\d+)\b"]
        return self._first_int_match(marker, patterns)

    def _parse_confidence(self, marker: str) -> float:
        """Parsea V={val} o Vval"""
        patterns = [r"V\s*=\s*([-+]?\d*\.?\d+)", r"V([-+]?\d*\.?\d+)"]
        for pattern in patterns:
            match = re.search(pattern, marker, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    pass
        return -1.0

    def _parse_calibration_type(self, marker: str) -> str:
        """Extrae EYES_CLOSED o EYES_OPEN - MEJORADO"""
        m = marker.upper()
        if "EYES_CLOSED" in m or "EYES CLOSED" in m:
            return "EYES_CLOSED"
        elif "EYES_OPEN" in m or "EYES OPEN" in m:
            return "EYES_OPEN"
        return "UNKNOWN"

    def _first_int_match(self, marker: str, patterns: list[str]) -> int:
        for pattern in patterns:
            match = re.search(pattern, marker, flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    pass
        return 0

    # CONSULTAS DE ESTADO - CON MEJORAS PARA CALIBRACIÓN
    def get_current_phase(self) -> TrialPhase:
        return self.current_phase

    def get_current_trial_info(self) -> TrialInfo:
        return self.state.current_trial

    def get_current_block(self) -> int:
        return self.state.current_block

    def get_last_marker(self) -> str:
        return self.state.last_marker

    def is_in_calibration(self) -> bool:
        return self.current_phase == TrialPhase.CALIBRATION and self.state.calibration_active

    def is_in_baseline(self) -> bool:
        return self.current_phase == TrialPhase.BASELINE

    def get_baseline_duration(self) -> float:
        if self.state.current_trial.t_baseline_start > 0 and self.state.current_trial.t_baseline_end > 0:
            return self.state.current_trial.t_baseline_end - self.state.current_trial.t_baseline_start
        return 0.0

    def get_calibration_type(self) -> str:
        """NUEVO: método para obtener el tipo de calibración actual"""
        return self.state.calibration_type

    def is_calibration_active(self) -> bool:
        """NUEVO: método para verificar si hay calibración activa"""
        return self.state.calibration_active

    def reset(self):
        self.current_phase = TrialPhase.IDLE
        self.state = TrialEngineState()
