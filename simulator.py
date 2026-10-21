"""
Entidad: Universidad Politécnica de Madrid
Proyecto: ATC Trust Project / Arquitectura multimodal para la inferencia de la confianza en la automatización en tiempo real mediante la fusión de señales neurofisiológicas en entornos ATC
Código: simulator.py
Autora: Ana Ribote Gutiérrez
Estado: Versión preliminar
Versión: 1.0
Fecha: 08/04/2026

Simulador CSV -> LSL.

Lee los CSV del participante y publica por LSL:
- EEG
- Gaze2D
- Pupil
- Markers

La reproducción respeta los timestamps de los archivos y puede acelerarse
con PLAYBACK_SPEED.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, local_clock

from config import PLAYBACK_SPEED, STREAM_NAMES


class CSVLSLSimulator:
    def __init__(
        self,
        data_folder: str | Path,
        eeg_channel_map: dict[str, str] | None = None,
        file_names: dict[str, str] | None = None,
    ):
        """
        data_folder:
            Carpeta donde están los CSV del participante.

        eeg_channel_map:
            Mapeo entre nombre lógico del canal y columna real del CSV.
            Ejemplo: {"Fp1": "chann_1", "Fp2": "chann_2", ...}

        file_names:
            Nombres de los CSV que se van a leer.
        """
        self._stop = False
        self.data_folder = Path(data_folder)

        self.file_names = file_names or {
            "eeg": "eeg_data.csv",
            "gaze2d": "gaze2d_data.csv",
            "pupil": "pupil_data.csv",
            "markers": "psychopymarkers_data.csv",
        }

        self.eeg_channel_map = eeg_channel_map or {
            "Fp1": "chann_1",
            "Fp2": "chann_2",
            "F3": "chann_3",
            "F4": "chann_4",
            "C3": "chann_5",
            "C4": "chann_6",
            "T7": "chann_7",
            "T8": "chann_8",
            "Pz": "chann_9",
            "O1": "chann_17",
            "O2": "chann_18",
        }

        self.eeg_df = self._load_eeg(self.data_folder / self.file_names["eeg"])
        self.gaze_df = self._load_numeric(self.data_folder / self.file_names["gaze2d"])
        self.pupil_df = self._load_numeric(self.data_folder / self.file_names["pupil"])
        self.markers_df = self._load_markers(self.data_folder / self.file_names["markers"])

        self.eeg_idx = 0
        self.gaze_idx = 0
        self.pupil_idx = 0
        self.marker_idx = 0

        starts = [
            float(self.eeg_df["timestamp"].iloc[0]),
            float(self.gaze_df["timestamp"].iloc[0]),
            float(self.pupil_df["timestamp"].iloc[0]),
        ]
        if len(self.markers_df) > 0:
            starts.append(float(self.markers_df["timestamp"].iloc[0]))

        self.t0 = min(starts)

        self.eeg_outlet = self._build_eeg_outlet()
        self.gaze_outlet = self._build_gaze_outlet()
        self.pupil_outlet = self._build_pupil_outlet()
        self.marker_outlet = self._build_marker_outlet()

    def stop(self):
        self._stop = True

    def _load_numeric(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")
        df = pd.read_csv(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_markers(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["timestamp", "marker", "notes"])
        df = pd.read_csv(path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_eeg(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        df = pd.read_csv(path)
        df = df.sort_values("timestamp").reset_index(drop=True)

        keep = ["timestamp"] + [col for col in self.eeg_channel_map.values() if col in df.columns]
        df = df[keep].copy()

        for csv_col in self.eeg_channel_map.values():
            if csv_col not in df.columns:
                df[csv_col] = np.nan

        return df

    def _build_eeg_outlet(self) -> StreamOutlet:
        channel_labels = list(self.eeg_channel_map.keys())

        info = StreamInfo(
            name=STREAM_NAMES["eeg"],
            type="EEG",
            channel_count=len(channel_labels),
            nominal_srate=0.0,
            channel_format="float32",
            source_id="csv_sim_eeg",
        )

        chs = info.desc().append_child("channels")
        for label in channel_labels:
            ch = chs.append_child("channel")
            ch.append_child_value("label", label)

        return StreamOutlet(info)

    def _build_gaze_outlet(self) -> StreamOutlet:
        info = StreamInfo(
            name=STREAM_NAMES["gaze2d"],
            type="Gaze",
            channel_count=2,
            nominal_srate=0.0,
            channel_format="float32",
            source_id="csv_sim_gaze2d",
        )
        return StreamOutlet(info)

    def _build_pupil_outlet(self) -> StreamOutlet:
        info = StreamInfo(
            name=STREAM_NAMES["pupil"],
            type="Pupil",
            channel_count=2,
            nominal_srate=0.0,
            channel_format="float32",
            source_id="csv_sim_pupil",
        )
        return StreamOutlet(info)

    def _build_marker_outlet(self) -> StreamOutlet:
        info = StreamInfo(
            name=STREAM_NAMES["markers"],
            type="Markers",
            channel_count=2,
            nominal_srate=0.0,
            channel_format="string",
            source_id="csv_sim_markers",
        )
        return StreamOutlet(info)

    def _elapsed_playback(self, wall_start: float) -> float:
        return (time.perf_counter() - wall_start) * PLAYBACK_SPEED

    def _relative_time(self, ts: float) -> float:
        return float(ts - self.t0)

    def run(self):
        wall_start = time.perf_counter()

        while not self._stop:
            elapsed = self._elapsed_playback(wall_start)
            progressed = False

            while self.eeg_idx < len(self.eeg_df):
                row = self.eeg_df.iloc[self.eeg_idx]
                rel_t = self._relative_time(float(row["timestamp"]))
                if rel_t > elapsed:
                    break

                sample = []
                for csv_col in self.eeg_channel_map.values():
                    val = row[csv_col]
                    sample.append(float(val) if pd.notna(val) else np.nan)

                self.eeg_outlet.push_sample(sample, local_clock())
                self.eeg_idx += 1
                progressed = True

            while self.gaze_idx < len(self.gaze_df):
                row = self.gaze_df.iloc[self.gaze_idx]
                rel_t = self._relative_time(float(row["timestamp"]))
                if rel_t > elapsed:
                    break

                gx = float(row["gaze2d_x"]) if pd.notna(row["gaze2d_x"]) else np.nan
                gy = float(row["gaze2d_y"]) if pd.notna(row["gaze2d_y"]) else np.nan
                self.gaze_outlet.push_sample([gx, gy], local_clock())
                self.gaze_idx += 1
                progressed = True

            while self.pupil_idx < len(self.pupil_df):
                row = self.pupil_df.iloc[self.pupil_idx]
                rel_t = self._relative_time(float(row["timestamp"]))
                if rel_t > elapsed:
                    break

                pl = float(row["pupil_left"]) if pd.notna(row["pupil_left"]) else np.nan
                pr = float(row["pupil_right"]) if pd.notna(row["pupil_right"]) else np.nan
                self.pupil_outlet.push_sample([pl, pr], local_clock())
                self.pupil_idx += 1
                progressed = True

            while self.marker_idx < len(self.markers_df):
                row = self.markers_df.iloc[self.marker_idx]
                rel_t = self._relative_time(float(row["timestamp"]))
                if rel_t > elapsed:
                    break

                marker = str(row["marker"]) if pd.notna(row["marker"]) else ""
                notes = str(row["notes"]) if "notes" in row.index and pd.notna(row["notes"]) else ""
                self.marker_outlet.push_sample([marker, notes], local_clock())
                self.marker_idx += 1
                progressed = True

            if (
                self.eeg_idx >= len(self.eeg_df)
                and self.gaze_idx >= len(self.gaze_df)
                and self.pupil_idx >= len(self.pupil_df)
                and self.marker_idx >= len(self.markers_df)
            ):
                time.sleep(0.01)
                continue

            if not progressed:
                time.sleep(0.001)


def start_simulator_in_thread(
    data_folder: str | Path,
    eeg_channel_map: dict[str, str] | None = None,
    file_names: dict[str, str] | None = None,
) -> tuple[CSVLSLSimulator, threading.Thread]:
    sim = CSVLSLSimulator(
        data_folder=data_folder,
        eeg_channel_map=eeg_channel_map,
        file_names=file_names,
    )
    thread = threading.Thread(target=sim.run, daemon=True)
    thread.start()
    return sim, thread