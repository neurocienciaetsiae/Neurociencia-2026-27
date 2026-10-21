"""Replay EEG data from an Excel workbook or CSV as LSL streams.

For an Excel workbook, the script reads the EEG worksheet and replays the
Markers worksheet when present. CSV input supports Timestamp/timestamp_lsl and
reconstructs marker transitions from a current_marker column when available.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, local_clock

from config import EEG_CHANNEL_NAMES, STREAM_NAMES


def _find_column(df: pd.DataFrame, *names: str) -> str | None:
    by_lower = {str(col).strip().lower(): str(col) for col in df.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def _normalise_eeg(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = _find_column(df, "Timestamp", "timestamp", "timestamp_lsl")
    if ts_col is None:
        raise ValueError("No encuentro una columna Timestamp, timestamp o timestamp_lsl.")
    channel_columns = {str(col).strip().upper(): str(col) for col in df.columns}
    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df[ts_col], errors="coerce")
    missing: list[str] = []
    for ch in EEG_CHANNEL_NAMES:
        source_col = channel_columns.get(ch.upper())
        if source_col is None:
            missing.append(ch)
            out[ch] = np.nan
        else:
            out[ch] = pd.to_numeric(df[source_col], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise ValueError("No hay filas EEG con timestamp válido.")
    if missing:
        raise ValueError(f"Faltan canales EEG requeridos: {missing}")
    return out


def _marker_events(df: pd.DataFrame) -> list[tuple[float, str]]:
    ts_col = _find_column(df, "timestamp_lsl", "Timestamp", "timestamp")
    marker_col = _find_column(df, "marker")
    if ts_col is None or marker_col is None:
        return []
    events = []
    for ts, marker in zip(df[ts_col], df[marker_col]):
        try:
            timestamp = float(ts)
        except (TypeError, ValueError):
            continue
        text = str(marker).strip() if pd.notna(marker) else ""
        if text and text.lower() != "nan":
            events.append((timestamp, text))
    return sorted(events)


def _read_input(path: Path) -> tuple[pd.DataFrame, list[tuple[float, str]]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        eeg_df = None
        eeg_sheet_name = None
        with pd.ExcelFile(path) as workbook:
            for name in workbook.sheet_names:
                header = pd.read_excel(workbook, sheet_name=name, nrows=0)
                cols = {str(c).strip().lower() for c in header.columns}
                if "timestamp_lsl" in cols and all(ch.lower() in cols for ch in EEG_CHANNEL_NAMES):
                    eeg_df = pd.read_excel(workbook, sheet_name=name)
                    eeg_sheet_name = name
                    break
            if eeg_df is None:
                raise ValueError("No encuentro una hoja con timestamp_lsl y los 8 canales EEG configurados.")
            out = _normalise_eeg(eeg_df)
            marker_name = next((n for n in workbook.sheet_names if n.strip().lower() == "markers"), None)
            marker_df = pd.read_excel(workbook, sheet_name=marker_name) if marker_name else None
        events = _marker_events(marker_df) if marker_df is not None else []
        if not events:
            marker_col = _find_column(eeg_df, "current_marker")
            ts_col = _find_column(eeg_df, "timestamp_lsl")
            if marker_col and ts_col:
                events = _transition_events(eeg_df[ts_col], eeg_df[marker_col])
        print(f"[replay] Excel: {path.name} | hoja EEG: {eeg_sheet_name} | marcadores: {len(events)}")
        return out, events

    if suffix == ".csv":
        # sep=None accepts both the project's comma CSVs and legacy semicolon CSVs.
        df = pd.read_csv(path, sep=None, engine="python")
        out = _normalise_eeg(df)
        ts_col = _find_column(df, "Timestamp", "timestamp", "timestamp_lsl")
        marker_col = _find_column(df, "current_marker")
        events = _transition_events(df[ts_col], df[marker_col]) if ts_col and marker_col else []
        print(f"[replay] CSV: {path.name} | marcadores reconstruidos: {len(events)}")
        return out, events
    raise ValueError("Formato no admitido. Usa un .xlsx, .xlsm o .csv.")


def _transition_events(timestamps, markers) -> list[tuple[float, str]]:
    events = []
    previous = ""
    for timestamp, marker in zip(timestamps, markers):
        text = str(marker).strip() if pd.notna(marker) else ""
        if not text or text.lower() == "nan":
            previous = ""
            continue
        if text != previous:
            try:
                events.append((float(timestamp), text))
            except (TypeError, ValueError):
                pass
        previous = text
    return events


def _build_eeg_outlet() -> StreamOutlet:
    info = StreamInfo(
        name=STREAM_NAMES["eeg"], type="EEG", channel_count=len(EEG_CHANNEL_NAMES),
        nominal_srate=125.0, channel_format="float32", source_id="file_replay_eeg",
    )
    channels = info.desc().append_child("channels")
    for label in EEG_CHANNEL_NAMES:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
    return StreamOutlet(info)


def _build_marker_outlet() -> StreamOutlet:
    info = StreamInfo(
        name=STREAM_NAMES["markers"], type="Markers", channel_count=2,
        nominal_srate=0.0, channel_format="string", source_id="file_replay_markers",
    )
    return StreamOutlet(info)


def replay(path: Path, speed: float, wait_before_start: float) -> None:
    df, events = _read_input(path)
    eeg_outlet = _build_eeg_outlet()
    marker_outlet = _build_marker_outlet()
    timestamps = df["timestamp"].to_numpy(dtype=float)
    samples = df[EEG_CHANNEL_NAMES].to_numpy(dtype=float)

    # If the source has no WELCOME, synthesize one at the first EEG sample.
    if not any(marker.strip().upper() == "WELCOME" for _, marker in events):
        events.append((float(timestamps[0]), "WELCOME"))
    events = sorted((t, m) for t, m in events if m.strip().upper() != "END")

    print("[replay] Outlets LSL preparados:")
    print(f"[replay] EEG: {STREAM_NAMES['eeg']} | Markers: {STREAM_NAMES['markers']}")
    print("[replay] Inicia main.py ahora; WELCOME llegará después de la espera configurada.")
    time.sleep(max(0.0, wait_before_start))

    source_t0 = float(timestamps[0])
    wall_t0 = time.perf_counter()
    event_idx = 0
    sent = 0
    for ts, sample in zip(timestamps, samples):
        target_elapsed = (float(ts) - source_t0) / speed
        while (time.perf_counter() - wall_t0) < target_elapsed:
            time.sleep(0.001)
        while event_idx < len(events) and events[event_idx][0] <= float(ts):
            marker, _notes = events[event_idx][1], ""
            marker_outlet.push_sample([marker, _notes], local_clock())
            print(f"[replay] Marcador: {marker}")
            event_idx += 1
        eeg_outlet.push_sample(np.nan_to_num(sample, nan=0.0).astype(float).tolist(), local_clock())
        sent += 1
        if sent % 5000 == 0:
            print(f"[replay] Enviadas {sent}/{len(samples)} muestras EEG")

    marker_outlet.push_sample(["END", ""], local_clock())
    print(f"[replay] Fin. Enviadas {sent} muestras EEG; marcador END enviado.")
    time.sleep(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce un Excel o CSV EEG como streams LSL.")
    parser.add_argument("file", type=Path, help="Ruta al Excel (.xlsx) o CSV de EEG.")
    parser.add_argument("--speed", type=float, default=4.0, help="Velocidad de reproducción (4 = cuatro veces más rápido).")
    parser.add_argument("--wait-before-start", type=float, default=8.0, help="Espera para que main.py conecte los streams.")
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("--speed debe ser mayor que cero.")
    path = args.file.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    replay(path, args.speed, args.wait_before_start)


if __name__ == "__main__":
    main()
