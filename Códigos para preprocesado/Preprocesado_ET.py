from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

# PARAMETROS
PUPIL_MIN_MM = 2.0
PUPIL_MAX_MM = 5.5
PUPIL_MAX_GAP_S = 0.35
BLINK_MIN_SAMPLES = 2
BLINK_MAX_SAMPLES = 15
ARTIFACT_MIN_S = 1.20
GAZE_MAX_GAP_S = 0.03
GAZE_MEDIAN_WINDOW = 3
SACCADE_VELOCITY_THRESHOLD = 1.50
SACCADE_MIN_DURATION_S = 0.020
SACCADE_MAX_DURATION_S = 0.200
SACCADE_MIN_AMPLITUDE = 0.010
MARKERS = ["phase", "block", "trial", "condition", "current_marker"]


def choose_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title="Selecciona el Excel de eye tracking", filetypes=[("Excel", "*.xlsx")])
    root.destroy()
    return Path(path) if path else None


def key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def find_sheet(xls, accepted):
    accepted = {key(x) for x in accepted}
    for sheet in xls.sheet_names:
        if key(sheet) in accepted:
            return sheet
    raise ValueError(f"No se encontro {accepted}. Hojas: {xls.sheet_names}")


def sampling(timestamps):
    t = pd.to_numeric(timestamps, errors="coerce").to_numpy(float)
    dt = np.diff(t)
    good = dt[np.isfinite(dt) & (dt > 0)]
    if not len(good):
        raise ValueError("No se pudo calcular la frecuencia desde timestamp_lsl")
    median_dt = float(np.median(good))
    return t, median_dt


def runs(mask):
    z = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    d = np.diff(z)
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def gap_duration(t, start, end, dt):
    if end - start <= 1 or not (np.isfinite(t[start]) and np.isfinite(t[end - 1])):
        return (end - start) * dt
    return max((end - start) * dt, t[end - 1] - t[start] + dt)


def interpolate(series, t, dt, max_gap, groups=None):
    out = pd.to_numeric(series, errors="coerce").astype(float).copy()
    arr = out.to_numpy(copy=True)
    flag = np.zeros(len(arr), dtype=np.int8)
    for start, end in runs(~np.isfinite(arr)):
        if gap_duration(t, start, end, dt) > max_gap or start == 0 or end >= len(arr):
            continue
        if groups is not None and groups[start - 1] != groups[end]:
            continue
        if np.isfinite(arr[start - 1]) and np.isfinite(arr[end]):
            arr[start:end] = np.linspace(arr[start - 1], arr[end], end - start + 2)[1:-1]
            flag[start:end] = 1
    out.iloc[:] = arr
    return out, flag


def marker_columns(df):
    return [c for c in MARKERS if c in df.columns]


def clean_pupil(raw):
    df = raw.copy().reset_index(drop=True)
    required = ["timestamp_lsl", "pupil_left", "pupil_right"]
    if any(c not in df for c in required):
        raise ValueError(f"Faltan columnas de pupila. Disponibles: {df.columns.tolist()}")
    t, dt = sampling(df["timestamp_lsl"])

    result = pd.DataFrame({"timestamp_lsl": df["timestamp_lsl"]})
    clean = {}
    interp_flags = {}
    for eye in ["pupil_left", "pupil_right"]:
        original = pd.to_numeric(df[eye], errors="coerce")
        result[f"{eye}_original"] = original
        v = original.copy()
        bad = ~np.isfinite(v) | (v < PUPIL_MIN_MM) | (v > PUPIL_MAX_MM)
        v[bad] = np.nan
        clean[eye], interp_flags[eye] = interpolate(v, t, dt, PUPIL_MAX_GAP_S)
        result[f"{eye}_limpio"] = clean[eye]

    pre_left = pd.to_numeric(df["pupil_left"], errors="coerce")
    pre_right = pd.to_numeric(df["pupil_right"], errors="coerce")
    pre_left[(pre_left < PUPIL_MIN_MM) | (pre_left > PUPIL_MAX_MM)] = np.nan
    pre_right[(pre_right < PUPIL_MIN_MM) | (pre_right > PUPIL_MAX_MM)] = np.nan
    missing_both = pre_left.isna() & pre_right.isna()

    result["pupil_mean_limpio"] = pd.concat(clean, axis=1).mean(axis=1, skipna=True)
    result.loc[clean["pupil_left"].isna() & clean["pupil_right"].isna(), "pupil_mean_limpio"] = np.nan
    result["blink"] = 0
    result["blink_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    events = []
    blink_id = 0
    event_id = 0
    for start, end in runs(missing_both.to_numpy()):
        event_id += 1
        n = end - start
        duration = gap_duration(t, start, end, dt)
        if BLINK_MIN_SAMPLES <= n <= BLINK_MAX_SAMPLES:
            kind = "blink"; blink_id += 1; bid = blink_id
            result.loc[start:end-1, "blink"] = 1
            result.loc[start:end-1, "blink_id"] = bid
        elif duration >= ARTIFACT_MIN_S:
            kind = "artifact"; bid = pd.NA
        elif n > BLINK_MAX_SAMPLES:
            kind = "long_loss"; bid = pd.NA
        else:
            kind = "short_missing"; bid = pd.NA
        row = {"blink_id": bid, "tipo": kind,
               "inicio_timestamp": t[start], "fin_timestamp": t[end-1],
               "duracion_s": duration}
        for c in marker_columns(df): row[c] = df.loc[start, c]
        events.append(row)

    for c in marker_columns(df): result[c] = df[c]
    return result, pd.DataFrame(events)


def trial_groups(df):
    cols = [c for c in ["phase", "block", "trial", "condition"] if c in df]
    if not cols: return np.zeros(len(df), dtype=int)
    return pd.factorize(pd.MultiIndex.from_frame(df[cols].fillna("NA")).to_flat_index())[0]


def clean_gaze(raw):
    df = raw.copy().reset_index(drop=True)
    required = ["timestamp_lsl", "gaze_x", "gaze_y"]
    if any(c not in df for c in required):
        raise ValueError(f"Faltan columnas de gaze. Disponibles: {df.columns.tolist()}")
    t, dt = sampling(df["timestamp_lsl"])
    groups = trial_groups(df)
    out = pd.DataFrame({"timestamp_lsl": df["timestamp_lsl"]})

    smooth = {}
    for axis in ["gaze_x", "gaze_y"]:
        original = pd.to_numeric(df[axis], errors="coerce")
        out[f"{axis}_original"] = original
        v = original.copy()
        v[(v < 0) | (v > 1)] = np.nan
        cleaned, flag = interpolate(v, t, dt, GAZE_MAX_GAP_S, groups)
        smooth[axis] = cleaned.rolling(GAZE_MEDIAN_WINDOW, center=True, min_periods=1).median()
        out[f"{axis}_limpio"] = smooth[axis]

    x, y = smooth["gaze_x"].to_numpy(), smooth["gaze_y"].to_numpy()
    delta_t = np.r_[np.nan, np.diff(t)]
    distance = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    same = np.r_[False, groups[1:] == groups[:-1]]
    valid = same & np.isfinite(distance) & np.isfinite(delta_t) & (delta_t > 0)
    velocity = np.full(len(df), np.nan); velocity[valid] = distance[valid] / delta_t[valid]
    candidate = np.isfinite(velocity) & (velocity >= SACCADE_VELOCITY_THRESHOLD)

    saccades = []
    sid = 0
    for start, end in runs(candidate):
        a = max(0, start - 1); b = end - 1
        if groups[a] != groups[b]: continue
        duration = t[b] - t[a]
        amplitude = np.hypot(x[b] - x[a], y[b] - y[a])
        if not (SACCADE_MIN_DURATION_S <= duration <= SACCADE_MAX_DURATION_S): continue
        if not np.isfinite(amplitude) or amplitude < SACCADE_MIN_AMPLITUDE: continue
        sid += 1
        row = {"saccade_id": sid, "inicio_timestamp": t[a], "fin_timestamp": t[b],
               "duracion_ms": duration * 1000, "amplitud_normalizada": amplitude,
               "velocidad_media_normalizada_s": np.nanmean(velocity[start:end])}
        for c in marker_columns(df): row[c] = df.loc[a, c]
        saccades.append(row)

    for c in marker_columns(df): out[c] = df[c]
    return out, pd.DataFrame(saccades)


def process(source):
    xls = pd.ExcelFile(source, engine="openpyxl")
    pupil_sheet = find_sheet(xls, ["pupil"])
    gaze_sheet = find_sheet(xls, ["gaze2d", "gaze_2d", "gaze 2d"])
    pupil, blinks = clean_pupil(pd.read_excel(xls, sheet_name=pupil_sheet))
    gaze, saccades = clean_gaze(pd.read_excel(xls, sheet_name=gaze_sheet))
    output = source.with_name(source.stem + "_ET_LIMPIO_ESENCIAL.xlsx")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pupil.to_excel(writer, sheet_name="Pupil", index=False)
        blinks.to_excel(writer, sheet_name="Blinks", index=False)
        gaze.to_excel(writer, sheet_name="Gaze2D", index=False)
        saccades.to_excel(writer, sheet_name="Sacadas", index=False)
    return output


def main():
    source = choose_file()
    if source is None:
        print("No se selecciono archivo"); return
    output = process(source)
    print(f"Archivo creado: {output}")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root=tk.Tk(); root.withdraw(); messagebox.showinfo("Terminado", f"Archivo creado:\n{output}"); root.destroy()
    except Exception: pass

if __name__ == "__main__":
    main()
