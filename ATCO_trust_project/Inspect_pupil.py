"""
Script de diagnostico: imprime los valores raw del stream Pupil durante 30 segundos.
Ejecutar con el eye-tracker conectado y parpadear varias veces a proposito.
Muestra: timestamp, pupil_left, pupil_right, y si el valor es cero/NaN/fuera de rango.
"""
import sys
try:
    from pylsl import StreamInlet, resolve_byprop
except ImportError:
    print("ERROR: pylsl no instalado")
    sys.exit(1)

import numpy as np
import time

STREAM_NAME = "Pupil"
DURATION_S = 30
PUPIL_MIN = 2.0
PUPIL_MAX = 5.5

print(f"Buscando stream '{STREAM_NAME}'...")
results = resolve_byprop("name", STREAM_NAME, timeout=10.0)
if not results:
    print(f"ERROR: stream '{STREAM_NAME}' no encontrado")
    sys.exit(1)

inlet = StreamInlet(results[0], max_buflen=60)
print(f"Conectado. Grabando {DURATION_S}s — parpadea varias veces a proposito\n")
print(f"{'t_rel':>8}  {'left':>8}  {'right':>8}  {'estado'}")
print("-" * 50)

t0 = time.time()
blink_candidates = []
prev_state = "ok"

while time.time() - t0 < DURATION_S:
    sample, ts = inlet.pull_sample(timeout=0.1)
    if sample is None:
        continue

    pl = float(sample[0]) if len(sample) > 0 else float('nan')
    pr = float(sample[1]) if len(sample) > 1 else float('nan')
    t_rel = time.time() - t0

    def classify(v):
        if not np.isfinite(v):
            return "NaN/inf"
        if v == 0.0:
            return "CERO"
        if v < PUPIL_MIN:
            return f"bajo({v:.2f})"
        if v > PUPIL_MAX:
            return f"alto({v:.2f})"
        return "ok"

    cl = classify(pl)
    cr = classify(pr)

    state = "ok" if cl == "ok" and cr == "ok" else f"L:{cl} R:{cr}"

    # Solo imprime cuando cambia el estado o es anomalo
    if state != "ok" or prev_state != "ok":
        print(f"{t_rel:8.3f}  {pl:8.4f}  {pr:8.4f}  {state}")
        if state != "ok":
            blink_candidates.append((t_rel, pl, pr, state))

    prev_state = state

print("\n--- Resumen ---")
print(f"Eventos detectados con valores anomalos: {len(blink_candidates)}")
if blink_candidates:
    print("Primeros 10:")
    for row in blink_candidates[:10]:
        print(f"  t={row[0]:.3f}s  L={row[1]:.4f}  R={row[2]:.4f}  -> {row[3]}")
else:
    print("NINGUN valor anomalo detectado — el eye-tracker no envia ceros ni NaN durante el blink.")
    print("Puede que el tracker interpole internamente los parpadeos.")
