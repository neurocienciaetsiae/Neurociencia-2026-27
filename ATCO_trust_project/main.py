"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: main.py
Autora: Ana Ribote Gutierrez
Version: 2.4
Fecha: 08/04/2026
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque

from PySide6 import QtWidgets

from config import EEG_CHANNEL_NAMES, EEG_FS
from lsl_bus import SharedState
from processor import OnlineProcessor
from ui import MainWindow, SetupDialog

# Para reproducción desde replay_eeg_lsl.py, main debe conectarse a streams
# externos. El simulador interno usa una ruta local específica del equipo autor.
USE_SIMULATOR = False


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    dialog = SetupDialog()
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        sys.exit(0)

    config = dialog.get_result()
    participant = config["participant"]
    session = config["session"]
    active_models = config["models"]
    active_modalities = config["modalities"]

    print(f"[main] Participante: {participant} | Sesión: {session}")
    print(f"[main] Modelos activos: {active_models}")
    print(f"[main] Modalidades activas: {active_modalities}")

    state = SharedState()

    for ch in EEG_CHANNEL_NAMES:
        state.eeg_data[ch] = deque(maxlen=5000)

    if USE_SIMULATOR:
        from simulator import start_simulator_in_thread

        simulator, _ = start_simulator_in_thread(
            data_folder=r"C:\Users\anagu\Desktop\TFM\Nautral_Condition\participant_22",
            eeg_channel_map={
                "AF7": "chann_1",
                "AF3": "chann_2",
                "AFZ": "chann_3",
                "AF4": "chann_4",
                "AF8": "chann_5",
                "P3": "chann_6",
                "PZ": "chann_7",
                "P4": "chann_8",
            },
        )
        time.sleep(2.0)
    else:
        simulator = None
        print("[main] Esperando streams LSL en red...")

    processor = OnlineProcessor(
        state=state,
        eeg_channel_names=EEG_CHANNEL_NAMES,
        fs=EEG_FS,
        active_models=active_models,
        active_modalities=active_modalities,
        participant=participant,
        session=session,
        recording_base_folder="recordings",
    )

    proc_thread = threading.Thread(target=processor.run_from_lsl, daemon=True)
    proc_thread.start()

    window = MainWindow(
        state=state,
        eeg_channel_names=EEG_CHANNEL_NAMES,
        participant=participant,
        session=session,
        active_models=active_models,
        active_modalities=active_modalities,
    )

    def handle_stop_requested():
        processor.request_stop()
        proc_thread.join(timeout=3.0)

        report_path = processor.stop(generate_report=True)

        if report_path is not None:
            window.set_recording_stopped(str(report_path))
        else:
            window.set_recording_stopped(
                "No report generated because no metrics were recorded yet."
            )

    window.stop_requested.connect(handle_stop_requested)
    window.show()

    exit_code = app.exec()

    processor.request_stop()
    proc_thread.join(timeout=3.0)
    processor.stop(generate_report=False)

    if simulator:
        simulator.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
