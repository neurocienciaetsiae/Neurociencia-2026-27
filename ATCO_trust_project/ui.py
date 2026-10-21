"""
Entidad: Universidad Politecnica de Madrid
Proyecto: ATC Trust Project
Codigo: ui.py
Autora: Ana Ribote Gutierrez
Version: 1.7
Fecha: 06/06/2026

Cambios v1.7:
- Anadida segunda curva en plot_blink para parpadeos detectados por ICA
  del EEG. Color cyan (#22D3EE) para distinguirla de la curva del ET
  (rojo #F43F5E). Aparece solo cuando state.ica_active = True (es decir,
  cuando ICA ya se ha entrenado al final de la calibracion con ojos
  abiertos).
- Anadidas dos tarjetas de metrica adicionales: "ICA blinks" y "ICA rate".
  Muestran "n/a" mientras ICA no esta activo.
- BUGFIX: la curva de parpadeos del ET ahora se dibuja correctamente.
  Antes, ningun componente rellenaba state.blink_line, por lo que la
  curva siempre estaba vacia.

Cambios v1.6:
- Optimizacion de latencia de UI.
- El timer principal sigue usando UPDATE_INTERVAL_MS, pero cada bloque se
  refresca a una frecuencia distinta:
  blink:  ms; EEG: 150 ms; pupila: 500 ms; bandas: 1000 ms;
  gauges: 200 ms; markers: 500 ms; Fs: 1000 ms.
- Se anade grafica rapida de blinks en el panel de Eye Tracking.
- Las barras de bandas ya no se eliminan y recrean en cada refresco:
  se actualiza el BarGraphItem con setOpts().
- EEG se recorta a DISPLAY_SECONDS y se submuestrea para visualizacion.
- Pupila se recorta a DISPLAY_SECONDS y labels de pupila usan solo la
  ventana visible, no todo el buffer.
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from config import (
    DEFAULT_EEG_PLOT_COLOURS,
    DISPLAY_SECONDS,
    UPDATE_INTERVAL_MS,
)
from lsl_bus import SharedState


# ---------------------------------------------------------------------------
# Setup dialog
# ---------------------------------------------------------------------------

class SetupDialog(QtWidgets.QDialog):
    """
    Popup que aparece al arrancar. Recoge:
      - ID participante
      - Sesion
      - Modelos a activar: Trust / Workload / Fatigue
      - Modalidades a usar: EEG / Eye-tracking
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session setup")
        self.setMinimumWidth(440)
        self.setModal(True)

        self.setStyleSheet("""
            QDialog, QWidget {
                background-color: #050816;
                color: #E6F1FF;
                font-family: 'Segoe UI';
            }
            QLabel { font-size: 13px; color: #94A3B8; }
            QLabel#title { font-size: 20px; font-weight: 700; color: #F8FAFC; }
            QLabel#section { font-size: 14px; font-weight: 600; color: #CBD5E1; margin-top: 8px; }
            QLineEdit {
                background-color: #0B1020;
                border: 1px solid #1E293B;
                border-radius: 8px;
                color: #E6F1FF;
                font-size: 14px;
                padding: 6px 10px;
            }
            QCheckBox {
                font-size: 14px;
                color: #E6F1FF;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px; height: 18px;
                border: 1px solid #334155;
                border-radius: 4px;
                background: #0B1020;
            }
            QCheckBox::indicator:checked {
                background: #8B5CF6;
                border: 1px solid #8B5CF6;
            }
            QPushButton {
                background-color: #8B5CF6;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                padding: 10px 24px;
            }
            QPushButton:hover { background-color: #7C3AED; }
            QPushButton:pressed { background-color: #6D28D9; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("ATCo HST - Session setup")
        title.setObjectName("title")
        layout.addWidget(title)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #1E293B;")
        layout.addWidget(line)

        sec1 = QtWidgets.QLabel("Participant info")
        sec1.setObjectName("section")
        layout.addWidget(sec1)

        grid = QtWidgets.QFormLayout()
        grid.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        grid.setSpacing(10)

        self.edit_participant = QtWidgets.QLineEdit()
        self.edit_participant.setPlaceholderText("e.g. P01")
        grid.addRow("Participant ID:", self.edit_participant)

        self.edit_session = QtWidgets.QLineEdit()
        self.edit_session.setPlaceholderText("e.g. S1")
        grid.addRow("Session:", self.edit_session)

        layout.addLayout(grid)

        sec2 = QtWidgets.QLabel("Models to compute and display")
        sec2.setObjectName("section")
        layout.addWidget(sec2)

        self.chk_trust = QtWidgets.QCheckBox("Trust")
        self.chk_workload = QtWidgets.QCheckBox("Workload")
        self.chk_fatigue = QtWidgets.QCheckBox("Fatigue")
        self.chk_trust.setChecked(True)

        for chk in (self.chk_trust, self.chk_workload, self.chk_fatigue):
            layout.addWidget(chk)

        sec3 = QtWidgets.QLabel("Signals to use")
        sec3.setObjectName("section")
        layout.addWidget(sec3)

        self.chk_eeg = QtWidgets.QCheckBox("EEG")
        self.chk_et = QtWidgets.QCheckBox("Eye-tracking")
        self.chk_eeg.setChecked(True)
        self.chk_et.setChecked(True)

        layout.addWidget(self.chk_eeg)
        layout.addWidget(self.chk_et)

        hint = QtWidgets.QLabel(
            "Select at least one model and one signal. Workload requires EEG."
        )
        hint.setStyleSheet("font-size: 11px; color: #475569;")
        layout.addWidget(hint)

        layout.addSpacing(8)

        btn = QtWidgets.QPushButton("Start session")
        btn.clicked.connect(self._on_accept)
        layout.addWidget(btn)

        self._result = None

    def _on_accept(self):
        models = []

        if self.chk_trust.isChecked():
            models.append("Trust")
        if self.chk_workload.isChecked():
            models.append("Workload")
        if self.chk_fatigue.isChecked():
            models.append("Fatigue")

        eeg_active = self.chk_eeg.isChecked()
        et_active = self.chk_et.isChecked()

        if not models:
            QtWidgets.QMessageBox.warning(
                self, "No model selected", "Please select at least one model.",
            )
            return

        if not eeg_active and not et_active:
            QtWidgets.QMessageBox.warning(
                self, "No signal selected", "Please select EEG, Eye-tracking or both.",
            )
            return

        if "Workload" in models and not eeg_active:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid configuration",
                "Workload requires EEG. Please enable EEG or disable Workload.",
            )
            return

        self._result = {
            "participant": self.edit_participant.text().strip() or "-",
            "session": self.edit_session.text().strip() or "-",
            "models": models,
            "modalities": {
                "eeg": eeg_active,
                "eye_tracking": et_active,
            },
        }

        self.accept()

    def get_result(self):
        return self._result


# ---------------------------------------------------------------------------
# Gauge circular
# ---------------------------------------------------------------------------

GAUGE_COLOURS = {
    "Trust": ("#22D3EE", "#8B5CF6"),
    "Workload": ("#F97316", "#EF4444"),
    "Fatigue": ("#4ADE80", "#06B6D4"),
}


class CircularGauge(QtWidgets.QWidget):
    def __init__(self, label: str = "Trust", parent=None):
        super().__init__(parent)
        self.label = label
        self.value = 50.0
        self.setMinimumSize(200, 200)

    def set_value(self, value: float):
        if not np.isfinite(value):
            value = 50.0
        new_value = float(np.clip(value, 0, 100))
        if abs(new_value - self.value) < 0.2:
            return
        self.value = new_value
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

            side = min(self.width(), self.height()) - 20
            x = (self.width() - side) / 2
            y = (self.height() - side) / 2
            rect = QtCore.QRectF(x, y, side, side)

            base_pen = QtGui.QPen(QtGui.QColor("#20263A"), 18)
            base_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(base_pen)
            painter.drawArc(rect, 225 * 16, -270 * 16)

            c1, c2 = GAUGE_COLOURS.get(self.label, ("#22D3EE", "#8B5CF6"))

            gradient = QtGui.QConicalGradient(rect.center(), 90)
            gradient.setColorAt(0.0, QtGui.QColor(c1))
            gradient.setColorAt(0.5, QtGui.QColor(c2))
            gradient.setColorAt(1.0, QtGui.QColor(c1))

            value_pen = QtGui.QPen(QtGui.QBrush(gradient), 18)
            value_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(value_pen)

            span = -270 * (self.value / 100.0)
            painter.drawArc(rect, 225 * 16, int(span * 16))

            inner_rect = rect.adjusted(30, 30, -30, -30)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#0B1020"))
            painter.drawEllipse(inner_rect)

            painter.setPen(QtGui.QColor("#E6F1FF"))

            font_size_label = max(8, min(12, side // 16))
            painter.setFont(QtGui.QFont("Segoe UI", font_size_label))
            painter.drawText(
                inner_rect.adjusted(0, -28, 0, 0),
                QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter,
                self.label,
            )

            font_big = QtGui.QFont("Segoe UI", max(14, min(24, side // 7)))
            font_big.setBold(True)
            painter.setFont(font_big)
            painter.drawText(
                inner_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{self.value:.0f}%",
            )

        finally:
            painter.end()


# ---------------------------------------------------------------------------
# Popup de calidad de señal EEG
# ---------------------------------------------------------------------------

class SignalQualityDialog(QtWidgets.QDialog):
    _STD_FLAT   = 0.5
    _STD_LOW    = 2.0
    _STD_HIGH   = 80.0
    _STD_NOISE  = 150.0

    _COLOR_GOOD   = "#22C55E"
    _COLOR_WARN   = "#F59E0B"
    _COLOR_BAD    = "#EF4444"
    _COLOR_NODATA = "#475569"

    def __init__(self, state: SharedState, channel_names: list[str], parent=None):
        super().__init__(parent)
        self.state = state
        self.channel_names = channel_names

        self.setWindowTitle("EEG Signal Quality")
        self.setMinimumWidth(360)
        self.setModal(False)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )

        self.setStyleSheet("""
            QDialog, QWidget {
                background-color: #050816;
                color: #E6F1FF;
                font-family: 'Segoe UI';
            }
            QLabel#title {
                font-size: 16px;
                font-weight: 700;
                color: #F8FAFC;
            }
            QLabel#subtitle {
                font-size: 10px;
                color: #475569;
            }
            QLabel#chName {
                font-size: 13px;
                font-weight: 600;
                color: #CBD5E1;
                min-width: 50px;
            }
            QLabel#stdVal {
                font-size: 11px;
                color: #94A3B8;
                min-width: 80px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("EEG Signal Quality")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Bad: confirmed by pipeline after calibration, or std out of range\\n"
            "Moderate: std borderline (possible noise or loose contact)\\n"
            "Good: clean signal\\n"
            "Before calibration: std proxy only. After calibration: pipeline result."
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #1E293B;")
        layout.addWidget(line)

        self._indicators: dict[str, tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = {}

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)

        for i, ch in enumerate(channel_names):
            dot = QtWidgets.QLabel("●")
            dot.setObjectName("chName")
            dot.setStyleSheet(f"color: {self._COLOR_NODATA}; font-size: 20px;")

            name_lbl = QtWidgets.QLabel(ch)
            name_lbl.setObjectName("chName")

            std_lbl = QtWidgets.QLabel("-- µV")
            std_lbl.setObjectName("stdVal")

            row = i // 2
            col_base = (i % 2) * 3

            grid.addWidget(dot,     row, col_base)
            grid.addWidget(name_lbl, row, col_base + 1)
            grid.addWidget(std_lbl,  row, col_base + 2)

            self._indicators[ch] = (dot, std_lbl)

        layout.addLayout(grid)

        self._lbl_global = QtWidgets.QLabel("")
        self._lbl_global.setStyleSheet("font-size: 12px; color: #94A3B8; padding-top: 4px;")
        self._lbl_global.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_global)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(500)

    def _refresh(self):
        n_good = 0
        n_warn = 0
        n_bad  = 0

        pipeline_bad = set(getattr(self.state, "bad_channels", []))
        calib_done   = getattr(self.state, "calibration_done", False)

        for ch, (dot, std_lbl) in self._indicators.items():
            arr = np.asarray(self.state.eeg_data.get(ch, []), dtype=float)

            if len(arr) < 10:
                dot.setStyleSheet(f"color: {self._COLOR_NODATA}; font-size: 20px;")
                std_lbl.setText("no data")
                n_bad += 1
                continue

            arr = arr[-250:]
            arr = arr[np.isfinite(arr)]

            if len(arr) < 5:
                dot.setStyleSheet(f"color: {self._COLOR_NODATA}; font-size: 20px;")
                std_lbl.setText("no data")
                n_bad += 1
                continue

            std = float(np.std(arr))
            std_lbl.setText(f"{std:.1f} µV")

            if calib_done and ch in pipeline_bad:
                color = self._COLOR_BAD
                n_bad += 1
            elif std < self._STD_FLAT or std > self._STD_NOISE:
                color = self._COLOR_BAD
                n_bad += 1
            elif std < self._STD_LOW or std > self._STD_HIGH:
                color = self._COLOR_WARN
                n_warn += 1
            else:
                color = self._COLOR_GOOD
                n_good += 1

            dot.setStyleSheet(f"color: {color}; font-size: 20px;")

        total = n_good + n_warn + n_bad
        if total > 0:
            status = ""
            if not calib_done:
                status = " (calibration pending, std proxy only)"
            self._lbl_global.setText(
                f"Channels, good: {n_good}  moderate: {n_warn}  bad: {n_bad}{status}"
            )

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Panel frame generico
# ---------------------------------------------------------------------------

class PanelFrame(QtWidgets.QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")

        self.layout_main = QtWidgets.QVBoxLayout(self)
        self.layout_main.setContentsMargins(16, 16, 16, 16)
        self.layout_main.setSpacing(10)

        if title:
            title_label = QtWidgets.QLabel(title)
            title_label.setObjectName("panelTitle")
            self.layout_main.addWidget(title_label)

    def add_widget(self, widget, stretch=0):
        self.layout_main.addWidget(widget, stretch)

    def add_layout(self, layout, stretch=0):
        self.layout_main.addLayout(layout, stretch)


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    stop_requested = QtCore.Signal()

    def __init__(
        self,
        state: SharedState,
        eeg_channel_names: list[str],
        participant: str = "-",
        session: str = "-",
        active_models: list[str] | None = None,
        active_modalities: dict[str, bool] | None = None,
    ):
        super().__init__()

        self.state = state
        self.eeg_channel_names = eeg_channel_names
        self.participant = participant
        self.session = session
        self.active_models = active_models or ["Trust"]
        self.active_modalities = active_modalities or {
            "eeg": True,
            "eye_tracking": True,
        }

        self.use_eeg = bool(self.active_modalities.get("eeg", True))
        self.use_et = bool(self.active_modalities.get("eye_tracking", True))

        self._quality_dialog: SignalQualityDialog | None = None

        self._last_blink_refresh = 0.0
        self._last_eye_refresh = 0.0
        self._last_eeg_refresh = 0.0
        self._last_band_refresh = 0.0
        self._last_model_refresh = 0.0
        self._last_marker_refresh = 0.0
        self._last_fs_refresh = 0.0

        self._blink_refresh_s = 0.08
        self._eye_refresh_s = 0.50
        self._eeg_refresh_s = 0.15
        self._band_refresh_s = 1.00
        self._model_refresh_s = 0.20
        self._marker_refresh_s = 0.50
        self._fs_refresh_s = 1.00

        self._blink_count_accum = 0
        self._blink_count_prev_window = 0

        self.setWindowTitle("ATCo-Trust Dashboard")
        self.resize(1750, 980)

        self._apply_styles()
        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_view)
        self.timer.start(UPDATE_INTERVAL_MS)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #050816;
                color: #E6F1FF;
                font-family: 'Segoe UI';
            }
            QFrame#panel {
                background-color: #0B1020;
                border: 1px solid #1E293B;
                border-radius: 20px;
            }
            QLabel#panelTitle {
                font-size: 24px;
                font-weight: 700;
                color: #F8FAFC;
                padding-bottom: 4px;
            }
            QLabel#metricValue {
                font-size: 18px;
                font-weight: 700;
                color: #FFFFFF;
            }
            QLabel#metricLabel {
                font-size: 11px;
                color: #94A3B8;
            }
            QLabel#sideText {
                font-size: 13px;
                color: #CBD5E1;
            }
            QLabel#infoLabel {
                font-size: 11px;
                color: #475569;
            }
            QLabel#infoValue {
                font-size: 11px;
                color: #94A3B8;
                font-weight: 600;
            }
            QLabel#fsLabel {
                font-size: 12px;
                color: #E6F1FF;
                background-color: #11182B;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 4px 8px;
            }
            QLabel#statusLabel {
                font-size: 11px;
                color: #94A3B8;
                padding-top: 4px;
            }
            QPushButton#stopButton {
                background-color: #DC2626;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 700;
                padding: 9px 16px;
            }
            QPushButton#stopButton:hover { background-color: #B91C1C; }
            QPushButton#stopButton:disabled {
                background-color: #334155;
                color: #94A3B8;
            }
            QPushButton#qualityButton {
                background-color: #1E3A5F;
                color: #7DD3FC;
                border: 1px solid #2563EB;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                padding: 6px 14px;
            }
            QPushButton#qualityButton:hover { background-color: #1D4ED8; color: white; }
            QGroupBox {
                border: 1px solid #1E293B;
                border-radius: 14px;
                margin-top: 6px;
                padding-top: 12px;
                font-size: 13px;
                font-weight: 600;
            }
            QTextEdit {
                background-color: #11182B;
                border: 1px solid #1E293B;
                border-radius: 10px;
                color: #E2E8F0;
                font-size: 11px;
            }
        """)

    def _metric_card(self, label_text: str, value_text: str):
        card = QtWidgets.QFrame()
        card.setStyleSheet(
            "background-color: #11182B; border: 1px solid #1E293B; border-radius: 14px;"
        )

        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)

        lbl = QtWidgets.QLabel(label_text)
        lbl.setObjectName("metricLabel")

        val = QtWidgets.QLabel(value_text)
        val.setObjectName("metricValue")

        layout.addWidget(lbl)
        layout.addWidget(val)

        return card, val

    def _style_plot(self, plot_widget, bottom_label, left_label):
        plot_widget.setMenuEnabled(False)
        plot_widget.showGrid(x=True, y=True, alpha=0.12)
        plot_widget.setLabel("bottom", bottom_label)
        plot_widget.setLabel("left", left_label)
        plot_widget.getAxis("left").setPen(pg.mkPen("#475569"))
        plot_widget.getAxis("bottom").setPen(pg.mkPen("#475569"))
        plot_widget.getAxis("left").setTextPen(pg.mkPen("#94A3B8"))
        plot_widget.getAxis("bottom").setTextPen(pg.mkPen("#94A3B8"))

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)

        root = QtWidgets.QHBoxLayout(cw)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        pg.setConfigOptions(
            antialias=False,
            background="#0B1020",
            foreground="#CBD5E1",
        )

        left_panel = self._build_left_panel()
        root.addWidget(left_panel, 3)

        self.plot_eeg = None
        self.eeg_curves = []
        self.eeg_labels = []
        self.plot_bands = None
        self.band_bar = None
        self.band_value_labels = []
        self.lbl_eeg_fs_corner = None

        if self.use_eeg:
            centre_panel = self._build_eeg_panel()
            root.addWidget(centre_panel, 5)

        self.plot_eye = None
        self.plot_blink = None
        self.curve_pupil_left = None
        self.curve_pupil_right = None
        self.curve_blink = None
        self.lbl_eye_fs_corner = None

        if self.use_et:
            right_panel = self._build_eye_panel()
            root.addWidget(right_panel, 3)

    def _build_left_panel(self):
        left_panel = PanelFrame("")
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(420)

        info_frame = QtWidgets.QFrame()
        info_frame.setStyleSheet(
            "background-color: #11182B; border: 1px solid #1E293B; border-radius: 10px;"
        )

        info_layout = QtWidgets.QGridLayout(info_frame)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(4)

        info_layout.addWidget(self._info_label("Participant"), 0, 0)
        self.lbl_participant = self._info_value(self.participant)
        info_layout.addWidget(self.lbl_participant, 0, 1)

        info_layout.addWidget(self._info_label("Session"), 1, 0)
        self.lbl_session = self._info_value(self.session)
        info_layout.addWidget(self.lbl_session, 1, 1)

        info_layout.addWidget(self._info_label("Models"), 2, 0)
        self.lbl_models = self._info_value(", ".join(self.active_models))
        info_layout.addWidget(self.lbl_models, 2, 1)

        modalities = []
        if self.use_eeg:
            modalities.append("EEG")
        if self.use_et:
            modalities.append("ET")

        info_layout.addWidget(self._info_label("Signals"), 3, 0)
        self.lbl_modalities = self._info_value(", ".join(modalities))
        info_layout.addWidget(self.lbl_modalities, 3, 1)

        left_panel.add_widget(info_frame)

        self.gauges: dict[str, CircularGauge] = {}
        n_models = len(self.active_models)

        if n_models == 1:
            g = CircularGauge(self.active_models[0])
            g.setMinimumHeight(200)
            g.setMaximumHeight(260)
            self.gauges[self.active_models[0]] = g
            left_panel.add_widget(g, 2)

        elif n_models == 2:
            gauge_row = QtWidgets.QHBoxLayout()
            gauge_row.setSpacing(4)

            for m in self.active_models:
                g = CircularGauge(m)
                g.setMinimumHeight(180)
                g.setMaximumHeight(220)
                self.gauges[m] = g
                gauge_row.addWidget(g)

            left_panel.add_layout(gauge_row, 2)

        else:
            gauge_row1 = QtWidgets.QHBoxLayout()
            gauge_row1.setSpacing(4)

            gauge_row2 = QtWidgets.QHBoxLayout()
            gauge_row2.setSpacing(4)

            for i, m in enumerate(self.active_models):
                g = CircularGauge(m)
                g.setMinimumHeight(160)
                g.setMaximumHeight(190)
                self.gauges[m] = g

                if i < 2:
                    gauge_row1.addWidget(g)
                else:
                    gauge_row2.addStretch()
                    gauge_row2.addWidget(g)
                    gauge_row2.addStretch()

            left_panel.add_layout(gauge_row1, 1)
            left_panel.add_layout(gauge_row2, 1)

        if self.use_eeg:
            self.btn_quality = QtWidgets.QPushButton("Signal quality")
            self.btn_quality.setObjectName("qualityButton")
            self.btn_quality.setToolTip(
                "Shows real-time EEG signal quality per channel\\n"
                "based on signal std (proxy for electrode contact)."
            )
            self.btn_quality.clicked.connect(self._on_quality_clicked)
            left_panel.add_widget(self.btn_quality)

        marker_title = QtWidgets.QLabel("LSL Markers")
        marker_title.setObjectName("sideText")
        left_panel.add_widget(marker_title)

        self.marker_box = QtWidgets.QTextEdit()
        self.marker_box.setReadOnly(True)
        self.marker_box.setMinimumHeight(160)
        left_panel.add_widget(self.marker_box, 1)

        left_panel.layout_main.addSpacing(14)

        self.btn_stop = QtWidgets.QPushButton("Stop recording and generate summary")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        left_panel.add_widget(self.btn_stop)

        self.lbl_stop_status = QtWidgets.QLabel("")
        self.lbl_stop_status.setObjectName("statusLabel")
        self.lbl_stop_status.setWordWrap(True)
        left_panel.add_widget(self.lbl_stop_status)

        return left_panel

    def _build_eeg_panel(self):
        centre_panel = PanelFrame("EEG signal")
        centre_panel.setMinimumWidth(680)

        header = QtWidgets.QHBoxLayout()
        header.addStretch()

        self.lbl_eeg_fs_corner = QtWidgets.QLabel("Fs real: -- Hz")
        self.lbl_eeg_fs_corner.setObjectName("fsLabel")
        header.addWidget(self.lbl_eeg_fs_corner)

        centre_panel.add_layout(header)

        self.plot_eeg = pg.PlotWidget()
        self._style_plot(self.plot_eeg, "Time (s)", "")
        self.plot_eeg.getAxis("left").setTicks([])
        self.plot_eeg.setYRange(-360, 360)
        self.plot_eeg.setXRange(-DISPLAY_SECONDS, 0)

        for idx, name in enumerate(self.eeg_channel_names):
            colour = DEFAULT_EEG_PLOT_COLOURS[idx % len(DEFAULT_EEG_PLOT_COLOURS)]

            curve = self.plot_eeg.plot(
                [], [],
                pen=pg.mkPen(colour, width=1.4),
            )

            self.eeg_curves.append(curve)

            label = pg.TextItem(text=name, color=colour, anchor=(0, 0.5))
            self.plot_eeg.addItem(label)
            self.eeg_labels.append(label)

        centre_panel.add_widget(self.plot_eeg, 3)

        band_label = QtWidgets.QLabel("Wave band frequency")
        band_label.setObjectName("sideText")
        centre_panel.add_widget(band_label)

        self.plot_bands = pg.PlotWidget()
        self._style_plot(self.plot_bands, "", "Power")
        self.plot_bands.setYRange(0, 50, padding=0.02)

        self.band_names = ["Theta", "Alpha", "LowBeta", "HighBeta", "Gamma"]
        self.band_colours = ["#7C9CFF", "#5EE6A8", "#FFC857", "#FF9B3D", "#FF7AA2"]

        x = np.arange(len(self.band_names))

        self.band_bar = pg.BarGraphItem(
            x=x,
            height=[0.0] * len(self.band_names),
            width=0.62,
            brushes=self.band_colours,
        )

        self.plot_bands.addItem(self.band_bar)
        self.plot_bands.getAxis("bottom").setTicks([list(zip(x, self.band_names))])

        self.band_value_labels = []

        for xi in x:
            lbl = pg.TextItem(text="", color="#E6F1FF", anchor=(0.5, 1.0))
            self.plot_bands.addItem(lbl)
            self.band_value_labels.append(lbl)

        centre_panel.add_widget(self.plot_bands, 1)

        return centre_panel

    def _build_eye_panel(self):
        right_panel = PanelFrame("Eye Tracking")
        right_panel.setMinimumWidth(400)

        header = QtWidgets.QHBoxLayout()
        header.addStretch()

        self.lbl_eye_fs_corner = QtWidgets.QLabel("Fs real: -- Hz")
        self.lbl_eye_fs_corner.setObjectName("fsLabel")
        header.addWidget(self.lbl_eye_fs_corner)

        right_panel.add_layout(header)

        self.plot_eye = pg.PlotWidget()
        self._style_plot(self.plot_eye, "Time (s)", "Pupil diameter (mm)")
        self.eye_legend = self.plot_eye.addLegend(offset=(10, 10))

        self.curve_pupil_left = self.plot_eye.plot(
            [], [],
            pen=pg.mkPen("#7DD3FC", width=2.0),
            name="Left pupil",
        )

        self.curve_pupil_right = self.plot_eye.plot(
            [], [],
            pen=pg.mkPen("#F6C85F", width=2.0),
            name="Right pupil",
        )

        right_panel.add_widget(self.plot_eye, 2)

        blink_title = QtWidgets.QLabel("Blink line")
        blink_title.setObjectName("sideText")
        right_panel.add_widget(blink_title)

        self.plot_blink = pg.PlotWidget()
        self._style_plot(self.plot_blink, "Time (s)", "Blink")
        self.plot_blink.setXRange(-DISPLAY_SECONDS, 0)
        self.plot_blink.setYRange(-0.1, 1.2)
        self.plot_blink.getAxis("left").setTicks([[(0, "0"), (1, "1")]])

        # Curva del ET (rojo, ya existente). Se dibuja a y=1 durante blink.
        self.curve_blink = self.plot_blink.plot(
            [], [],
            pen=pg.mkPen("#F43F5E", width=2.0),
            stepMode=False,
            name="ET blinks",
        )

        # NUEVO v1.7: curva del ICA (cyan). Se dibuja a y=1.08 (un poco
        # mas arriba) para que no se solape con la curva ET y se vean
        # ambas claramente cuando coinciden.
        self.curve_blink_ica = self.plot_blink.plot(
            [], [],
            pen=pg.mkPen("#22D3EE", width=2.0),
            stepMode=False,
            name="ICA blinks",
        )

        # Leyenda discreta sobre la grafica
        legend = pg.LegendItem(offset=(70, 8))
        legend.setParentItem(self.plot_blink.getPlotItem())
        legend.addItem(self.curve_blink, "ET")
        legend.addItem(self.curve_blink_ica, "ICA (EEG)")

        right_panel.add_widget(self.plot_blink, 1)

        eye_info_box = QtWidgets.QGroupBox("Eye-tracking information")
        eye_info_layout = QtWidgets.QGridLayout(eye_info_box)
        eye_info_layout.setHorizontalSpacing(10)
        eye_info_layout.setVerticalSpacing(10)

        pupil_left_card, self.lbl_pupil_left = self._metric_card(
            "Left pupil diameter", "0.00",
        )
        pupil_right_card, self.lbl_pupil_right = self._metric_card(
            "Right pupil diameter", "0.00",
        )
        blink_card, self.lbl_blinks = self._metric_card("ET blinks", "0")
        rate_card, self.lbl_blink_rate = self._metric_card("ET blink rate", "0.0 / min")
        # NUEVO v1.7: tarjetas para blinks detectados por ICA
        ica_card, self.lbl_ica_blinks = self._metric_card("ICA blinks", "n/a")
        ica_rate_card, self.lbl_ica_rate = self._metric_card("ICA rate", "n/a")

        eye_info_layout.addWidget(pupil_left_card,  0, 0)
        eye_info_layout.addWidget(pupil_right_card, 0, 1)
        eye_info_layout.addWidget(blink_card,       1, 0)
        eye_info_layout.addWidget(rate_card,        1, 1)
        eye_info_layout.addWidget(ica_card,         2, 0)
        eye_info_layout.addWidget(ica_rate_card,    2, 1)

        right_panel.add_widget(eye_info_box)

        return right_panel

    def _info_label(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("infoLabel")
        return lbl

    def _info_value(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("infoValue")
        return lbl

    def _on_stop_clicked(self):
        self.btn_stop.setEnabled(False)
        self.lbl_stop_status.setText("Stopping recording and generating summary...")
        self.stop_requested.emit()

    def set_recording_stopped(self, report_path: str):
        self.lbl_stop_status.setText(f"Recording stopped. Summary: {report_path}")

    def _on_quality_clicked(self):
        if self._quality_dialog is None or not self._quality_dialog.isVisible():
            self._quality_dialog = SignalQualityDialog(
                state=self.state,
                channel_names=self.eeg_channel_names,
                parent=self,
            )
            self._quality_dialog.show()
        else:
            self._quality_dialog.raise_()
            self._quality_dialog.activateWindow()

    def refresh_view(self):
        now = time.perf_counter()

        if self.use_eeg and self.plot_eeg is not None:
            if now - self._last_eeg_refresh >= self._eeg_refresh_s:
                self._refresh_eeg_signal()
                self._last_eeg_refresh = now

            if now - self._last_band_refresh >= self._band_refresh_s:
                self._refresh_bands()
                self._last_band_refresh = now

        if self.use_et and self.plot_eye is not None:
            if now - self._last_blink_refresh >= self._blink_refresh_s:
                self._refresh_blink_fast()
                self._last_blink_refresh = now

            if now - self._last_eye_refresh >= self._eye_refresh_s:
                self._refresh_eye_slow()
                self._last_eye_refresh = now

        if now - self._last_model_refresh >= self._model_refresh_s:
            self._refresh_models()
            self._last_model_refresh = now

        if now - self._last_marker_refresh >= self._marker_refresh_s:
            self.marker_box.setPlainText(
                "\\n".join(list(self.state.recent_markers))
            )
            self._last_marker_refresh = now

        if now - self._last_fs_refresh >= self._fs_refresh_s:
            self._refresh_fs_labels()
            self._last_fs_refresh = now

    def _refresh_models(self):
        if "Trust" in self.gauges:
            self.gauges["Trust"].set_value(self.state.trust_value)

        if "Workload" in self.gauges:
            val = getattr(self.state, "workload_value", 50.0)
            self.gauges["Workload"].set_value(val)

        if "Fatigue" in self.gauges:
            val = getattr(self.state, "fatigue_value", 50.0)
            self.gauges["Fatigue"].set_value(val)

    def _refresh_fs_labels(self):
        if self.lbl_eeg_fs_corner is not None:
            self.lbl_eeg_fs_corner.setText(
                f"Fs real: {self.state.eeg_fs_real:.1f} Hz"
            )

        if self.lbl_eye_fs_corner is not None:
            self.lbl_eye_fs_corner.setText(
                f"Fs real: {self.state.eye_fs_real:.1f} Hz"
            )

    def _refresh_eeg_signal(self):
        if len(self.state.eeg_time) <= 1:
            return

        time_arr = np.asarray(self.state.eeg_time, dtype=float)
        if len(time_arr) < 2:
            return

        t_rel = time_arr - time_arr[-1]
        mask = t_rel >= -DISPLAY_SECONDS

        if not np.any(mask):
            return

        t_plot = t_rel[mask]

        max_points = 500
        if len(t_plot) > max_points:
            step = int(np.ceil(len(t_plot) / max_points))
            idx = np.where(mask)[0][::step]
            t_plot = t_rel[idx]
        else:
            idx = np.where(mask)[0]

        offsets = np.linspace(
            260, -260,
            num=max(1, len(self.eeg_channel_names)),
        )

        for i, ch in enumerate(self.eeg_channel_names):
            arr = np.asarray(self.state.eeg_data.get(ch, []), dtype=float)

            if len(arr) == 0:
                continue

            n_common = min(len(time_arr), len(arr))
            if n_common < 2:
                continue

            arr = arr[-n_common:]
            local_idx = idx - (len(time_arr) - n_common)
            valid_idx = local_idx[(local_idx >= 0) & (local_idx < len(arr))]

            if len(valid_idx) < 2:
                continue

            arr_plot = arr[valid_idx]
            t_ch = t_rel[-n_common:][valid_idx]

            arr_plot = np.nan_to_num(arr_plot, nan=0.0, posinf=0.0, neginf=0.0)
            arr_plot = arr_plot - np.median(arr_plot)

            if len(arr_plot) > 5:
                p95 = np.percentile(np.abs(arr_plot), 95)
            else:
                p95 = np.std(arr_plot)

            if not np.isfinite(p95) or p95 < 1e-6:
                p95 = 1.0

            display_signal = 35.0 * (arr_plot / p95) + offsets[i]
            display_signal = np.clip(
                display_signal,
                offsets[i] - 60,
                offsets[i] + 60,
            )

            self.eeg_curves[i].setData(t_ch, display_signal)

            if len(display_signal) > 0:
                self.eeg_labels[i].setPos(0.15, float(display_signal[-1]))

        self.plot_eeg.setXRange(-DISPLAY_SECONDS, 0, padding=0)

    def _refresh_bands(self):
        if self.plot_bands is None or self.band_bar is None:
            return

        bar_heights = [
            float(self.state.band_powers.get(name, 0.0))
            for name in self.band_names
        ]

        bar_heights = [
            0.0 if not np.isfinite(v) else max(0.0, v)
            for v in bar_heights
        ]

        self.band_bar.setOpts(height=bar_heights)

        ymax = max(50.0, max(bar_heights) * 1.15 if bar_heights else 50.0)
        self.plot_bands.setYRange(0, ymax, padding=0.02)

        for i, h in enumerate(bar_heights):
            text = f"{h:.2f}" if h < 1000 else f"{h:.2e}"
            self.band_value_labels[i].setText(text)
            self.band_value_labels[i].setPos(
                i,
                h + max(0.5, 0.03 * max(h, 1.0)),
            )

    def _refresh_blink_fast(self):
        """
        Dibuja las dos lineas de blink en la UI:
          - Curva roja (#F43F5E) en y=1.0: parpadeos detectados por ET.
          - Curva cyan (#22D3EE) en y=1.08: parpadeos detectados por ICA-EEG.

        Las curvas se construyen como funciones escalon a partir de las
        listas de eventos (start_ts, end_ts) publicadas por el processor:
            state.et_blink_events  <- get_blink_line_fast del ET pipeline
            state.ica_blink_events <- BlinkICADetector.process() del EEG

        La curva ICA solo se dibuja cuando state.ica_active es True (es decir,
        despues de que la calibracion con ojos abiertos haya entrenado el ICA).
        """
        if self.curve_blink is None:
            return

        # Tomar el tiempo de referencia como el ultimo timestamp visible
        # (preferimos EEG si esta disponible, sino ET)
        t_now = 0.0
        if len(self.state.eeg_time) > 0:
            t_now = float(self.state.eeg_time[-1])
        elif len(self.state.eye_time) > 0:
            t_now = float(self.state.eye_time[-1])
        else:
            self.curve_blink.setData([], [])
            self.curve_blink_ica.setData([], [])
            return

        # --- Curva ET (rojo, y=1.0) ---
        et_events = list(getattr(self.state, "et_blink_events", []))
        t_rel_et, y_et = self._events_to_step_line(
            et_events, t_now, float(DISPLAY_SECONDS), y_value=1.0,
        )
        self.curve_blink.setData(t_rel_et, y_et)

        # --- Curva ICA (cyan, y=1.08) ---
        # Solo dibujar si ICA esta activo. Si no, curva vacia.
        if bool(getattr(self.state, "ica_active", False)):
            ica_events = list(getattr(self.state, "ica_blink_events", []))
            t_rel_ica, y_ica = self._events_to_step_line(
                ica_events, t_now, float(DISPLAY_SECONDS), y_value=1.08,
            )
            self.curve_blink_ica.setData(t_rel_ica, y_ica)
        else:
            self.curve_blink_ica.setData([], [])

        self.plot_blink.setXRange(-DISPLAY_SECONDS, 0, padding=0)

        # --- Contadores ET ---
        current_window = int(getattr(self.state, "blink_count", 0))
        delta = current_window - self._blink_count_prev_window

        if delta > 0:
            self._blink_count_accum += delta

        self._blink_count_prev_window = current_window

        self.lbl_blinks.setText(str(self._blink_count_accum))
        self.lbl_blink_rate.setText(
            f"{getattr(self.state, 'blink_rate_per_min', 0.0):.1f} / min"
        )

        # --- Contadores ICA ---
        if bool(getattr(self.state, "ica_active", False)):
            ica_count = int(getattr(self.state, "ica_blink_count", 0))
            ica_rate = float(getattr(self.state, "ica_blink_rate_per_min", 0.0))
            self.lbl_ica_blinks.setText(str(ica_count))
            self.lbl_ica_rate.setText(f"{ica_rate:.1f} / min")
        else:
            self.lbl_ica_blinks.setText("n/a")
            self.lbl_ica_rate.setText("n/a")

    def _events_to_step_line(
        self,
        events,
        t_end: float,
        window_s: float,
        y_value: float = 1.0,
    ):
        """
        Convierte una lista de eventos (start_ts, end_ts) en una linea
        escalon dibujable en pyqtgraph.

        Tiempos absolutos LSL -> tiempos relativos en [-window_s, 0].
        Valor 0 fuera de blink, y_value durante blink. y_value permite
        apilar dos curvas (ET en y=1.0, ICA en y=1.08) sin solaparlas.
        """
        t_start = t_end - window_s

        # Filtrar eventos que tienen interseccion con la ventana visible
        visible = [
            (s, e) for (s, e) in events
            if e >= t_start and s <= t_end
        ]

        if not visible:
            return np.array([-window_s, 0.0]), np.array([0.0, 0.0])

        # Ordenar por inicio
        visible.sort(key=lambda x: x[0])

        xs = [-window_s]
        ys = [0.0]

        eps = 1e-6
        for s, e in visible:
            s_rel = max(-window_s, s - t_end)
            e_rel = min(0.0, e - t_end)
            if e_rel < s_rel:
                continue
            xs.extend([s_rel - eps, s_rel, e_rel, e_rel + eps])
            ys.extend([0.0, y_value, y_value, 0.0])

        xs.append(0.0)
        ys.append(0.0)

        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    def _refresh_eye_slow(self):
        if len(self.state.pupil_left) <= 1 or len(self.state.pupil_right) <= 1:
            return

        left_arr = np.asarray(self.state.pupil_left, dtype=float)
        right_arr = np.asarray(self.state.pupil_right, dtype=float)
        eye_time = np.asarray(self.state.eye_time, dtype=float)

        n = min(len(left_arr), len(right_arr), len(eye_time))
        if n < 2:
            return

        left_arr  = left_arr[-n:]
        right_arr = right_arr[-n:]
        eye_time  = eye_time[-n:]

        t_rel_eye = eye_time - eye_time[-1]

        mask = t_rel_eye >= -DISPLAY_SECONDS
        if not np.any(mask):
            return

        t_plot = t_rel_eye[mask]
        l_plot = left_arr[mask].copy()
        r_plot = right_arr[mask].copy()

        self.curve_pupil_left.setData(
            t_plot, l_plot,
            connect=np.isfinite(l_plot).astype(np.uint8),
        )
        self.curve_pupil_right.setData(
            t_plot, r_plot,
            connect=np.isfinite(r_plot).astype(np.uint8),
        )
        self.plot_eye.setXRange(-DISPLAY_SECONDS, 0, padding=0)

        valid = np.concatenate([
            l_plot[np.isfinite(l_plot)],
            r_plot[np.isfinite(r_plot)],
        ])

        if len(valid) > 0:
            ymin = float(np.min(valid)) - 0.5
            ymax = float(np.max(valid)) + 0.5

            if ymax <= ymin:
                ymax = ymin + 1.0

            self.plot_eye.setYRange(ymin, ymax)

        v_left = float(np.nanmean(l_plot)) if np.any(np.isfinite(l_plot)) else np.nan
        if np.isfinite(v_left):
            self.lbl_pupil_left.setText(f"{v_left:.2f} mm")

        v_right = float(np.nanmean(r_plot)) if np.any(np.isfinite(r_plot)) else np.nan
        if np.isfinite(v_right):
            self.lbl_pupil_right.setText(f"{v_right:.2f} mm")
