#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple

from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer, QPointF
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QSizePolicy,
    QProgressBar, QTabWidget, QDoubleSpinBox, QSpinBox, QSlider
)

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

import wavegen
import fhss_hopset_gen


class ChartView(QChartView):
    def __init__(self, chart):
        super().__init__(chart)
        self._last_pos = None

    def wheelEvent(self, event):
        factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self.chart().zoom(factor)
        self.chart().update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_pos = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last_pos is not None and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.chart().scroll(-delta.x(), delta.y())
            self.chart().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


WAVEFORM_CHOICES = [
    "noise", "cw", "multitone", "two_tone",
    "am", "fm", "pm",
    "ask", "ook",
    "fsk2", "gfsk",
    "bpsk", "qpsk", "qam16", "qam64",
    "ofdm", "chirp", "fhss",
]


def safe_eval_float(expr: str, default: Optional[float]) -> Optional[float]:
    """Matches your prior behavior: supports entries like 20e6."""
    try:
        return float(eval(expr, {"__builtins__": None}, {}))
    except Exception:
        return default


def generate_single_waveform(
    wf: str,
    fs: float,
    dur: float,
    sym_rate: float,
    level_for_this: float,
    pulse: str,
    rolloff: float,
    span: int,
    tonefreq: float,
    modfreq: float,
    depth: float,
    dev: float,
    phase_dev: float,
    bw: float,
    freqs: List[float],
    centerfreq: float,
    spacing: float,
    a0: float,
    a1: float,
    duty: float,
    f0: float,
    f1: float,
    h: float,
    bt: float,
    fstart: float,
    fend: float,
    hoprate: float,
    fftsize: int,
    cplen: int,
    burst_period: float,
    burst_duty: float,
    burst_rise: float,
    burst_fall: float,
    packet_len: float,
    guard_time: float,
):
    if wf == "noise":
        x = wavegen.gen_noise(fs, dur, bw, level_for_this)
    elif wf == "cw":
        x = wavegen.gen_cw(fs, dur, tonefreq, level_for_this)
    elif wf == "multitone":
        if not freqs:
            raise ValueError("multitone requires a non-empty frequency list.")
        x = wavegen.gen_multitone(fs, dur, freqs, level_for_this)
    elif wf == "two_tone":
        x = wavegen.gen_two_tone(fs, dur, centerfreq, spacing, level_for_this)
    elif wf == "am":
        x = wavegen.gen_am(fs, dur, tonefreq, modfreq, depth, level_for_this)
    elif wf == "fm":
        x = wavegen.gen_fm(fs, dur, tonefreq, modfreq, dev, level_for_this)
    elif wf == "pm":
        x = wavegen.gen_pm(fs, dur, tonefreq, modfreq,
                           phase_dev, level_for_this)
    elif wf == "ask":
        x = wavegen.gen_ask(fs, dur, sym_rate, a0, a1,
                            level_for_this, pulse, rolloff, span)
    elif wf == "ook":
        x = wavegen.gen_ook(fs, dur, sym_rate, duty, level_for_this)
    elif wf == "fsk2":
        x = wavegen.gen_fsk2(fs, dur, sym_rate, f0, f1, level_for_this)
    elif wf == "gfsk":
        x = wavegen.gen_gfsk(fs, dur, sym_rate, h, bt, level_for_this)
    elif wf == "bpsk":
        x = wavegen.gen_bpsk(
            fs, dur, sym_rate, level_for_this, pulse, rolloff, span)
    elif wf == "qpsk":
        x = wavegen.gen_qpsk(
            fs, dur, sym_rate, level_for_this, pulse, rolloff, span)
    elif wf == "qam16":
        x = wavegen.gen_qam16(
            fs, dur, sym_rate, level_for_this, pulse, rolloff, span)
    elif wf == "qam64":
        x = wavegen.gen_qam64(
            fs, dur, sym_rate, level_for_this, pulse, rolloff, span)
    elif wf == "ofdm":
        x = wavegen.gen_ofdm(fs, dur, bw, fftsize, cplen, level_for_this)
    elif wf == "chirp":
        x = wavegen.gen_chirp(fs, dur, fstart, fend, level_for_this)
    elif wf == "fhss":
        if not freqs:
            raise ValueError(
                "fhss requires a non-empty hopset. (Use the hopset generator panel.)")
        x = wavegen.gen_fhss(fs, dur, hoprate, freqs, level_for_this)
    else:
        raise ValueError(f"Unknown waveform: {wf}")

    if packet_len > 0:
        x = wavegen.apply_packetization(x, fs, dur, packet_len, guard_time)
    elif burst_period > 0:
        x = wavegen.apply_burst_envelope(
            x, fs, burst_period, burst_duty, burst_rise, burst_fall)
    return x


@dataclass(frozen=True)
class JobParams:
    fs: float
    dur: float
    out_level: float
    sym_rate: float

    pulse: str
    rolloff: float
    span: int

    tonefreq: float
    modfreq: float
    depth: float
    dev: float
    phase_dev: float
    bw: float

    # multitone/fhss per waveform
    freqs1: List[float]
    freqs2: List[float]

    centerfreq: float
    spacing: float
    a0: float
    a1: float
    duty: float
    f0: float
    f1: float
    h: float
    bt: float
    fstart: float
    fend: float
    hoprate: float
    fftsize: int
    cplen: int
    burst_period: float
    burst_duty: float
    burst_rise: float
    burst_fall: float
    packet_len: float
    guard_time: float

    wf1: str
    wf2: str
    w1_weight: float
    w2_weight: float
    use_sum: bool

    outfile: str
    fmt: str

    sigmf_enabled: bool
    sigmf_desc: str
    sigmf_author: str
    sigmf_cf: Optional[float]


class GeneratorWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, params: JobParams):
        super().__init__()
        self.params = params

    def run(self):
        p = self.params
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if not p.use_sum:
                    x = generate_single_waveform(
                        p.wf1, p.fs, p.dur, p.sym_rate,
                        p.out_level, p.pulse, p.rolloff, p.span,
                        p.tonefreq, p.modfreq, p.depth, p.dev, p.phase_dev,
                        p.bw, p.freqs1, p.centerfreq, p.spacing,
                        p.a0, p.a1, p.duty, p.f0, p.f1, p.h, p.bt,
                        p.fstart, p.fend, p.hoprate, p.fftsize, p.cplen,
                        p.burst_period, p.burst_duty, p.burst_rise, p.burst_fall,
                        p.packet_len, p.guard_time
                    )
                else:
                    x1 = generate_single_waveform(
                        p.wf1, p.fs, p.dur, p.sym_rate,
                        1.0, p.pulse, p.rolloff, p.span,
                        p.tonefreq, p.modfreq, p.depth, p.dev, p.phase_dev,
                        p.bw, p.freqs1, p.centerfreq, p.spacing,
                        p.a0, p.a1, p.duty, p.f0, p.f1, p.h, p.bt,
                        p.fstart, p.fend, p.hoprate, p.fftsize, p.cplen,
                        p.burst_period, p.burst_duty, p.burst_rise, p.burst_fall,
                        p.packet_len, p.guard_time
                    )
                    x2 = generate_single_waveform(
                        p.wf2, p.fs, p.dur, p.sym_rate,
                        1.0, p.pulse, p.rolloff, p.span,
                        p.tonefreq, p.modfreq, p.depth, p.dev, p.phase_dev,
                        p.bw, p.freqs2, p.centerfreq, p.spacing,
                        p.a0, p.a1, p.duty, p.f0, p.f1, p.h, p.bt,
                        p.fstart, p.fend, p.hoprate, p.fftsize, p.cplen,
                        p.burst_period, p.burst_duty, p.burst_rise, p.burst_fall,
                        p.packet_len, p.guard_time
                    )
                    combo = wavegen.combine_waveforms(
                        [x1, x2], [p.w1_weight, p.w2_weight])
                    x = wavegen.normalize_to_level(combo, p.out_level)

                if p.fmt == "int16":
                    iq = wavegen.iq_to_int16_interleaved(x)
                elif p.fmt == "int8":
                    iq = wavegen.iq_to_int8_interleaved(x)
                else:
                    raise ValueError(f"Unknown format: {p.fmt}")

                iq.tofile(p.outfile)
                size_bytes = os.path.getsize(p.outfile)

                num_complex_samples = x.size
                waveform_label = p.wf1 if not p.use_sum else f"{p.wf1}+{p.wf2}"
                if p.sigmf_enabled:
                    params = {
                        "fs": p.fs,
                        "waveform": waveform_label,
                    "sym_rate": p.sym_rate,
                    "bw": p.bw,
                    "pulse": p.pulse,
                    "rolloff": p.rolloff,
                    "span": p.span,
                    "fmt": p.fmt,
                    "burst_period": p.burst_period,
                    "burst_duty": p.burst_duty,
                    "burst_rise": p.burst_rise,
                    "burst_fall": p.burst_fall,
                    "packet_len": p.packet_len,
                    "guard_time": p.guard_time,
                    "hw": "USRP X310" if p.fmt == "int16" else "HackRF One",
                }

                wavegen.write_sigmf_meta(
                    data_path=p.outfile,
                    fmt=p.fmt,
                    fs=p.fs,
                    num_samples=num_complex_samples,
                    waveform=waveform_label,
                    params=params,
                    center_freq=p.sigmf_cf,
                    description=p.sigmf_desc,
                    author=p.sigmf_author,
                )

                for w in caught:
                    self.log.emit(f"Warning: {w.message}")

                self.finished.emit(
                    f"Done. Wrote {size_bytes} bytes to {p.outfile} ({p.fmt})")
            except Exception as e:
                for w in caught:
                    self.log.emit(f"Warning: {w.message}")
                self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waveform Generator (wavegen) - PySide6")
        self.resize(1100, 820)

        # Analyzer state (init early to avoid attribute errors)
        self.analyzer_data: Optional[np.memmap] = None
        self.analyzer_num_samples = 0
        self.analyzer_scale = 1.0
        self.analyzer_path = ""
        self._analyzer_region_busy = False
        self.analyzer_center_freq = 0.0
        self.analyzer_total_duration = 0.0

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_generator_tab()
        self._build_analyzer_tab()

        # Worker thread handles
        self._thread: Optional[QThread] = None
        self._worker: Optional[GeneratorWorker] = None

        self._update_visibility()

    # ---------------- UI helpers ----------------

    def _row_with_units(self, edit: QLineEdit, units: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(edit)
        if units:
            u = QLabel(units)
            u.setStyleSheet("color: #666;")
            h.addWidget(u)
        h.addStretch(1)
        return w

    def _add_validated(self, edit: QLineEdit, label: str):
        self._validation_targets.append((edit, label))
        edit.textChanged.connect(self._validate_all)

    def _mark_valid(self, edit: QLineEdit, ok: bool, msg: str = ""):
        if ok:
            edit.setStyleSheet("")
            edit.setToolTip("")
        else:
            edit.setStyleSheet(
                "background: #ffecec; border: 1px solid #cc4444;")
            edit.setToolTip(msg)

    def _log(self, msg: str):
        self.log_text.append(msg)

    def _warn(self, msg: str):
        self._log(f"Warning: {msg}")

    def _parse_float(
        self,
        edit: QLineEdit,
        default: float,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        s = edit.text().strip()
        val = safe_eval_float(s, None)
        if val is None:
            if s:
                self._warn(f"{name} invalid '{s}'; using {default}.")
            else:
                self._warn(f"{name} empty; using {default}.")
            val = default
        if min_value is not None and val < min_value:
            self._warn(f"{name} too low; using {min_value}.")
            val = min_value
        if max_value is not None and val > max_value:
            self._warn(f"{name} too high; using {max_value}.")
            val = max_value
        return float(val)

    def _parse_int(
        self,
        edit: QLineEdit,
        default: int,
        name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        val = self._parse_float(
            edit, float(default), name,
            float(min_value) if min_value is not None else None,
            float(max_value) if max_value is not None else None,
        )
        return int(val)

    def _build_generator_tab(self):
        generator_tab = QWidget()
        generator_v = QVBoxLayout(generator_tab)
        generator_v.setContentsMargins(0, 0, 0, 0)
        generator_v.setSpacing(0)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.root_v = QVBoxLayout(content)
        self.root_v.setContentsMargins(12, 12, 12, 12)
        self.root_v.setSpacing(12)

        # Two columns
        self.columns = QWidget()
        self.columns_h = QHBoxLayout(self.columns)
        self.columns_h.setContentsMargins(0, 0, 0, 0)
        self.columns_h.setSpacing(12)

        self.left_col = QWidget()
        self.left_v = QVBoxLayout(self.left_col)
        self.left_v.setContentsMargins(0, 0, 0, 0)
        self.left_v.setSpacing(12)

        self.right_col = QWidget()
        self.right_v = QVBoxLayout(self.right_col)
        self.right_v.setContentsMargins(0, 0, 0, 0)
        self.right_v.setSpacing(12)

        self.columns_h.addWidget(self.left_col, 1)
        self.columns_h.addWidget(self.right_col, 1)

        self.root_v.addWidget(self.columns)
        scroll.setWidget(content)

        # Validation targets
        self._validation_targets: List[Tuple[QLineEdit, str]] = []

        # Build UI
        self._build_waveforms_box()     # left
        self._build_common_box()        # left

        self._build_output_box()        # right
        self._build_sigmf_box()         # right
        self._build_log_box()           # right

        self._build_params_boxes()      # left (dynamic panels)

        self.left_v.addStretch(1)
        self.right_v.addStretch(1)

        # Bottom fixed bar
        bottom = QWidget()
        bottom_h = QHBoxLayout(bottom)
        bottom_h.setContentsMargins(12, 6, 12, 10)
        bottom_h.setSpacing(10)

        self.status_label = QLabel("Ready.")
        self.status_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(220)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)

        self.auto_load_analyzer_chk = QCheckBox("Auto-load into Analyzer")
        self.auto_load_analyzer_chk.setChecked(False)

        bottom_h.addWidget(self.status_label)
        bottom_h.addWidget(self.progress)
        bottom_h.addWidget(self.auto_load_analyzer_chk)
        bottom_h.addWidget(self.generate_btn)

        generator_v.addWidget(scroll)
        generator_v.addWidget(bottom)

        self.tabs.addTab(generator_tab, "Generator")

    def _build_analyzer_tab(self):
        analyzer_tab = QWidget()
        analyzer_tab_v = QVBoxLayout(analyzer_tab)
        analyzer_tab_v.setContentsMargins(0, 0, 0, 0)
        analyzer_tab_v.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        analyzer_v = QVBoxLayout(content)
        analyzer_v.setContentsMargins(12, 12, 12, 12)
        analyzer_v.setSpacing(10)

        controls_box = QGroupBox("IQ Analyzer")
        controls_g = QGridLayout(controls_box)
        controls_g.setColumnStretch(1, 1)

        self.analyzer_path_edit = QLineEdit("")
        self.analyzer_browse_btn = QPushButton("Browse...")
        self.analyzer_load_btn = QPushButton("Load")
        self.analyzer_browse_btn.clicked.connect(self._browse_analyzer_file)
        self.analyzer_load_btn.clicked.connect(self._load_analyzer_file)

        file_row = QWidget()
        file_h = QHBoxLayout(file_row)
        file_h.setContentsMargins(0, 0, 0, 0)
        file_h.setSpacing(6)
        file_h.addWidget(self.analyzer_path_edit)
        file_h.addWidget(self.analyzer_browse_btn)
        file_h.addWidget(self.analyzer_load_btn)

        self.analyzer_fmt_cb = QComboBox()
        self.analyzer_fmt_cb.addItems(["int16", "int8"])
        self.analyzer_fmt_cb.setCurrentText("int16")

        self.analyzer_fs_spin = QDoubleSpinBox()
        self.analyzer_fs_spin.setRange(1.0, 1e9)
        self.analyzer_fs_spin.setDecimals(3)
        self.analyzer_fs_spin.setValue(20e6)
        self.analyzer_fs_spin.setSuffix(" Hz")
        self.analyzer_fs_spin.valueChanged.connect(
            self._schedule_analyzer_update)

        self.analyzer_start_spin = QDoubleSpinBox()
        self.analyzer_start_spin.setRange(0.0, 1e9)
        self.analyzer_start_spin.setDecimals(6)
        self.analyzer_start_spin.setValue(0.0)
        self.analyzer_start_spin.setSuffix(" s")
        self.analyzer_start_spin.valueChanged.connect(
            self._schedule_analyzer_update)

        self.analyzer_duration_spin = QDoubleSpinBox()
        self.analyzer_duration_spin.setRange(1e-6, 1e9)
        self.analyzer_duration_spin.setDecimals(6)
        self.analyzer_duration_spin.setValue(0.01)
        self.analyzer_duration_spin.setSuffix(" s")
        self.analyzer_duration_spin.valueChanged.connect(
            self._schedule_analyzer_update)

        self.analyzer_fft_spin = QSpinBox()
        self.analyzer_fft_spin.setRange(64, 262144)
        self.analyzer_fft_spin.setSingleStep(64)
        self.analyzer_fft_spin.setValue(4096)
        self.analyzer_fft_spin.valueChanged.connect(
            self._schedule_analyzer_update)

        self.analyzer_overlap_spin = QSpinBox()
        self.analyzer_overlap_spin.setRange(0, 90)
        self.analyzer_overlap_spin.setValue(50)
        self.analyzer_overlap_spin.setSuffix(" %")
        self.analyzer_overlap_spin.valueChanged.connect(
            self._schedule_analyzer_update)

        self.analyzer_units_cb = QComboBox()
        self.analyzer_units_cb.addItems(["dBFS", "dBm"])
        self.analyzer_units_cb.currentIndexChanged.connect(
            self._schedule_analyzer_update)

        self.analyzer_fullscale_dbm = QDoubleSpinBox()
        self.analyzer_fullscale_dbm.setRange(-200.0, 200.0)
        self.analyzer_fullscale_dbm.setDecimals(1)
        self.analyzer_fullscale_dbm.setValue(0.0)
        self.analyzer_fullscale_dbm.setSuffix(" dBm")
        self.analyzer_fullscale_dbm.valueChanged.connect(
            self._schedule_analyzer_update)

        controls_g.addWidget(QLabel("File:"), 0, 0)
        controls_g.addWidget(file_row, 0, 1, 1, 3)

        controls_g.addWidget(QLabel("Format:"), 1, 0)
        controls_g.addWidget(self.analyzer_fmt_cb, 1, 1)

        controls_g.addWidget(QLabel("Sample rate:"), 1, 2)
        controls_g.addWidget(self.analyzer_fs_spin, 1, 3)

        controls_g.addWidget(QLabel("Start time:"), 2, 0)
        controls_g.addWidget(self.analyzer_start_spin, 2, 1)

        controls_g.addWidget(QLabel("Window duration:"), 2, 2)
        controls_g.addWidget(self.analyzer_duration_spin, 2, 3)

        controls_g.addWidget(QLabel("FFT size:"), 3, 0)
        controls_g.addWidget(self.analyzer_fft_spin, 3, 1)

        controls_g.addWidget(QLabel("Waterfall overlap:"), 3, 2)
        controls_g.addWidget(self.analyzer_overlap_spin, 3, 3)

        controls_g.addWidget(QLabel("Spectrum units:"), 4, 0)
        controls_g.addWidget(self.analyzer_units_cb, 4, 1)

        controls_g.addWidget(QLabel("Full-scale ref:"), 4, 2)
        controls_g.addWidget(self.analyzer_fullscale_dbm, 4, 3)

        header_row = QWidget()
        header_h = QHBoxLayout(header_row)
        header_h.setContentsMargins(0, 0, 0, 0)
        header_h.setSpacing(12)
        header_h.addWidget(controls_box, 2)

        self.analyzer_meta_box = QGroupBox("SigMF Metadata")
        meta_g = QGridLayout(self.analyzer_meta_box)
        meta_g.setColumnStretch(1, 1)
        self.analyzer_meta_desc = QLabel("-")
        self.analyzer_meta_author = QLabel("-")
        self.analyzer_meta_freq = QLabel("-")
        self.analyzer_meta_dtype = QLabel("-")
        meta_g.addWidget(QLabel("Description:"), 0, 0)
        meta_g.addWidget(self.analyzer_meta_desc, 0, 1)
        meta_g.addWidget(QLabel("Author:"), 1, 0)
        meta_g.addWidget(self.analyzer_meta_author, 1, 1)
        meta_g.addWidget(QLabel("Center freq:"), 2, 0)
        meta_g.addWidget(self.analyzer_meta_freq, 2, 1)
        meta_g.addWidget(QLabel("Datatype:"), 3, 0)
        meta_g.addWidget(self.analyzer_meta_dtype, 3, 1)
        header_h.addWidget(self.analyzer_meta_box, 1)

        analyzer_v.addWidget(header_row)

        self.analyzer_slider = QSlider(Qt.Horizontal)
        self.analyzer_slider.setRange(0, 0)
        self.analyzer_slider.setValue(0)
        self.analyzer_slider.valueChanged.connect(
            self._on_analyzer_slider_changed)
        analyzer_v.addWidget(self.analyzer_slider)

        self.analyzer_plot_tabs = QTabWidget()
        self.analyzer_plot_tabs.setMinimumHeight(420)
        self.analyzer_plot_tabs.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.analyzer_plot_tabs.currentChanged.connect(
            self._schedule_analyzer_update)
        self.analyzer_plot_tabs.currentChanged.connect(
            lambda _idx: self._refresh_analyzer())

        time_tab = QWidget()
        time_v = QVBoxLayout(time_tab)
        time_v.setContentsMargins(0, 0, 0, 0)
        self.analyzer_time_series_i = QLineSeries()
        self.analyzer_time_series_q = QLineSeries()
        self.analyzer_time_series_i.setUseOpenGL(False)
        self.analyzer_time_series_q.setUseOpenGL(False)
        self.analyzer_time_series_i.setName("I")
        self.analyzer_time_series_q.setName("Q")
        self.analyzer_time_chart = QChart()
        self.analyzer_time_chart.setTitle("Time Domain (I/Q)")
        self.analyzer_time_chart.addSeries(self.analyzer_time_series_i)
        self.analyzer_time_chart.addSeries(self.analyzer_time_series_q)
        self.analyzer_time_axis_x = QValueAxis()
        self.analyzer_time_axis_x.setTitleText("Time (s)")
        self.analyzer_time_axis_y = QValueAxis()
        self.analyzer_time_axis_y.setTitleText("Amplitude")
        self.analyzer_time_chart.addAxis(
            self.analyzer_time_axis_x, Qt.AlignBottom)
        self.analyzer_time_chart.addAxis(
            self.analyzer_time_axis_y, Qt.AlignLeft)
        self.analyzer_time_series_i.attachAxis(self.analyzer_time_axis_x)
        self.analyzer_time_series_i.attachAxis(self.analyzer_time_axis_y)
        self.analyzer_time_series_q.attachAxis(self.analyzer_time_axis_x)
        self.analyzer_time_series_q.attachAxis(self.analyzer_time_axis_y)
        self.analyzer_time_view = ChartView(self.analyzer_time_chart)
        self.analyzer_time_view.setMinimumHeight(300)
        time_v.addWidget(self.analyzer_time_view)
        self.analyzer_plot_tabs.addTab(time_tab, "Time")

        spec_tab = QWidget()
        spec_v = QVBoxLayout(spec_tab)
        spec_v.setContentsMargins(0, 0, 0, 0)
        self.analyzer_spec_series = QLineSeries()
        self.analyzer_spec_series.setUseOpenGL(False)
        self.analyzer_spec_chart = QChart()
        self.analyzer_spec_chart.setTitle("Spectrum")
        self.analyzer_spec_chart.addSeries(self.analyzer_spec_series)
        self.analyzer_spec_chart.legend().hide()
        self.analyzer_spec_axis_x = QValueAxis()
        self.analyzer_spec_axis_x.setTitleText("Frequency (Hz)")
        self.analyzer_spec_axis_x.setLabelFormat("%.0f")
        self.analyzer_spec_axis_x.setTickCount(7)
        self.analyzer_spec_axis_y = QValueAxis()
        self.analyzer_spec_axis_y.setTitleText("Magnitude (dBFS)")
        self.analyzer_spec_chart.addAxis(
            self.analyzer_spec_axis_x, Qt.AlignBottom)
        self.analyzer_spec_chart.addAxis(
            self.analyzer_spec_axis_y, Qt.AlignLeft)
        self.analyzer_spec_series.attachAxis(self.analyzer_spec_axis_x)
        self.analyzer_spec_series.attachAxis(self.analyzer_spec_axis_y)
        self.analyzer_spec_view = ChartView(self.analyzer_spec_chart)
        self.analyzer_spec_view.setMinimumHeight(300)
        spec_v.addWidget(self.analyzer_spec_view)
        self.analyzer_plot_tabs.addTab(spec_tab, "Spectrum")

        wf_tab = QWidget()
        wf_v = QVBoxLayout(wf_tab)
        wf_v.setContentsMargins(0, 0, 0, 0)
        self.analyzer_wf_fig = Figure(figsize=(5, 3), tight_layout=True)
        self.analyzer_wf_canvas = FigureCanvas(self.analyzer_wf_fig)
        self.analyzer_wf_ax = self.analyzer_wf_fig.add_subplot(1, 1, 1)
        self.analyzer_wf_ax.set_title("Waterfall")
        self.analyzer_wf_ax.set_xlabel("Time (s)")
        self.analyzer_wf_ax.set_ylabel("Frequency (Hz)")
        wf_v.addWidget(self.analyzer_wf_canvas)
        self.analyzer_plot_tabs.addTab(wf_tab, "Waterfall")

        analyzer_v.addWidget(self.analyzer_plot_tabs, 1)

        metrics_box = QGroupBox("Measurements (Selected Window)")
        metrics_v = QVBoxLayout(metrics_box)
        metrics_v.setContentsMargins(8, 8, 8, 8)
        self.analyzer_metrics_text = QTextEdit()
        self.analyzer_metrics_text.setReadOnly(True)
        self.analyzer_metrics_text.setMinimumHeight(140)
        metrics_v.addWidget(self.analyzer_metrics_text)
        analyzer_v.addWidget(metrics_box)

        analyzer_v.setStretchFactor(self.analyzer_plot_tabs, 1)

        scroll.setWidget(content)
        analyzer_tab_v.addWidget(scroll)

        self.tabs.addTab(analyzer_tab, "Analyzer")

        self.analyzer_update_timer = QTimer(self)
        self.analyzer_update_timer.setSingleShot(True)
        self.analyzer_update_timer.setInterval(150)
        self.analyzer_update_timer.timeout.connect(self._refresh_analyzer)

    def _browse_analyzer_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open IQ file",
            "",
            "Binary files (*.bin);;All files (*.*)",
        )
        if path:
            self.analyzer_path_edit.setText(path)

    def _load_analyzer_file(self):
        path = self.analyzer_path_edit.text().strip()
        if not path:
            QMessageBox.critical(
                self, "Error", "Please select a file to load.")
            return
        if not os.path.exists(path):
            QMessageBox.critical(self, "Error", "File does not exist.")
            return

        meta_path = path + ".sigmf-meta"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                global_info = meta.get("global", {})
                captures = meta.get("captures", [])
                c0 = captures[0] if captures else {}
                sample_rate = global_info.get("core:sample_rate")
                datatype = global_info.get("core:datatype")
                description = global_info.get("core:description", "")
                author = global_info.get("core:author", "")
                center_freq = c0.get("core:frequency")
                if sample_rate:
                    self.analyzer_fs_spin.setValue(float(sample_rate))
                if datatype == "ci16_le":
                    self.analyzer_fmt_cb.setCurrentText("int16")
                elif datatype == "ci8_le":
                    self.analyzer_fmt_cb.setCurrentText("int8")
                self.analyzer_center_freq = float(
                    center_freq) if center_freq else 0.0
                self.analyzer_meta_desc.setText(description or "-")
                self.analyzer_meta_author.setText(author or "-")
                self.analyzer_meta_freq.setText(
                    f"{self.analyzer_center_freq:.3f} Hz" if center_freq else "-")
                self.analyzer_meta_dtype.setText(datatype or "-")
                self._log(
                    f"Loaded SigMF metadata: {os.path.basename(meta_path)}")
            except Exception as e:
                self._warn(f"Failed to read SigMF metadata: {e}")

        fmt = self.analyzer_fmt_cb.currentText()
        dtype = np.int16 if fmt == "int16" else np.int8
        self.analyzer_scale = 32767.0 if fmt == "int16" else 127.0
        itemsize = np.dtype(dtype).itemsize

        size_bytes = os.path.getsize(path)
        if size_bytes < 2 * itemsize:
            QMessageBox.critical(self, "Error", "File is too small.")
            return

        if size_bytes % (2 * itemsize) != 0:
            self._warn(
                "File size not aligned to I/Q pairs; trimming trailing bytes.")
            size_bytes = size_bytes - (size_bytes % (2 * itemsize))

        num_samples = size_bytes // (2 * itemsize)
        if num_samples <= 0:
            QMessageBox.critical(self, "Error", "No complex samples found.")
            return

        self.analyzer_data = np.memmap(
            path, dtype=dtype, mode="r", shape=(num_samples * 2,))
        self.analyzer_num_samples = num_samples
        self.analyzer_path = path

        fs = self.analyzer_fs_spin.value()
        if fs <= 0:
            fs = 1.0
            self.analyzer_fs_spin.setValue(fs)
            self._warn("Sample rate must be > 0; using 1.0 Hz.")

        duration = num_samples / fs
        self.analyzer_total_duration = duration
        self._log(
            f"Loaded {num_samples} samples from '{os.path.basename(path)}' "
            f"({fmt}, {size_bytes} bytes). Duration {duration:.6f} s.")
        preview_count = min(8, num_samples)
        if preview_count > 0:
            preview = self.analyzer_data[:preview_count * 2].astype(np.int64)
            self._log(f"First IQ pairs: {preview.tolist()}")
        self.analyzer_start_spin.setRange(0.0, max(0.0, duration))
        self.analyzer_duration_spin.setRange(1e-6, max(1e-6, duration))
        self.analyzer_start_spin.setValue(0.0)
        self.analyzer_duration_spin.setValue(duration)

        self.analyzer_time_axis_x.setRange(0.0, duration)
        self.analyzer_time_axis_y.setRange(-1.0, 1.0)
        half_bw = fs / 2.0
        self.analyzer_spec_axis_x.setRange(
            self.analyzer_center_freq - half_bw,
            self.analyzer_center_freq + half_bw,
        )

        self.analyzer_slider.setRange(0, max(0, num_samples - 1))
        self.analyzer_slider.setValue(0)

        self._update_analyzer_overview()
        self._refresh_analyzer()

    def _on_analyzer_slider_changed(self, value: int):
        fs = self.analyzer_fs_spin.value()
        if fs <= 0:
            return
        self.analyzer_start_spin.blockSignals(True)
        self.analyzer_start_spin.setValue(value / fs)
        self.analyzer_start_spin.blockSignals(False)
        self._schedule_analyzer_update()

    def _schedule_analyzer_update(self):
        if not hasattr(self, "analyzer_data"):
            return
        if self.analyzer_data is None:
            return
        if self.analyzer_update_timer.isActive():
            self.analyzer_update_timer.stop()
        self.analyzer_update_timer.start()

    def _update_analyzer_overview(self):
        return

    def _get_window_samples(self) -> Tuple[int, int]:
        if self.analyzer_data is None:
            return 0, 0
        fs = self.analyzer_fs_spin.value()
        if fs <= 0:
            fs = 1.0
        start_s = self.analyzer_start_spin.value()
        dur_s = self.analyzer_duration_spin.value()
        start = int(start_s * fs)
        count = int(dur_s * fs)
        count = max(1, count)
        if start < 0:
            start = 0
        if start >= self.analyzer_num_samples:
            start = max(0, self.analyzer_num_samples - count)
        end = min(self.analyzer_num_samples, start + count)
        if end <= start:
            end = min(self.analyzer_num_samples, start + 1)
        start = max(0, end - count)
        return start, end

    def _extract_window(self, start: int, end: int) -> np.ndarray:
        raw = self.analyzer_data
        start_iq = start * 2
        end_iq = end * 2
        i = raw[start_iq:end_iq:2].astype(np.float32)
        q = raw[start_iq + 1:end_iq:2].astype(np.float32)
        return (i + 1j * q) / self.analyzer_scale

    def _compute_spectrum(self, x: np.ndarray, fs: float, fft_size: int):
        if x.size == 0:
            return np.array([]), np.array([])
        if fft_size < 64:
            fft_size = 64
        if x.size < fft_size:
            pad = np.zeros(fft_size - x.size, dtype=x.dtype)
            xw = np.concatenate([x, pad])
        else:
            xw = x[:fft_size]
        window = np.hanning(xw.size)
        xw = xw * window
        X = np.fft.fftshift(np.fft.fft(xw))
        mag = np.abs(X) / max(1.0, np.sum(window) / 2.0)
        mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
        freqs = np.fft.fftshift(np.fft.fftfreq(xw.size, d=1.0 / fs))
        return freqs, mag_db

    def _compute_waterfall(self, x: np.ndarray, fs: float, fft_size: int, overlap: int):
        if x.size < fft_size or fft_size < 64:
            return np.empty((0, 0)), np.array([]), np.array([])
        overlap = max(0, min(90, overlap))
        hop = max(1, int(fft_size * (1.0 - overlap / 100.0)))
        max_cols = 400
        max_samples = fft_size + hop * (max_cols - 1)
        if x.size > max_samples:
            x = x[:max_samples]
        num_cols = 1 + (x.size - fft_size) // hop
        if num_cols <= 0:
            return np.empty((0, 0)), np.array([]), np.array([])
        window = np.hanning(fft_size)
        spec = np.zeros((num_cols, fft_size), dtype=np.float32)
        for i in range(num_cols):
            seg = x[i * hop:i * hop + fft_size] * window
            X = np.fft.fftshift(np.fft.fft(seg))
            p = np.abs(X) ** 2
            spec[i, :] = 10 * np.log10(np.maximum(p, 1e-12))
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / fs))
        times = np.arange(num_cols) * hop / fs
        return spec, freqs, times

    def _refresh_analyzer(self):
        if self.analyzer_data is None:
            return

        fs = self.analyzer_fs_spin.value()
        if fs <= 0:
            fs = 1.0
            self.analyzer_fs_spin.setValue(fs)
            self._warn("Sample rate must be > 0; using 1.0 Hz.")

        freq_offset = float(self.analyzer_center_freq or 0.0)
        units = self.analyzer_units_cb.currentText()
        fullscale_dbm = self.analyzer_fullscale_dbm.value()

        self._update_analyzer_overview()

        start, end = self._get_window_samples()
        duration = (end - start) / fs
        start_s = start / fs

        self.analyzer_start_spin.blockSignals(True)
        self.analyzer_duration_spin.blockSignals(True)
        self.analyzer_start_spin.setValue(start_s)
        self.analyzer_duration_spin.setValue(max(1e-6, duration))
        self.analyzer_start_spin.blockSignals(False)
        self.analyzer_duration_spin.blockSignals(False)

        if hasattr(self, "analyzer_slider"):
            self.analyzer_slider.blockSignals(True)
            self.analyzer_slider.setValue(start)
            self.analyzer_slider.blockSignals(False)

        x = self._extract_window(start, end)
        if x.size == 0:
            return

        max_points = 20000
        step = max(1, x.size // max_points)
        x_plot = x[::step]
        t = start_s + (np.arange(x_plot.size) * step / fs)
        i_vals = np.real(x_plot)
        q_vals = np.imag(x_plot)
        time_points_i = [QPointF(float(tt), float(ii))
                         for tt, ii in zip(t, i_vals)]
        time_points_q = [QPointF(float(tt), float(qq))
                         for tt, qq in zip(t, q_vals)]
        self.analyzer_time_series_i.replace(time_points_i)
        self.analyzer_time_series_q.replace(time_points_q)
        if t.size:
            self.analyzer_time_axis_x.setRange(float(t[0]), float(t[-1]))
            ymax = float(np.max(np.abs(x_plot))) if x_plot.size else 1.0
            if ymax <= 0:
                ymax = 1.0
            self.analyzer_time_axis_y.setRange(-ymax * 1.1, ymax * 1.1)
        self.analyzer_time_chart.update()
        self.analyzer_time_view.repaint()

        fft_size = self.analyzer_fft_spin.value()
        freqs, mag_db = self._compute_spectrum(x, fs, fft_size)
        if units == "dBm":
            mag_db = mag_db + fullscale_dbm
            self.analyzer_spec_axis_y.setTitleText("Magnitude (dBm)")
        else:
            self.analyzer_spec_axis_y.setTitleText("Magnitude (dBFS)")
        freqs = freqs + freq_offset
        spec_points = [QPointF(float(ff), float(mm))
                       for ff, mm in zip(freqs, mag_db)]
        self.analyzer_spec_series.replace(spec_points)
        if freqs.size:
            self.analyzer_spec_axis_x.setRange(
                float(freqs[0]), float(freqs[-1]))
            min_db = float(np.min(mag_db)) if mag_db.size else -120.0
            max_db = float(np.max(mag_db)) if mag_db.size else 0.0
            if min_db == max_db:
                min_db -= 1.0
                max_db += 1.0
            self.analyzer_spec_axis_y.setRange(min_db, max_db)
        self.analyzer_spec_chart.update()
        self.analyzer_spec_view.repaint()

        if self.analyzer_plot_tabs.currentIndex() == 2:
            spec, wf_freqs, wf_times = self._compute_waterfall(
                x, fs, fft_size, self.analyzer_overlap_spin.value())
            if spec.size > 0 and wf_times.size > 1 and wf_freqs.size > 1:
                if units == "dBm":
                    spec = spec + fullscale_dbm
                self.analyzer_wf_ax.clear()
                self.analyzer_wf_ax.set_title("Waterfall")
                self.analyzer_wf_ax.set_xlabel("Time (s)")
                self.analyzer_wf_ax.set_ylabel("Frequency (Hz)")
                wf_freqs = wf_freqs + freq_offset
                extent = [
                    float(wf_times[0]),
                    float(wf_times[-1]),
                    float(wf_freqs[0]),
                    float(wf_freqs[-1]),
                ]
                self.analyzer_wf_ax.imshow(
                    spec.T,
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                )
                self.analyzer_wf_ax.set_xlim(extent[0], extent[1])
                self.analyzer_wf_ax.set_ylim(extent[2], extent[3])
                self.analyzer_wf_ax.ticklabel_format(axis="y", style="plain")
                self.analyzer_wf_canvas.draw_idle()

        self._update_measurements(x, fs, freqs, mag_db)

    def _update_measurements(self, x: np.ndarray, fs: float, freqs: np.ndarray, mag_db: np.ndarray):
        if x.size == 0:
            self.analyzer_metrics_text.setPlainText("No samples loaded.")
            return

        rms = np.sqrt(np.mean(np.abs(x) ** 2))
        peak = np.max(np.abs(x))
        crest = peak / rms if rms > 0 else 0.0
        mean_i = float(np.mean(np.real(x)))
        mean_q = float(np.mean(np.imag(x)))
        dc_mag = np.sqrt(mean_i ** 2 + mean_q ** 2)

        rms_dbfs = 20 * np.log10(max(rms, 1e-12))
        peak_dbfs = 20 * np.log10(max(peak, 1e-12))
        crest_db = 20 * np.log10(max(crest, 1e-12))

        peak_freq = 0.0
        occ_bw = 0.0
        p2m_db = 0.0
        if freqs.size and mag_db.size:
            peak_idx = int(np.argmax(mag_db))
            peak_freq = float(freqs[peak_idx])

            power = 10 ** (mag_db / 10.0)
            cum = np.cumsum(power)
            total = cum[-1] if cum.size else 0.0
            if total > 0:
                low_idx = int(np.searchsorted(cum, total * 0.005))
                high_idx = int(np.searchsorted(cum, total * 0.995))
                low_idx = min(max(low_idx, 0), freqs.size - 1)
                high_idx = min(max(high_idx, low_idx), freqs.size - 1)
                occ_bw = float(freqs[high_idx] - freqs[low_idx])

            median_power = np.median(power)
            peak_power = np.max(power)
            p2m_db = 10 * np.log10(max(peak_power, 1e-12) /
                                   max(median_power, 1e-12))

        lines = [
            f"Samples: {x.size}",
            f"Duration: {x.size / fs:.6f} s",
            f"RMS: {rms:.6f} ({rms_dbfs:.2f} dBFS)",
            f"Peak: {peak:.6f} ({peak_dbfs:.2f} dBFS)",
            f"Crest factor: {crest:.3f} ({crest_db:.2f} dB)",
            f"Mean I/Q: {mean_i:.6f}, {mean_q:.6f}",
            f"DC magnitude: {dc_mag:.6f}",
            f"Peak frequency: {peak_freq:.3f} Hz",
            f"Occupied BW (99%): {occ_bw:.3f} Hz",
            f"Peak-to-median: {p2m_db:.2f} dB",
        ]
        self.analyzer_metrics_text.setPlainText("\n".join(lines))

    def _validate_all(self):
        for edit, label in self._validation_targets:
            s = edit.text().strip()
            if not s:
                self._mark_valid(edit, True)
                continue
            v = safe_eval_float(s, None)
            self._mark_valid(edit, v is not None,
                             f"Invalid {label}. Examples: 1e6, 2.5, 20e6")

    # ---------------- Build sections ----------------

    def _build_waveforms_box(self):
        box = QGroupBox("Waveforms")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)

        g.addWidget(QLabel("Waveform 1:"), 0, 0)
        self.wf1_cb = QComboBox()
        self.wf1_cb.addItems(WAVEFORM_CHOICES)
        self.wf1_cb.setCurrentText("noise")
        self.wf1_cb.currentTextChanged.connect(self._update_visibility)
        g.addWidget(self.wf1_cb, 0, 1)

        g.addWidget(QLabel("W1 weight:"), 0, 2)
        self.w1_weight_edit = QLineEdit("1.0")
        self.w1_weight_edit.setMaximumWidth(110)
        self._add_validated(self.w1_weight_edit, "W1 weight")
        g.addWidget(self.w1_weight_edit, 0, 3)

        self.use_sum_chk = QCheckBox("Enable Waveform 2 (sum)")
        self.use_sum_chk.toggled.connect(self._update_visibility)
        g.addWidget(self.use_sum_chk, 1, 0, 1, 4)

        self.wf2_row = QWidget()
        r = QGridLayout(self.wf2_row)
        r.setContentsMargins(0, 0, 0, 0)
        r.setColumnStretch(1, 1)

        r.addWidget(QLabel("Waveform 2:"), 0, 0)
        self.wf2_cb = QComboBox()
        self.wf2_cb.addItems(WAVEFORM_CHOICES)
        self.wf2_cb.setCurrentText("noise")
        self.wf2_cb.currentTextChanged.connect(self._update_visibility)
        r.addWidget(self.wf2_cb, 0, 1)

        r.addWidget(QLabel("W2 weight:"), 0, 2)
        self.w2_weight_edit = QLineEdit("0.5")
        self.w2_weight_edit.setMaximumWidth(110)
        self._add_validated(self.w2_weight_edit, "W2 weight")
        r.addWidget(self.w2_weight_edit, 0, 3)

        g.addWidget(self.wf2_row, 2, 0, 1, 4)

        self.left_v.addWidget(box)

    def _build_common_box(self):
        box = QGroupBox("Common")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)

        self.fs_edit = QLineEdit("20e6")
        self.dur_edit = QLineEdit("0.05")
        self.level_edit = QLineEdit("0.3")
        self.symrate_edit = QLineEdit("1e6")

        self._add_validated(self.fs_edit, "sample rate")
        self._add_validated(self.dur_edit, "duration")
        self._add_validated(self.level_edit, "level")
        self._add_validated(self.symrate_edit, "symbol rate")

        g.addWidget(QLabel("Sample rate:"), 0, 0)
        g.addWidget(self._row_with_units(self.fs_edit, "Hz"), 0, 1, 1, 3)

        g.addWidget(QLabel("Duration:"), 1, 0)
        g.addWidget(self._row_with_units(self.dur_edit, "s"), 1, 1, 1, 3)

        g.addWidget(QLabel("Output level (RMS 0-1):"), 2, 0)
        g.addWidget(self.level_edit, 2, 1)

        g.addWidget(QLabel("Symbol rate (digital):"), 3, 0)
        g.addWidget(self._row_with_units(self.symrate_edit, "Hz"), 3, 1, 1, 3)

        self.left_v.addWidget(box)

    def _build_output_box(self):
        box = QGroupBox("Output")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)

        self.outfile_edit = QLineEdit("output.bin")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)

        g.addWidget(QLabel("File:"), 0, 0)
        g.addWidget(self.outfile_edit, 0, 1)
        g.addWidget(self.browse_btn, 0, 2)

        self.fmt_cb = QComboBox()
        self.fmt_cb.addItems(["int16", "int8"])
        self.fmt_cb.setCurrentText("int16")

        g.addWidget(QLabel("Format:"), 1, 0)
        g.addWidget(self.fmt_cb, 1, 1, 1, 2)

        self.right_v.addWidget(box)

    def _build_sigmf_box(self):
        box = QGroupBox("SigMF")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        note = QLabel(
            "SigMF metadata is always written alongside the IQ file.")
        note.setStyleSheet("color: #666;")
        v.addWidget(note)

        self.sigmf_fields = QWidget()
        g = QGridLayout(self.sigmf_fields)
        g.setContentsMargins(0, 0, 0, 0)
        g.setColumnStretch(1, 1)

        self.sigmf_desc_edit = QLineEdit("")
        self.sigmf_author_edit = QLineEdit("wavegen-gui")
        self.sigmf_cf_edit = QLineEdit("")
        self._add_validated(self.sigmf_cf_edit, "SigMF center freq")

        g.addWidget(QLabel("Description:"), 0, 0)
        g.addWidget(self.sigmf_desc_edit, 0, 1)

        g.addWidget(QLabel("Author:"), 1, 0)
        g.addWidget(self.sigmf_author_edit, 1, 1)

        g.addWidget(QLabel("Center freq:"), 2, 0)
        g.addWidget(self._row_with_units(self.sigmf_cf_edit, "Hz"), 2, 1)

        v.addWidget(self.sigmf_fields)

        btn_row = QWidget()
        btn_h = QHBoxLayout(btn_row)
        btn_h.setContentsMargins(0, 0, 0, 0)
        self.open_meta_btn = QPushButton("Open .sigmf-meta...")
        self.open_meta_btn.clicked.connect(self.open_sigmf_meta)
        btn_h.addWidget(self.open_meta_btn)
        btn_h.addStretch(1)
        v.addWidget(btn_row)

        self.meta_text = QTextEdit()
        self.meta_text.setMinimumHeight(150)
        v.addWidget(self.meta_text)

        self.right_v.addWidget(box)

    def _build_log_box(self):
        box = QGroupBox("Log")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        v.addWidget(self.log_text)

        self.right_v.addWidget(box)

    def _build_params_boxes(self):
        # Pulse shaping
        self.pulse_box = QGroupBox("Pulse shaping (PSK/QAM/ASK)")
        g = QGridLayout(self.pulse_box)
        g.setColumnStretch(1, 1)

        self.pulse_cb = QComboBox()
        self.pulse_cb.addItems(["rect", "sinc"])
        self.pulse_cb.setCurrentText("rect")

        self.rolloff_edit = QLineEdit("0.25")
        self.span_edit = QLineEdit("6")
        self._add_validated(self.rolloff_edit, "rolloff")
        self._add_validated(self.span_edit, "span")

        g.addWidget(QLabel("Pulse:"), 0, 0)
        g.addWidget(self.pulse_cb, 0, 1)

        g.addWidget(QLabel("Rolloff:"), 1, 0)
        g.addWidget(self.rolloff_edit, 1, 1)

        g.addWidget(QLabel("Span:"), 2, 0)
        g.addWidget(self._row_with_units(self.span_edit, "symbols"), 2, 1)

        self.left_v.addWidget(self.pulse_box)

        # Analog/tone
        self.analog_box = QGroupBox("Tone / AM / FM / PM")
        g = QGridLayout(self.analog_box)
        g.setColumnStretch(1, 1)

        self.tonefreq_edit = QLineEdit("0")
        self.modfreq_edit = QLineEdit("1e3")
        self.depth_edit = QLineEdit("0.5")
        self.dev_edit = QLineEdit("5e3")
        self.phase_dev_edit = QLineEdit("1.0")

        for e, name in [
            (self.tonefreq_edit, "tone freq"),
            (self.modfreq_edit, "mod freq"),
            (self.depth_edit, "AM depth"),
            (self.dev_edit, "FM deviation"),
            (self.phase_dev_edit, "PM phase dev"),
        ]:
            self._add_validated(e, name)

        g.addWidget(QLabel("Tone/carrier:"), 0, 0)
        g.addWidget(self._row_with_units(self.tonefreq_edit, "Hz"), 0, 1)

        g.addWidget(QLabel("Mod freq:"), 1, 0)
        g.addWidget(self._row_with_units(self.modfreq_edit, "Hz"), 1, 1)

        g.addWidget(QLabel("AM depth (0-1):"), 2, 0)
        g.addWidget(self.depth_edit, 2, 1)

        g.addWidget(QLabel("FM deviation:"), 3, 0)
        g.addWidget(self._row_with_units(self.dev_edit, "Hz"), 3, 1)

        g.addWidget(QLabel("PM phase dev:"), 4, 0)
        g.addWidget(self._row_with_units(self.phase_dev_edit, "rad"), 4, 1)

        self.left_v.addWidget(self.analog_box)

        # BW
        self.bw_box = QGroupBox("Bandwidth (Noise / OFDM)")
        g = QGridLayout(self.bw_box)
        g.setColumnStretch(1, 1)
        self.bw_edit = QLineEdit("10e6")
        self._add_validated(self.bw_edit, "bandwidth")
        g.addWidget(QLabel("Bandwidth:"), 0, 0)
        g.addWidget(self._row_with_units(self.bw_edit, "Hz"), 0, 1)
        self.left_v.addWidget(self.bw_box)

        # OFDM
        self.ofdm_box = QGroupBox("OFDM")
        g = QGridLayout(self.ofdm_box)
        g.setColumnStretch(1, 1)
        self.fftsize_edit = QLineEdit("1024")
        self.cplen_edit = QLineEdit("128")
        self._add_validated(self.fftsize_edit, "FFT size")
        self._add_validated(self.cplen_edit, "CP length")
        g.addWidget(QLabel("FFT size:"), 0, 0)
        g.addWidget(self.fftsize_edit, 0, 1)
        g.addWidget(QLabel("CP length:"), 1, 0)
        g.addWidget(self.cplen_edit, 1, 1)
        self.left_v.addWidget(self.ofdm_box)

        # Multitone freqs ONLY
        self.freqs_box = QGroupBox("Multitone frequencies")
        g = QGridLayout(self.freqs_box)
        g.setColumnStretch(1, 1)
        self.freqs_edit = QLineEdit("")
        g.addWidget(QLabel("Freqs (comma):"), 0, 0)
        g.addWidget(self.freqs_edit, 0, 1)
        hint = QLabel("Example: -2e6, -1e6, 1e6, 2e6")
        hint.setStyleSheet("color: #666;")
        g.addWidget(hint, 1, 1)
        self.left_v.addWidget(self.freqs_box)

        # FHSS hopset generator
        self.fhss_box = QGroupBox("FHSS hopset generator")
        g = QGridLayout(self.fhss_box)
        g.setColumnStretch(1, 1)

        self.fhss_span_edit = QLineEdit("10e6")
        self._add_validated(self.fhss_span_edit, "FHSS span")

        self.fhss_mode_cb = QComboBox()
        self.fhss_mode_cb.addItems(["spacing", "num"])
        self.fhss_mode_cb.setCurrentText("spacing")
        self.fhss_mode_cb.currentTextChanged.connect(self._update_visibility)

        self.fhss_spacing_edit = QLineEdit("0.5e6")
        self._add_validated(self.fhss_spacing_edit, "FHSS spacing")

        self.fhss_num_edit = QLineEdit("40")
        self._add_validated(self.fhss_num_edit, "FHSS num")

        self.fhss_include_zero_chk = QCheckBox("Include 0 Hz (DC)")
        self.fhss_include_zero_chk.setChecked(False)

        self.fhss_shuffle_chk = QCheckBox("Shuffle hop order")
        self.fhss_shuffle_chk.setChecked(True)

        self.fhss_seed_edit = QLineEdit("123")
        self._add_validated(self.fhss_seed_edit, "FHSS seed")

        g.addWidget(QLabel("Half-span (+/-):"), 0, 0)
        g.addWidget(self._row_with_units(self.fhss_span_edit, "Hz"), 0, 1)

        g.addWidget(QLabel("Mode:"), 1, 0)
        g.addWidget(self.fhss_mode_cb, 1, 1)

        # Spacing row
        self.fhss_spacing_row = QWidget()
        sr = QHBoxLayout(self.fhss_spacing_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(self._row_with_units(self.fhss_spacing_edit, "Hz"))
        sr.addStretch(1)

        g.addWidget(QLabel("Spacing:"), 2, 0)
        g.addWidget(self.fhss_spacing_row, 2, 1)

        # Num row
        self.fhss_num_row = QWidget()
        nr = QHBoxLayout(self.fhss_num_row)
        nr.setContentsMargins(0, 0, 0, 0)
        nr.addWidget(self.fhss_num_edit)
        nr.addStretch(1)

        g.addWidget(QLabel("Num channels:"), 3, 0)
        g.addWidget(self.fhss_num_row, 3, 1)

        g.addWidget(self.fhss_include_zero_chk, 4, 0, 1, 2)
        g.addWidget(self.fhss_shuffle_chk, 5, 0, 1, 2)

        g.addWidget(QLabel("Seed:"), 6, 0)
        g.addWidget(self.fhss_seed_edit, 6, 1)

        hint = QLabel("Hop frequencies are generated around 0 Hz (baseband).")
        hint.setStyleSheet("color: #666;")
        g.addWidget(hint, 7, 0, 1, 2)

        self.left_v.addWidget(self.fhss_box)

        # Two-tone
        self.twotone_box = QGroupBox("Two-tone")
        g = QGridLayout(self.twotone_box)
        g.setColumnStretch(1, 1)
        self.centerfreq_edit = QLineEdit("0")
        self.spacing_edit = QLineEdit("1e6")
        self._add_validated(self.centerfreq_edit, "center freq")
        self._add_validated(self.spacing_edit, "spacing")
        g.addWidget(QLabel("Center:"), 0, 0)
        g.addWidget(self._row_with_units(self.centerfreq_edit, "Hz"), 0, 1)
        g.addWidget(QLabel("Spacing:"), 1, 0)
        g.addWidget(self._row_with_units(self.spacing_edit, "Hz"), 1, 1)
        self.left_v.addWidget(self.twotone_box)

        # Digital specifics
        self.digital_box = QGroupBox("Digital (ASK/OOK/FSK/GFSK)")
        g = QGridLayout(self.digital_box)
        g.setColumnStretch(1, 1)

        self.a0_edit = QLineEdit("0.2")
        self.a1_edit = QLineEdit("1.0")
        self.duty_edit = QLineEdit("0.5")
        self.f0_edit = QLineEdit("-50e3")
        self.f1_edit = QLineEdit("50e3")
        self.h_edit = QLineEdit("0.5")
        self.bt_edit = QLineEdit("0.5")

        for e, name in [
            (self.a0_edit, "a0"), (self.a1_edit, "a1"),
            (self.duty_edit, "OOK duty"),
            (self.f0_edit, "f0"), (self.f1_edit, "f1"),
            (self.h_edit, "h"), (self.bt_edit, "BT"),
        ]:
            self._add_validated(e, name)

        g.addWidget(QLabel("ASK a0 / a1:"), 0, 0)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.a0_edit)
        h.addWidget(self.a1_edit)
        g.addWidget(row, 0, 1)

        g.addWidget(QLabel("OOK duty (0-1):"), 1, 0)
        g.addWidget(self.duty_edit, 1, 1)

        g.addWidget(QLabel("FSK f0 / f1:"), 2, 0)
        row2 = QWidget()
        h2 = QHBoxLayout(row2)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(self._row_with_units(self.f0_edit, "Hz"))
        h2.addWidget(self._row_with_units(self.f1_edit, "Hz"))
        g.addWidget(row2, 2, 1)

        g.addWidget(QLabel("GFSK h / BT:"), 3, 0)
        row3 = QWidget()
        h3 = QHBoxLayout(row3)
        h3.setContentsMargins(0, 0, 0, 0)
        h3.addWidget(self.h_edit)
        h3.addWidget(self.bt_edit)
        g.addWidget(row3, 3, 1)

        self.left_v.addWidget(self.digital_box)

        # Chirp
        self.chirp_box = QGroupBox("Chirp")
        g = QGridLayout(self.chirp_box)
        g.setColumnStretch(1, 1)
        self.fstart_edit = QLineEdit("-10e6")
        self.fend_edit = QLineEdit("10e6")
        self._add_validated(self.fstart_edit, "f_start")
        self._add_validated(self.fend_edit, "f_end")
        g.addWidget(QLabel("Start:"), 0, 0)
        g.addWidget(self._row_with_units(self.fstart_edit, "Hz"), 0, 1)
        g.addWidget(QLabel("End:"), 1, 0)
        g.addWidget(self._row_with_units(self.fend_edit, "Hz"), 1, 1)
        self.left_v.addWidget(self.chirp_box)

        # FHSS hoprate (still required by wavegen.gen_fhss)
        self.hoprate_box = QGroupBox("FHSS hop rate")
        g = QGridLayout(self.hoprate_box)
        g.setColumnStretch(1, 1)
        self.hoprate_edit = QLineEdit("1e3")
        self._add_validated(self.hoprate_edit, "hop rate")
        g.addWidget(QLabel("Hop rate:"), 0, 0)
        g.addWidget(self._row_with_units(self.hoprate_edit, "Hz"), 0, 1)
        self.left_v.addWidget(self.hoprate_box)

        # Burst / packet
        self.burst_box = QGroupBox("Burst / Packet")
        g = QGridLayout(self.burst_box)
        g.setColumnStretch(1, 1)

        self.burst_period_edit = QLineEdit("0.0")
        self.burst_duty_edit = QLineEdit("0.5")
        self.burst_rise_edit = QLineEdit("0.0")
        self.burst_fall_edit = QLineEdit("0.0")
        self.packet_len_edit = QLineEdit("0.0")
        self.guard_time_edit = QLineEdit("0.0")

        for e, name in [
            (self.burst_period_edit, "burst period"),
            (self.burst_duty_edit, "burst duty"),
            (self.burst_rise_edit, "burst rise"),
            (self.burst_fall_edit, "burst fall"),
            (self.packet_len_edit, "packet len"),
            (self.guard_time_edit, "guard time"),
        ]:
            self._add_validated(e, name)

        g.addWidget(QLabel("Burst period:"), 0, 0)
        g.addWidget(self._row_with_units(self.burst_period_edit, "s"), 0, 1)
        g.addWidget(QLabel("Burst duty:"), 1, 0)
        g.addWidget(self.burst_duty_edit, 1, 1)
        g.addWidget(QLabel("Rise / fall:"), 2, 0)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self._row_with_units(self.burst_rise_edit, "s"))
        h.addWidget(self._row_with_units(self.burst_fall_edit, "s"))
        g.addWidget(row, 2, 1)

        g.addWidget(QLabel("Packet len:"), 3, 0)
        g.addWidget(self._row_with_units(self.packet_len_edit, "s"), 3, 1)
        g.addWidget(QLabel("Guard time:"), 4, 0)
        g.addWidget(self._row_with_units(self.guard_time_edit, "s"), 4, 1)

        hint = QLabel("Packet mode overrides burst mode when packet len > 0.")
        hint.setStyleSheet("color: #666;")
        g.addWidget(hint, 5, 0, 1, 2)

        self.left_v.addWidget(self.burst_box)

    # ---------------- Visibility logic ----------------

    def _update_visibility(self):
        use_sum = self.use_sum_chk.isChecked()
        self.wf2_row.setVisible(use_sum)

        self.sigmf_fields.setVisible(True)

        wf = self.wf1_cb.currentText()

        # Hide all param panels first
        for box in [
            self.analog_box, self.bw_box, self.ofdm_box, self.freqs_box,
            self.fhss_box, self.hoprate_box, self.twotone_box,
            self.digital_box, self.chirp_box, self.pulse_box
        ]:
            box.setVisible(False)

        # Then selectively show
        if wf in {"cw", "am", "fm", "pm"}:
            self.analog_box.setVisible(True)

        if wf in {"noise", "ofdm"}:
            self.bw_box.setVisible(True)

        if wf == "ofdm":
            self.ofdm_box.setVisible(True)

        if wf == "multitone":
            self.freqs_box.setVisible(True)

        if wf == "fhss":
            self.fhss_box.setVisible(True)
            self.hoprate_box.setVisible(True)

            use_spacing = (self.fhss_mode_cb.currentText() == "spacing")
            self.fhss_spacing_row.setVisible(use_spacing)
            self.fhss_num_row.setVisible(not use_spacing)

        if wf == "two_tone":
            self.twotone_box.setVisible(True)

        if wf in {"ask", "ook", "fsk2", "gfsk"}:
            self.digital_box.setVisible(True)

        if wf in {"bpsk", "qpsk", "qam16", "qam64", "ask"}:
            self.pulse_box.setVisible(True)

        if wf == "chirp":
            self.chirp_box.setVisible(True)

        self._validate_all()

    # ---------------- FHSS hopset generation ----------------

    def _build_fhss_freqs(self) -> List[float]:
        span = safe_eval_float(self.fhss_span_edit.text().strip(), None)
        if span is None or span <= 0:
            self._warn("FHSS half-span invalid; using 10e6.")
            span = 10e6

        include_zero = self.fhss_include_zero_chk.isChecked()
        mode = self.fhss_mode_cb.currentText()

        if mode == "spacing":
            spacing = safe_eval_float(
                self.fhss_spacing_edit.text().strip(), None)
            if spacing is None or spacing <= 0:
                self._warn("FHSS spacing invalid; using 0.5e6.")
                spacing = 0.5e6
            freqs = fhss_hopset_gen.gen_hopset_with_spacing(
                span, spacing, include_zero)
        else:
            num = int(safe_eval_float(
                self.fhss_num_edit.text().strip(), None) or 0)
            if num < 1:
                self._warn("FHSS num channels invalid; using 40.")
                num = 40
            freqs = fhss_hopset_gen.gen_hopset_with_num(
                span, num, include_zero)

        if self.fhss_shuffle_chk.isChecked():
            seed_str = self.fhss_seed_edit.text().strip()
            seed = int(seed_str) if seed_str else None
            rnd = random.Random(seed)
            rnd.shuffle(freqs)

        return freqs

    # ---------------- actions ----------------

    def browse_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output file",
            self.outfile_edit.text().strip() or "output.bin",
            "Binary files (*.bin);;All files (*.*)",
        )
        if path:
            self.outfile_edit.setText(path)

    def open_sigmf_meta(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SigMF metadata",
            "",
            "SigMF meta (*.sigmf-meta);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            global_info = meta.get("global", {})
            captures = meta.get("captures", [])
            ann = meta.get("annotations", [])
            ann0 = ann[0] if ann else {}
            params = ann0.get("sdr:params", {})

            txt_lines = []
            txt_lines.append(f"File: {os.path.basename(path)}")
            txt_lines.append("")
            txt_lines.append("[global]")
            for k in ("core:datatype", "core:sample_rate", "core:description",
                      "core:version", "core:author", "core:hw"):
                if k in global_info:
                    txt_lines.append(f"{k} = {global_info[k]}")
            txt_lines.append("")

            if captures:
                c0 = captures[0]
                txt_lines.append("[capture 0]")
                for k in ("core:datetime", "core:frequency", "core:sample_start"):
                    if k in c0:
                        txt_lines.append(f"{k} = {c0[k]}")
                txt_lines.append("")

            txt_lines.append("[annotation 0]")
            if "sdr:waveform" in ann0:
                txt_lines.append(f"sdr:waveform = {ann0['sdr:waveform']}")
            if "core:sample_count" in ann0:
                txt_lines.append(
                    f"core:sample_count = {ann0['core:sample_count']}")
            if "core:comment" in ann0:
                txt_lines.append(f"core:comment = {ann0['core:comment']}")
            txt_lines.append("")
            txt_lines.append("[sdr:params]")
            for k, v in params.items():
                txt_lines.append(f"{k} = {v}")

            self.meta_text.setPlainText("\n".join(txt_lines))
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to read SigMF meta:\n{e}")

    def _parse_multitone_freqs(self) -> List[float]:
        freqs_str = self.freqs_edit.text().strip()
        if not freqs_str:
            return []
        freqs = []
        for token in freqs_str.split(","):
            token = token.strip()
            if not token:
                continue
            val = safe_eval_float(token, None)
            if val is None:
                self._warn(f"Multitone frequency invalid '{token}'; skipping.")
                continue
            freqs.append(float(val))
        return freqs

    def _snapshot_params(self) -> JobParams:
        outfile = self.outfile_edit.text().strip()
        if not outfile:
            raise ValueError("Please specify an output file.")

        self._validate_all()

        wf1 = self.wf1_cb.currentText()
        wf2 = self.wf2_cb.currentText()
        use_sum = self.use_sum_chk.isChecked()

        # Common
        fs = self._parse_float(self.fs_edit, 20e6, "sample rate")
        if fs <= 0:
            self._warn("Sample rate must be > 0; using 20e6.")
            fs = 20e6
        dur = self._parse_float(self.dur_edit, 0.05, "duration")
        if dur <= 0:
            self._warn("Duration must be > 0; using 0.05.")
            dur = 0.05
        out_level = self._parse_float(
            self.level_edit, 0.3, "level", min_value=0.0, max_value=1.0)
        if out_level == 0:
            self._warn("Output level is 0; output will be silent.")
        sym_rate = self._parse_float(self.symrate_edit, 1e6, "symbol rate")
        if sym_rate <= 0:
            self._warn("Symbol rate must be > 0; using 1e6.")
            sym_rate = 1e6
        if sym_rate > fs:
            self._warn("Symbol rate exceeds sample rate; clamping to fs.")
            sym_rate = fs

        # Pulse
        pulse = self.pulse_cb.currentText()
        rolloff = self._parse_float(
            self.rolloff_edit, 0.25, "rolloff", min_value=0.0, max_value=1.0)
        span = self._parse_int(self.span_edit, 6, "span", min_value=1)

        # Analog
        tonefreq = self._parse_float(self.tonefreq_edit, 0.0, "tone freq")
        modfreq = self._parse_float(self.modfreq_edit, 1e3, "mod freq")
        if modfreq <= 0:
            self._warn("Mod freq must be > 0; using 1e3.")
            modfreq = 1e3
        depth = self._parse_float(
            self.depth_edit, 0.5, "AM depth", min_value=0.0, max_value=1.0)
        dev = self._parse_float(
            self.dev_edit, 5e3, "FM deviation", min_value=0.0)
        phase_dev = self._parse_float(
            self.phase_dev_edit, 1.0, "PM phase dev", min_value=0.0)

        # BW
        bw = self._parse_float(self.bw_edit, 10e6, "bandwidth")
        if bw <= 0:
            self._warn("Bandwidth must be > 0; using 10e6.")
            bw = 10e6
        if bw > fs:
            self._warn("Bandwidth exceeds sample rate; clamping to fs.")
            bw = fs

        # Two-tone
        centerfreq = self._parse_float(
            self.centerfreq_edit, 0.0, "center freq")
        spacing = self._parse_float(self.spacing_edit, 1e6, "spacing")
        if spacing <= 0:
            self._warn("Spacing must be > 0; using 1e6.")
            spacing = 1e6

        # Digital specifics
        a0 = self._parse_float(self.a0_edit, 0.2, "a0")
        a1 = self._parse_float(self.a1_edit, 1.0, "a1")
        duty = self._parse_float(
            self.duty_edit, 0.5, "OOK duty", min_value=0.0, max_value=1.0)
        f0 = self._parse_float(self.f0_edit, -50e3, "f0")
        f1 = self._parse_float(self.f1_edit, 50e3, "f1")
        h = self._parse_float(self.h_edit, 0.5, "h")
        bt = self._parse_float(self.bt_edit, 0.5, "BT")

        # Chirp
        fstart = self._parse_float(self.fstart_edit, -10e6, "f_start")
        fend = self._parse_float(self.fend_edit, 10e6, "f_end")

        # FHSS hop rate
        hoprate = self._parse_float(self.hoprate_edit, 1e3, "hop rate")
        if hoprate <= 0:
            self._warn("Hop rate must be > 0; using 1e3.")
            hoprate = 1e3

        # Burst / packet
        burst_period = self._parse_float(
            self.burst_period_edit, 0.0, "burst period")
        burst_duty = self._parse_float(self.burst_duty_edit, 0.5, "burst duty")
        burst_rise = self._parse_float(self.burst_rise_edit, 0.0, "burst rise")
        burst_fall = self._parse_float(self.burst_fall_edit, 0.0, "burst fall")
        packet_len = self._parse_float(self.packet_len_edit, 0.0, "packet len")
        guard_time = self._parse_float(self.guard_time_edit, 0.0, "guard time")

        if burst_duty < 0.0 or burst_duty > 1.0:
            self._warn("Burst duty should be 0..1; clamping.")
            burst_duty = min(max(burst_duty, 0.0), 1.0)
        if packet_len > 0 and burst_period > 0:
            self._warn("Packet mode overrides burst mode when packet len > 0.")

        # OFDM
        fftsize = self._parse_int(
            self.fftsize_edit, 1024, "FFT size", min_value=8)
        cplen = self._parse_int(self.cplen_edit, 128, "CP length", min_value=0)
        if cplen >= fftsize:
            self._warn("CP length >= FFT size; clamping to FFT size - 1.")
            cplen = max(0, fftsize - 1)

        # Weights
        w1_weight = self._parse_float(self.w1_weight_edit, 1.0, "W1 weight")
        w2_weight = self._parse_float(self.w2_weight_edit, 0.5, "W2 weight")

        # Multitone freqs parsed once (shared if multitone is selected in either slot)
        multitone_freqs = self._parse_multitone_freqs()

        # Compute freqs for each waveform slot
        if wf1 == "fhss":
            freqs1 = self._build_fhss_freqs()
        elif wf1 == "multitone":
            freqs1 = multitone_freqs
        else:
            freqs1 = []

        if use_sum and wf2 == "fhss":
            freqs2 = self._build_fhss_freqs()
        elif use_sum and wf2 == "multitone":
            freqs2 = multitone_freqs
        else:
            freqs2 = []

        if wf1 in {"multitone", "fhss"} and not freqs1:
            self._warn("Waveform 1 frequency list empty; using 0 Hz.")
            freqs1 = [0.0]
        if use_sum and wf2 in {"multitone", "fhss"} and not freqs2:
            self._warn("Waveform 2 frequency list empty; using 0 Hz.")
            freqs2 = [0.0]

        # SigMF
        sigmf_enabled = True
        sigmf_desc = self.sigmf_desc_edit.text().strip()
        sigmf_author = self.sigmf_author_edit.text().strip() or "wavegen-gui"
        sigmf_cf_str = self.sigmf_cf_edit.text().strip()
        sigmf_cf = safe_eval_float(
            sigmf_cf_str, None) if sigmf_cf_str else None
        if sigmf_cf_str and sigmf_cf is None:
            self._warn("SigMF center freq invalid; ignoring.")

        return JobParams(
            fs=float(fs), dur=float(dur), out_level=float(out_level), sym_rate=float(sym_rate),
            pulse=pulse, rolloff=float(rolloff), span=int(span),
            tonefreq=float(tonefreq), modfreq=float(modfreq), depth=float(depth), dev=float(dev),
            phase_dev=float(phase_dev), bw=float(bw),
            freqs1=freqs1, freqs2=freqs2,
            centerfreq=float(centerfreq), spacing=float(spacing),
            a0=float(a0), a1=float(a1), duty=float(duty), f0=float(f0), f1=float(f1),
            h=float(h), bt=float(bt),
            fstart=float(fstart), fend=float(fend),
            hoprate=float(hoprate), fftsize=int(fftsize), cplen=int(cplen),
            burst_period=float(burst_period), burst_duty=float(burst_duty),
            burst_rise=float(burst_rise), burst_fall=float(burst_fall),
            packet_len=float(packet_len), guard_time=float(guard_time),
            wf1=wf1, wf2=wf2, w1_weight=float(w1_weight), w2_weight=float(w2_weight),
            use_sum=use_sum,
            outfile=outfile, fmt=self.fmt_cb.currentText(),
            sigmf_enabled=sigmf_enabled, sigmf_desc=sigmf_desc,
            sigmf_author=sigmf_author, sigmf_cf=sigmf_cf,
        )

    def on_generate(self):
        try:
            params = self._snapshot_params()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self._log("Generate requested.")
        self.status_label.setText("Generating...")
        self.generate_btn.setEnabled(False)
        self.progress.setVisible(False)

        self._thread = QThread(self)
        self._worker = GeneratorWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.log.connect(self._log)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_worker_refs)

        self._thread.start()

    def _clear_worker_refs(self):
        self._worker = None
        self._thread = None

    def _on_done(self, msg: str):
        self.status_label.setText(msg)
        self.generate_btn.setEnabled(True)
        self._log(msg)
        if self.auto_load_analyzer_chk.isChecked():
            try:
                self.analyzer_path_edit.setText(self._worker.params.outfile)
                self._load_analyzer_file()
                self.tabs.setCurrentIndex(1)
            except Exception as e:
                self._warn(f"Auto-load failed: {e}")

    def _on_error(self, err: str):
        self.status_label.setText("Error.")
        self.generate_btn.setEnabled(True)
        self._log(f"Error: {err}")
        QMessageBox.critical(self, "Error", err)


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
