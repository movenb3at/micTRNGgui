import sys
import datetime
import hashlib
import numpy as np
from scipy.fft import rfft, rfftfreq

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QTextEdit, QFrame, QSplitter,
    QScrollArea, QProgressBar
)

from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QPoint
from PyQt6.QtGui import QFont, QPainter, QPen, QColor, QPolygon

# Matplotlib integration safely in PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------------------------------------
# [System Settings & Exception Handling]
# ---------------------------------------------------------
try:
    import sounddevice as sd
    devices = sd.query_devices()
    has_input = any(d['max_input_channels'] > 0 for d in devices)
    MIC_AVAILABLE = has_input
except Exception:
    MIC_AVAILABLE = False

# 테마 색상 정의 (Modern Synthwave / Cyber Dark Theme)
COLOR_BG = "#121212"
COLOR_CARD = "#1a1a1a"
COLOR_TEXT = "#ffffff"
COLOR_MUTED = "#888888"
COLOR_ACCENT = "#00f0ff"
COLOR_GREEN = "#39ff14"
COLOR_PURPLE = "#b026ff"
COLOR_RED = "#ff3b30"
COLOR_BORDER = "#2c2c2c"


# ---------------------------------------------------------
# [1. Audio Recording Thread (Real-time Stream)]
# ---------------------------------------------------------
class AudioWorker(QThread):
    data_ready = pyqtSignal(np.ndarray)          # 2초 주기 TRNG 처리용
    realtime_data_ready = pyqtSignal(np.ndarray) # 실시간 파형 렌더링용
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal()
    
    def __init__(self, samplerate=44100, duration=2.0):
        super().__init__()
        self.samplerate = samplerate
        self.duration = duration
        self.is_running = True
        
    def run(self):
        try:
            if not MIC_AVAILABLE:
                self.log_signal.emit("[시스템 에러] 물리 소스(마이크) 소실로 인해 TRNG 작동이 강제 중단되었습니다.")
                self.error_signal.emit()
                return

            self.log_signal.emit("실시간 마이크 입력 스트리밍 및 TRNG 데이터 수집을 시작합니다...")
            
            target_samples = int(self.samplerate * self.duration)
            trng_buffer = []
            chunk_size = 2048  # 약 46ms 단위로 데이터를 쪼개어 실시간성 확보
            
            with sd.InputStream(samplerate=self.samplerate, channels=1, dtype='int16', blocksize=chunk_size) as stream:
                while self.is_running:
                    data, overflow = stream.read(chunk_size)
                    audio_chunk = data[:, 0]
                    
                    # 1. 실시간 비주얼라이저로 즉각 데이터 전송
                    self.realtime_data_ready.emit(audio_chunk.copy())
                    
                    # 2. TRNG (2초 주기) 처리를 위해 버퍼에 누적
                    trng_buffer.extend(audio_chunk)
                    if len(trng_buffer) >= target_samples:
                        trng_data = np.array(trng_buffer[:target_samples])
                        self.data_ready.emit(trng_data)
                        # 분석이 끝난 앞부분 2초 분량은 비우고, 남은 짜투리 데이터는 유지
                        del trng_buffer[:target_samples]
                        
        except Exception as e:
            self.log_signal.emit(f"[에러 발생] 마이크 수집 실패: {str(e)}")
            self.error_signal.emit()
            
    def stop(self):
        self.is_running = False

# ---------------------------------------------------------
# [2-A. Real-time High FPS Waveform Widget]
# QPainter를 사용해 프레임 드랍 없이 초고속으로 실시간 파형을 그리는 위젯
# ---------------------------------------------------------
class RealTimeWaveWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.buffer_size = 4096 * 2  # 화면에 보여줄 최근 샘플 개수
        self.audio_buffer = np.zeros(self.buffer_size)
        
    def update_wave(self, new_data):
        shift = len(new_data)
        if shift > self.buffer_size:
            self.audio_buffer = new_data[-self.buffer_size:]
        else:
            self.audio_buffer[:-shift] = self.audio_buffer[shift:]
            self.audio_buffer[-shift:] = new_data
        self.update()  # 화면 다시 그리기 (paintEvent 호출)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. 배경 및 테두리
        painter.fillRect(self.rect(), QColor(COLOR_CARD))
        painter.setPen(QPen(QColor(COLOR_BORDER), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        
        # 2. 타이틀 텍스트
        painter.setPen(QColor(COLOR_TEXT))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(10, 20, "🔴 LIVE AUDIO STREAM (Real-time)")
        
        # 3. 중앙 기준선
        mid_y = self.height() / 2
        painter.setPen(QPen(QColor(COLOR_MUTED), 1, Qt.PenStyle.DotLine))
        painter.drawLine(0, int(mid_y), self.width(), int(mid_y))
        
        # 4. 실시간 파형 그리기 (성능 최적화를 위해 점 간격(step) 조절)
        painter.setPen(QPen(QColor(COLOR_GREEN), 1))
        width = self.width()
        height = self.height()
        
        max_amp = max(2000.0, float(np.max(np.abs(self.audio_buffer))))
        step = max(1, self.buffer_size // width)
        x_scale = width / (self.buffer_size / step)
        
        points = []
        for i in range(0, self.buffer_size, step):
            val = self.audio_buffer[i]
            x = int((i // step) * x_scale)
            y = int(mid_y - (val / max_amp) * (height / 2.2))
            points.append(QPoint(x, y))
            
        if points:
            painter.drawPolyline(QPolygon(points))


# ---------------------------------------------------------
# [2-B. Matplotlib 2-Second Snapshot Canvas]
# 2초 단위로 TRNG 분석 시점의 파형과 FFT를 보여주는 위젯
# ---------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor=COLOR_BG)
        self.ax_wave = fig.add_subplot(211)
        self.ax_fft = fig.add_subplot(212)
        
        fig.subplots_adjust(hspace=0.45, top=0.90, bottom=0.1, left=0.12, right=0.95)
        super().__init__(fig)
        self.setParent(parent)
        
        for ax in [self.ax_wave, self.ax_fft]:
            ax.set_facecolor(COLOR_CARD)
            ax.spines['bottom'].set_color('#3c3c3c')
            ax.spines['top'].set_color('#3c3c3c')
            ax.spines['left'].set_color('#3c3c3c')
            ax.spines['right'].set_color('#3c3c3c')
            ax.tick_params(colors=COLOR_MUTED, labelsize=8)
            ax.yaxis.label.set_color(COLOR_MUTED)
            ax.xaxis.label.set_color(COLOR_MUTED)
            ax.title.set_color(COLOR_TEXT)
            ax.grid(True, color='#2c2c2c', linestyle=':')
            
        self.ax_wave.set_title("TRNG Snapshot Waveform (Every 2s)", fontsize=10, fontweight='bold', pad=8)
        self.ax_wave.set_xlabel("Sample Index", fontsize=8)
        self.ax_wave.set_ylabel("Amplitude", fontsize=8)
        self.line_wave, = self.ax_wave.plot([], [], color=COLOR_ACCENT, linewidth=0.8)
        
        self.ax_fft.set_title("Frequency Spectrum (FFT)", fontsize=10, fontweight='bold', pad=8)
        self.ax_fft.set_xlabel("Frequency (Hz)", fontsize=8)
        self.ax_fft.set_ylabel("Magnitude (dB)", fontsize=8)
        self.line_fft, = self.ax_fft.plot([], [], color=COLOR_PURPLE, linewidth=0.8)

    def update_plots(self, data, samplerate=44100):
        x_wave = np.arange(len(data))
        self.line_wave.set_data(x_wave, data)
        self.ax_wave.set_xlim(0, len(data))
        peak = max(float(np.max(np.abs(data))), 100.0)
        self.ax_wave.set_ylim(-peak * 1.1, peak * 1.1)
        
        n = len(data)
        fft_mags = np.abs(rfft(data))
        fft_mags_db = 20 * np.log10(fft_mags + 1e-5)
        freqs = rfftfreq(n, d=1/samplerate)
        
        self.line_fft.set_data(freqs, fft_mags_db)
        self.ax_fft.set_xlim(0, samplerate / 2)
        self.ax_fft.set_ylim(np.min(fft_mags_db) - 5, np.max(fft_mags_db) + 10)
        
        self.draw()


# ---------------------------------------------------------
# [3. Flowchart Node Widget]
# ---------------------------------------------------------
class FlowStepWidget(QFrame):
    def __init__(self, step_num, title, parent=None):
        super().__init__(parent)
        self.step_num = step_num
        self.title = title
        self.init_ui()
        
    def init_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(110)
        self.setFixedHeight(65)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(2)
        
        self.num_label = QLabel(f"STEP {self.step_num}")
        self.num_label.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        self.num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        
        layout.addWidget(self.num_label)
        layout.addWidget(self.title_label)
        self.setLayout(layout)
        self.set_state("idle")
        
    def set_state(self, state):
        if state == "idle":
            self.setStyleSheet(f"FlowStepWidget {{ background-color: #151515; border: 2px solid {COLOR_BORDER}; border-radius: 8px; }}")
            self.num_label.setStyleSheet("color: #555555;")
            self.title_label.setStyleSheet("color: #777777;")
        elif state == "active":
            self.setStyleSheet(f"FlowStepWidget {{ background-color: #0c2d33; border: 2px solid {COLOR_ACCENT}; border-radius: 8px; }}")
            self.num_label.setStyleSheet(f"color: {COLOR_ACCENT};")
            self.title_label.setStyleSheet("color: #ffffff;")
        elif state == "completed":
            self.setStyleSheet(f"FlowStepWidget {{ background-color: #112d1b; border: 2px solid {COLOR_GREEN}; border-radius: 8px; }}")
            self.num_label.setStyleSheet(f"color: {COLOR_GREEN};")
            self.title_label.setStyleSheet("color: #e0e0e0;")


# ---------------------------------------------------------
# [4. Main TRNG Application Window]
# ---------------------------------------------------------
class TRNGVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.samplerate = 44100
        self.duration = 2.0
        self.latest_random_data = None
        
        self.flow_step = 0
        self.flow_timer = QTimer()
        self.flow_timer.timeout.connect(self.on_flow_sequence_tick)
        
        self.init_window_properties()
        self.init_ui_layout()
        self.apply_dark_theme()
        
        self.append_log("TRNG 시각화 시뮬레이터 준비 완료. 'START' 버튼을 눌러 측정을 시작하십시오.")
        
    def init_window_properties(self):
        self.setWindowTitle("True Random Number Generator (TRNG) Real-time Analyzer")
        self.resize(1380, 880)
        self.setMinimumSize(1200, 800)
        
    def init_ui_layout(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # --- TOP PANEL ---
        top_panel = QHBoxLayout()
        title_tag = QLabel("⚡ HARDWARE TRNG VISUALIZER")
        title_tag.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_tag.setStyleSheet(f"color: {COLOR_ACCENT}; letter-spacing: 1px;")
        top_panel.addWidget(title_tag)
        top_panel.addStretch()
        
        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self.on_start_clicked)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        
        self.btn_save = QPushButton("SAVE RANDOM NUMBER")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.on_save_clicked)
        
        self.btn_clear_log = QPushButton("CLEAR LOG")
        self.btn_clear_log.setObjectName("btn_clear")
        self.btn_clear_log.clicked.connect(self.on_clear_log_clicked)
        
        top_panel.addWidget(self.btn_start)
        top_panel.addWidget(self.btn_stop)
        top_panel.addWidget(self.btn_save)
        top_panel.addWidget(self.btn_clear_log)
        main_layout.addLayout(top_panel)
        
        # --- MIDDLE PANEL ---
        mid_splitter = QSplitter(Qt.Orientation.Horizontal)
        mid_splitter.setHandleWidth(4)
        
        # 1) 좌측: 그래프 영역 (실시간 파형 + 2초 스냅샷)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        self.live_wave = RealTimeWaveWidget(self)
        self.canvas = MplCanvas(self, width=5, height=6, dpi=100)
        
        left_layout.addWidget(self.live_wave)
        left_layout.addWidget(self.canvas)
        mid_splitter.addWidget(left_panel)
        
        # 2) 우측: 실시간 정보 영역
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        right_container = QWidget()
        right_container.setObjectName("RightContainer")
        right_container.setStyleSheet(f"QWidget#RightContainer {{ background-color: {COLOR_BG}; }}")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(5, 0, 5, 0)
        right_layout.setSpacing(12)
        
        # A. Flowchart
        flow_card = QFrame()
        flow_card_layout = QVBoxLayout(flow_card)
        flow_card_layout.setContentsMargins(12, 10, 12, 10)
        flow_card_layout.addWidget(self.create_card_title("1. TRNG Real-time State Pipeline"))
        
        flow_row = QHBoxLayout()
        flow_row.setSpacing(3)
        self.flow_widgets = []
        steps = [
            ("Analog Microphone", 1),
            ("Digitized Samples", 2),
            ("LSB Harvesting", 3),
            ("SHA-256 Extraction", 4),
            ("True Random Integer", 5)
        ]
        for i, (title, num) in enumerate(steps):
            widget = FlowStepWidget(num, title)
            self.flow_widgets.append(widget)
            flow_row.addWidget(widget)
            if i < len(steps) - 1:
                arrow = QLabel("➔")
                arrow.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
                arrow.setStyleSheet("color: #444444;")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                flow_row.addWidget(arrow)
        flow_card_layout.addLayout(flow_row)
        right_layout.addWidget(flow_card)
        
        # B. 신호 통계 및 엔트로피 요약
        stats_entropy_row = QHBoxLayout()
        stats_entropy_row.setSpacing(12)
        
        # B-1. 신호 통계
        stats_card = QFrame()
        stats_card_layout = QVBoxLayout(stats_card)
        stats_card_layout.addWidget(self.create_card_title("2. Raw Noise Signal Statistics"))
        stats_grid = QGridLayout()
        stats_grid.setSpacing(8)
        
        self.lbl_max = QLabel("Max Amplitude : -")
        self.lbl_min = QLabel("Min Amplitude : -")
        self.lbl_mean = QLabel("Mean : -")
        self.lbl_std = QLabel("Std Dev : -")
        self.lbl_rms = QLabel("RMS : -")
        self.lbl_samples = QLabel("Samples : -")
        self.lbl_rate = QLabel("Sample Rate : -")
        
        for idx, lbl in enumerate([self.lbl_max, self.lbl_min, self.lbl_mean, self.lbl_std, self.lbl_rms, self.lbl_samples, self.lbl_rate]):
            lbl.setFont(QFont("Consolas", 9))
            lbl.setStyleSheet("color: #e0e0e0;")
        
        stats_grid.addWidget(self.lbl_max, 0, 0)
        stats_grid.addWidget(self.lbl_min, 0, 1)
        stats_grid.addWidget(self.lbl_mean, 1, 0)
        stats_grid.addWidget(self.lbl_std, 1, 1)
        stats_grid.addWidget(self.lbl_rms, 2, 0)
        stats_grid.addWidget(self.lbl_samples, 2, 1)
        stats_grid.addWidget(self.lbl_rate, 3, 0, 1, 2)
        stats_card_layout.addLayout(stats_grid)
        stats_entropy_row.addWidget(stats_card)
        
        # B-2. 엔트로피 분석
        entropy_card = QFrame()
        entropy_card_layout = QVBoxLayout(entropy_card)
        entropy_card_layout.addWidget(self.create_card_title("3. Entropy & Signal Quality"))
        
        ent_layout = QHBoxLayout()
        ent_text_layout = QVBoxLayout()
        
        lbl_ent_title = QLabel("ESTIMATED ENTROPY")
        lbl_ent_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_ent_title.setStyleSheet(f"color: {COLOR_MUTED};")
        self.lbl_entropy = QLabel("- bits")
        self.lbl_entropy.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.lbl_entropy.setStyleSheet(f"color: {COLOR_GREEN};")
        
        lbl_qual_title = QLabel("NOISE QUALITY (Max 16-bit)")
        lbl_qual_title.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_qual_title.setStyleSheet(f"color: {COLOR_MUTED};")
        self.lbl_quality = QLabel("-%")
        self.lbl_quality.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.lbl_quality.setStyleSheet(f"color: {COLOR_ACCENT};")
        
        ent_text_layout.addWidget(lbl_ent_title)
        ent_text_layout.addWidget(self.lbl_entropy)
        ent_text_layout.addSpacing(4)
        ent_text_layout.addWidget(lbl_qual_title)
        ent_text_layout.addWidget(self.lbl_quality)
        ent_layout.addLayout(ent_text_layout)
        
        self.bar_quality = QProgressBar()
        self.bar_quality.setOrientation(Qt.Orientation.Vertical)
        self.bar_quality.setRange(0, 100)
        self.bar_quality.setValue(0)
        self.bar_quality.setFixedWidth(16)
        self.bar_quality.setTextVisible(False)
        self.bar_quality.setStyleSheet(f"""
            QProgressBar {{ background-color: #111111; border: 1px solid #333333; border-radius: 3px; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:1, x2:0, y2:0, stop:0 {COLOR_PURPLE}, stop:1 {COLOR_ACCENT}); border-radius: 2px; }}
        """)
        ent_layout.addWidget(self.bar_quality)
        entropy_card_layout.addLayout(ent_layout)
        stats_entropy_row.addWidget(entropy_card)
        
        right_layout.addLayout(stats_entropy_row)
        
        # C. LSB 시각화
        lsb_card = QFrame()
        lsb_layout = QVBoxLayout(lsb_card)
        lsb_layout.setSpacing(6)
        lsb_layout.addWidget(self.create_card_title("4. Least Significant Bit (LSB) Extraction"))
        
        lbl_desc = QLabel("아날로그 노이즈 데이터의 하위 비트(LSB)는 미세 기하학적 흔들림(열적 카오스)을 포함하여 물리적 무작위성이 가장 강력한 영역입니다.")
        lbl_desc.setFont(QFont("Segoe UI", 7))
        lbl_desc.setStyleSheet(f"color: {COLOR_MUTED}; line-height: 1.2;")
        lbl_desc.setWordWrap(True)
        lsb_layout.addWidget(lbl_desc)
        
        self.lbl_lsb_table = QLabel()
        self.lbl_lsb_table.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_lsb_table.setText("<p style='color:#666666; font-family:Consolas; font-size:10px;'>수집 데이터를 대기 중입니다...</p>")
        lsb_layout.addWidget(self.lbl_lsb_table)
        
        lbl_raw_stream = QLabel("EXTRACTED RAW LSB BITSTREAM (FIRST 48 SAMPLES)")
        lbl_raw_stream.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_raw_stream.setStyleSheet(f"color: {COLOR_MUTED};")
        lsb_layout.addWidget(lbl_raw_stream)
        
        self.lbl_lsb_stream = QLabel("-")
        self.lbl_lsb_stream.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        self.lbl_lsb_stream.setStyleSheet(f"color: {COLOR_RED}; letter-spacing: 1px;")
        self.lbl_lsb_stream.setWordWrap(True)
        lsb_layout.addWidget(self.lbl_lsb_stream)
        
        right_layout.addWidget(lsb_card)
        
        # D. SHA-256 후처리
        post_card = QFrame()
        post_layout = QVBoxLayout(post_card)
        post_layout.setSpacing(6)
        post_layout.addWidget(self.create_card_title("5. Cryptographic Post-Processing & Output"))
        
        lbl_sha_lbl = QLabel("SHA-256 ENCRYPTION HASH DIGEST (DEBIASTING & COMPRESSION)")
        lbl_sha_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_sha_lbl.setStyleSheet(f"color: {COLOR_MUTED};")
        post_layout.addWidget(lbl_sha_lbl)
        
        self.lbl_sha256 = QLabel("-")
        self.lbl_sha256.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        self.lbl_sha256.setStyleSheet(f"color: {COLOR_PURPLE};")
        self.lbl_sha256.setWordWrap(True)
        post_layout.addWidget(self.lbl_sha256)
        
        lbl_rand_lbl = QLabel("FINAL GENERATED 64-BIT TRUE RANDOM INTEGER")
        lbl_rand_lbl.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_rand_lbl.setStyleSheet(f"color: {COLOR_MUTED};")
        post_layout.addWidget(lbl_rand_lbl)
        
        self.lbl_rand_num = QLabel("-")
        self.lbl_rand_num.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        self.lbl_rand_num.setStyleSheet(f"color: {COLOR_GREEN}; letter-spacing: 1px;")
        self.lbl_rand_num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        post_layout.addWidget(self.lbl_rand_num)
        
        right_layout.addWidget(post_card)
        right_scroll.setWidget(right_container)
        mid_splitter.addWidget(right_scroll)
        
        mid_splitter.setSizes([450, 650])
        main_layout.addWidget(mid_splitter)
        
        # --- BOTTOM PANEL ---
        log_panel = QVBoxLayout()
        log_panel.setSpacing(4)
        
        lbl_log_title = QLabel("PROCESS EXECUTION LOG")
        lbl_log_title.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_log_title.setStyleSheet(f"color: {COLOR_MUTED};")
        log_panel.addWidget(lbl_log_title)
        
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(120)
        log_panel.addWidget(self.txt_log)
        
        main_layout.addLayout(log_panel)
        
    def create_card_title(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl.setStyleSheet(f"color: {COLOR_ACCENT}; text-transform: uppercase; letter-spacing: 0.5px; padding-bottom: 3px;")
        return lbl

    # ---------------------------------------------------------
    # [5. Thread Controller & Interactions]
    # ---------------------------------------------------------
    def on_start_clicked(self):
        if not MIC_AVAILABLE:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "하드웨어 에러", "물리적 마이크 장치가 감지되지 않아 시작할 수 없습니다.")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(True)
        
        self.worker = AudioWorker(samplerate=self.samplerate, duration=self.duration)
        
        # 실시간 데이터 연결
        self.worker.realtime_data_ready.connect(self.live_wave.update_wave)
        # 2초 주기 TRNG 데이터 연결
        self.worker.data_ready.connect(self.on_audio_data_received)
        
        self.worker.log_signal.connect(self.append_log)
        self.worker.error_signal.connect(self.on_stop_clicked)
        self.worker.start()
        
    def on_stop_clicked(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.append_log("수집 프로세스를 중지했습니다.")
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.reset_flow_pipeline()
        
    def on_clear_log_clicked(self):
        self.txt_log.clear()
        
    def on_save_clicked(self):
        if self.latest_random_data is None:
            self.append_log("[경고] 추출된 난수 데이터가 없습니다.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"TRNG_Output_{timestamp}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(f"Generated Timestamp : {self.latest_random_data['timestamp']}\n")
                file.write(f"Estimated Entropy   : {self.latest_random_data['entropy']:.6f} bits/sample\n")
                file.write(f"Noise Quality       : {self.latest_random_data['quality']:.2f}%\n")
                file.write(f"SHA-256 Hash Digest : {self.latest_random_data['sha256']}\n")
                file.write(f"64-bit Random Uint  : {self.latest_random_data['rand_num']}\n")
            self.append_log(f"파일 저장 완료: '{filename}'")
        except Exception as e:
            self.append_log(f"파일 저장 실패: {str(e)}")

    def append_log(self, text):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"<span style='color:#666666;'>[{now}]</span> <span style='color:#f0f0f0;'>{text}</span>")
        self.txt_log.ensureCursorVisible()

    # ---------------------------------------------------------
    # [6. TRNG Computations & Processing]
    # ---------------------------------------------------------
    def on_audio_data_received(self, data):
        if self.flow_timer.isActive():
            self.flow_timer.stop()
            self.finalize_instant_update()
            
        self.raw_data = data
        self.calculated_results = self.perform_trng_computations(data)
        self.flow_step = 1
        self.flow_timer.start(250)
        
    def perform_trng_computations(self, data):
        results = {}
        results['max'] = int(np.max(data))
        results['min'] = int(np.min(data))
        results['mean'] = float(np.mean(data))
        results['std'] = float(np.std(data))
        results['rms'] = float(np.sqrt(np.mean(data.astype(np.float64)**2)))
        results['samples'] = len(data)
        results['rate'] = self.samplerate
        
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        results['entropy'] = float(entropy)
        results['quality'] = min(100.0, (entropy / 14.0) * 100.0)
        
        results['lsb_table_html'] = self.generate_lsb_html(data[:8])
        results['lsb_bitstream'] = "".join([str(np.uint16(val) & 1) for val in data[:48]])
        
        hex_digest = hashlib.sha256(data.tobytes()).hexdigest()
        results['sha256'] = hex_digest
        results['rand_num'] = int(hex_digest[:16], 16)
        return results

    def generate_lsb_html(self, samples):
        html = "<table style='width:100%; border-collapse: collapse; font-family:Consolas, monospace; font-size:11px;'>"
        html += "<tr style='color:#888888; border-bottom: 1px solid #2a2a2a;'><th align='left'>Sample Node</th><th align='right'>Dec Value</th><th align='center'>16-bit Binary (LSB in red)</th><th align='center'>LSB</th></tr>"
        for idx, val in enumerate(samples):
            bin_str = format(np.uint16(val), '016b')
            lsb = bin_str[-1]
            styled_bin = f"<span style='color:#555555;'>{bin_str[:-1]}</span><span style='color:{COLOR_RED}; font-weight:bold;'>{lsb}</span>"
            val_color = COLOR_GREEN if val >= 0 else COLOR_RED
            styled_val = f"<span style='color:{val_color};'>{val:6d}</span>"
            
            html += f"<tr style='border-bottom: 1px solid #1f1f1f;'>"
            html += f"<td style='color:#666666;'>Audio[{idx}]</td>"
            html += f"<td align='right'>{styled_val}</td>"
            html += f"<td align='center'>{styled_bin}</td>"
            html += f"<td align='center' style='color:{COLOR_RED}; font-weight:bold;'>{lsb}</td></tr>"
        html += "</table>"
        return html

    # ---------------------------------------------------------
    # [7. Real-time Pipeline Sequencer]
    # ---------------------------------------------------------
    def on_flow_sequence_tick(self):
        for i in range(1, 6):
            widget = self.flow_widgets[i-1]
            if i < self.flow_step: widget.set_state("completed")
            elif i == self.flow_step: widget.set_state("active")
            else: widget.set_state("idle")
                
        r = self.calculated_results
        
        if self.flow_step == 1:
            self.canvas.update_plots(self.raw_data, self.samplerate)
        elif self.flow_step == 2:
            self.lbl_max.setText(f"Max Amplitude : {r['max']}")
            self.lbl_min.setText(f"Min Amplitude : {r['min']}")
            self.lbl_mean.setText(f"Mean : {r['mean']:.2f}")
            self.lbl_std.setText(f"Std Dev : {r['std']:.2f}")
            self.lbl_rms.setText(f"RMS : {r['rms']:.2f}")
            self.lbl_samples.setText(f"Samples : {r['samples']}")
            self.lbl_rate.setText(f"Sample Rate : {r['rate']} Hz")
            self.lbl_entropy.setText(f"{r['entropy']:.4f} bits")
            self.lbl_quality.setText(f"{r['quality']:.1f}%")
            self.bar_quality.setValue(int(r['quality']))
        elif self.flow_step == 3:
            self.lbl_lsb_table.setText(r['lsb_table_html'])
            raw_bits = r['lsb_bitstream']
            self.lbl_lsb_stream.setText(" ".join([raw_bits[i:i+8] for i in range(0, len(raw_bits), 8)]))
        elif self.flow_step == 4:
            full_hash = r['sha256']
            self.lbl_sha256.setText(" ".join([full_hash[i:i+8] for i in range(0, len(full_hash), 8)]))
        elif self.flow_step == 5:
            self.lbl_rand_num.setText(f"{r['rand_num']}")
            self.flow_timer.stop()
            self.flow_widgets[4].set_state("completed")
            self.latest_random_data = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'entropy': r['entropy'],
                'quality': r['quality'],
                'sha256': r['sha256'],
                'rand_num': r['rand_num']
            }
        self.flow_step += 1
        
    def finalize_instant_update(self):
        if not hasattr(self, 'calculated_results') or self.calculated_results is None: return
        r = self.calculated_results
        self.canvas.update_plots(self.raw_data, self.samplerate)
        self.lbl_max.setText(f"Max Amplitude : {r['max']}")
        self.lbl_min.setText(f"Min Amplitude : {r['min']}")
        self.lbl_mean.setText(f"Mean : {r['mean']:.2f}")
        self.lbl_std.setText(f"Std Dev : {r['std']:.2f}")
        self.lbl_rms.setText(f"RMS : {r['rms']:.2f}")
        self.lbl_samples.setText(f"Samples : {r['samples']}")
        self.lbl_rate.setText(f"Sample Rate : {r['rate']} Hz")
        self.lbl_entropy.setText(f"{r['entropy']:.4f} bits")
        self.lbl_quality.setText(f"{r['quality']:.1f}%")
        self.bar_quality.setValue(int(r['quality']))
        self.lbl_lsb_table.setText(r['lsb_table_html'])
        raw_bits = r['lsb_bitstream']
        self.lbl_lsb_stream.setText(" ".join([raw_bits[i:i+8] for i in range(0, len(raw_bits), 8)]))
        full_hash = r['sha256']
        self.lbl_sha256.setText(" ".join([full_hash[i:i+8] for i in range(0, len(full_hash), 8)]))
        self.lbl_rand_num.setText(f"{r['rand_num']}")
        for w in self.flow_widgets: w.set_state("completed")

    def reset_flow_pipeline(self):
        self.flow_timer.stop()
        for w in self.flow_widgets: w.set_state("idle")

    # ---------------------------------------------------------
    # [8. Custom Dark CSS Stylesheet]
    # ---------------------------------------------------------
    def apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {COLOR_BG}; }}
            QFrame {{ background-color: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; }}
            QLabel {{ color: {COLOR_TEXT}; }}
            QTextEdit {{ background-color: #0b0b0b; border: 1px solid {COLOR_BORDER}; border-radius: 6px; color: {COLOR_GREEN}; font-family: 'Consolas', monospace; font-size: 11px; padding: 5px; }}
            QPushButton {{ background-color: #252525; border: 1px solid #3d3d3d; border-radius: 5px; color: #ffffff; padding: 6px 14px; font-weight: bold; font-size: 11px; min-width: 80px; }}
            QPushButton:hover {{ background-color: #353535; border-color: #555555; }}
            QPushButton:pressed {{ background-color: #1a1a1a; }}
            QPushButton:disabled {{ background-color: #151515; border-color: #222222; color: #555555; }}
            QPushButton#btn_start {{ background-color: #0d2613; border: 1px solid {COLOR_GREEN}; color: {COLOR_GREEN}; }}
            QPushButton#btn_start:hover {{ background-color: #153c1e; }}
            QPushButton#btn_stop {{ background-color: #2c0f0e; border: 1px solid {COLOR_RED}; color: {COLOR_RED}; }}
            QPushButton#btn_stop:hover {{ background-color: #421817; }}
            QPushButton#btn_save {{ background-color: #0d2a30; border: 1px solid {COLOR_ACCENT}; color: {COLOR_ACCENT}; }}
            QPushButton#btn_save:hover {{ background-color: #143e47; }}
            
            QScrollBar:vertical {{ background: #111111; width: 8px; margin: 0px 0px 0px 0px; }}
            QScrollBar::handle:vertical {{ background: #333333; min-height: 20px; border-radius: 4px; }}
            QScrollBar::handle:vertical:hover {{ background: {COLOR_ACCENT}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

    def closeEvent(self, event):
        self.on_stop_clicked()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    visualizer = TRNGVisualizer()
    visualizer.show()
    sys.exit(app.exec())