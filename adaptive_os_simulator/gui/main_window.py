"""
Main window for the OS Load Simulator GUI.
"""

import sys
from typing import List, Optional, Dict
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QTabWidget,
    QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRect, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtCharts import QChart, QChartView, QLineSeries

from ..backend.core import PCB, ProcessState
from ..backend.schedulers import FCFSScheduler, SJFScheduler, SRTFScheduler, RoundRobinScheduler, PriorityScheduler
from ..backend.adaptive_scheduler import AdaptiveScheduler

# Process creation dialog
class ProcessDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Process")
        layout = QVBoxLayout(self)

        # PID
        pid_layout = QHBoxLayout()
        pid_label = QLabel("PID:")
        self.pid = QSpinBox()
        self.pid.setRange(1, 9999)
        pid_layout.addWidget(pid_label)
        pid_layout.addWidget(self.pid)
        layout.addLayout(pid_layout)

        # Burst Time
        burst_layout = QHBoxLayout()
        burst_label = QLabel("Burst Time:")
        self.burst_time = QDoubleSpinBox()
        self.burst_time.setRange(0.1, 100.0)
        self.burst_time.setValue(1.0)
        burst_layout.addWidget(burst_label)
        burst_layout.addWidget(self.burst_time)
        layout.addLayout(burst_layout)

        # Arrival Time
        arrival_layout = QHBoxLayout()
        arrival_label = QLabel("Arrival Time:")
        self.arrival_time = QDoubleSpinBox()
        self.arrival_time.setRange(0.0, 100.0)
        self.arrival_time.setValue(0.0)
        arrival_layout.addWidget(arrival_label)
        arrival_layout.addWidget(self.arrival_time)
        layout.addLayout(arrival_layout)

        # Priority
        priority_layout = QHBoxLayout()
        priority_label = QLabel("Priority:")
        self.priority = QSpinBox()
        self.priority.setRange(1, 10)
        self.priority.setValue(5)
        priority_layout.addWidget(priority_label)
        priority_layout.addWidget(self.priority)
        layout.addLayout(priority_layout)

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

# Process table with all columns
class ProcessTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up columns
        self.setColumnCount(10)
        self.setHorizontalHeaderLabels([
            "PID", "Arrival", "Burst", "Remaining",
            "Priority", "State", "Waiting", "Turnaround",
            "Response", "Ctx Switches"
        ])
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                background: #fdfdfd;
                alternate-background-color: #f5f9ff;
                border: 1px solid #dbe3f0;
            }
            QHeaderView::section {
                background: #eef3fb;
                font-weight: 600;
                padding: 6px;
                border: none;
            }
        """)

    def _set_item(self, row: int, column: int, text: str) -> None:
        item = self.item(row, column)
        if item is None:
            self.setItem(row, column, QTableWidgetItem(text))
        else:
            item.setText(text)

    def update_process(self, pcb: PCB) -> None:
        """Update or add a process row."""
        # Look for existing row
        found = False
        found_row = 0
        for row in range(self.rowCount()):
            if self.item(row, 0) and int(self.item(row, 0).text()) == pcb.pid:
                found = True
                found_row = row
                break

        if not found:
            # Add new row
            row = self.rowCount()
            self.insertRow(row)
            self._set_item(row, 0, str(pcb.pid))
            self._set_item(row, 1, f"{pcb.arrival_time:.2f}")
            self._set_item(row, 2, f"{pcb.burst_time:.2f}")
            self._set_item(row, 3, f"{pcb.remaining_time:.2f}")
            self._set_item(row, 4, str(pcb.priority))
        else:
            row = found_row
            # Update static fields that might change
            self._set_item(row, 3, f"{max(0.0, pcb.remaining_time):.2f}")
            self._set_item(row, 4, str(pcb.priority))

        # Always update status and metrics
        self._set_item(row, 5, pcb.state.name)
        self._set_item(row, 3, f"{max(0.0, pcb.remaining_time):.2f}")
        self._set_item(row, 6, f"{pcb.stats.waiting_time:.2f}")
        self._set_item(row, 7, f"{pcb.stats.turnaround_time:.2f}")
        response = pcb.stats.response_time if pcb.stats.response_time is not None else 0.0
        self._set_item(row, 8, f"{response:.2f}")
        self._set_item(row, 9, str(pcb.stats.context_switches))


class ContextSwitchTable(QTableWidget):
    """Tabular view of context switch events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Time", "From PID", "To PID", "Reason", "Scheduler"])
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def add_event(self, time_stamp: float, from_pid: Optional[int], to_pid: Optional[int], reason: str, scheduler_name: str) -> None:
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(f"{time_stamp:.2f}"))
        self.setItem(row, 1, QTableWidgetItem("-" if from_pid is None else str(from_pid)))
        self.setItem(row, 2, QTableWidgetItem("-" if to_pid is None else str(to_pid)))
        self.setItem(row, 3, QTableWidgetItem(reason))
        self.setItem(row, 4, QTableWidgetItem(scheduler_name))
        self.scrollToBottom()


class StatCard(QFrame):
    """Compact metric card component."""

    def __init__(self, title: str, formatter=None, parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        layout = QVBoxLayout(self)
        # Make cards more compact so the metrics panel fits smaller screens
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self.title_label = QLabel(title.upper())
        self.title_label.setStyleSheet("color:#7486a8;font-size:12px;font-weight:600;letter-spacing:1px;")
        layout.addWidget(self.title_label)

        self.value_label = QLabel("0.00")
        # Slightly smaller font to improve fit in the layout
        self.value_label.setStyleSheet("color:#0b2447;font-size:18px;font-weight:700;")
        layout.addWidget(self.value_label)

        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color:#4b5d7d;font-size:12px;")
        layout.addWidget(self.detail_label)

        self.formatter = formatter or (lambda v: f"{v:.2f}")

    def update_value(self, value: float, detail: str = "") -> None:
        self.value_label.setText(self.formatter(value))
        self.detail_label.setText(detail)
# Gantt chart visualization
class GanttChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.exec_log = []
        self.processes = []
        self.current_time = 0.0
        self.colors = [QColor('#0077cc'), QColor('#43a047'), QColor('#fbc02d'), 
                      QColor('#e53935'), QColor('#8e24aa')]
        self.row_height = 24
        self.setMinimumHeight(200)

    def set_exec_log(self, exec_log):
        self.exec_log = exec_log
        self.update()

    def update_time(self, time):
        self.current_time = time
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dynamic scaling
        left_margin = 80
        right_margin = 20
        bottom_margin = 30
        usable_width = max(100, self.width() - left_margin - right_margin)
        segments = sorted(self.exec_log, key=lambda x: x.get('start', 0.0))
        max_segment_end = max(
            [seg.get('end', self.current_time) or self.current_time for seg in segments] + [self.current_time, 1.0]
        )
        max_time = max(1.0, max_segment_end)
        time_scale = usable_width / max_time

        # Assign rows to each PID preserving order of appearance
        process_rows = {}
        for seg in segments:
            pid = seg['pid']
            if pid not in process_rows:
                process_rows[pid] = len(process_rows)

        row_height = self.row_height + 8  # spacing
        required_height = 30 + len(process_rows) * row_height + bottom_margin
        if required_height > self.height():
            self.setMinimumHeight(required_height)

        base_y = self.height() - bottom_margin

        # Draw timeline
        painter.setPen(QPen(QColor('#444444')))
        painter.drawLine(left_margin, base_y, self.width() - right_margin, base_y)

        # Draw time markers
        painter.setPen(QPen(QColor('#666666')))
        marker_interval = max(1, round(max_time / 10))  # At most 10 markers
        for t in range(0, int(max_time) + 1, marker_interval):
            x = int(left_margin + t * time_scale)
            if x < self.width() - right_margin:
                painter.drawLine(x, base_y - 5, x, base_y + 5)
                painter.drawText(x - 10, base_y + 20, str(t))

        # Draw process rows and execution blocks
        for seg in segments:
            pid = seg['pid']
            start = seg.get('start', 0.0)
            end = seg.get('end', self.current_time)
            if end is None:
                end = self.current_time
            row_y = 10 + process_rows.get(pid, 0) * row_height

            # Draw process label at left margin
            text_rect = QRect(5, row_y, left_margin - 10, self.row_height)
            painter.setPen(QPen(QColor('#333333')))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"P{pid}")

            # Create rectangle for this execution segment
            rect = QRect(int(left_margin + start * time_scale), row_y, max(int((end - start) * time_scale), 1), self.row_height)

            # Draw process block with specific color
            color_idx = (pid - 1) % len(self.colors)
            color = self.colors[color_idx]
            if seg.get('scheduler'):
                tint = (hash(seg['scheduler']) % 30) - 15
                color = QColor(
                    min(255, max(0, color.red() + tint)),
                    min(255, max(0, color.green() + tint)),
                    min(255, max(0, color.blue() + tint))
                )

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor('#333333')))
            painter.drawRoundedRect(rect, 4, 4)

            # Draw segment label
            painter.setPen(QPen(QColor('#ffffff')))
            label = f"P{pid}"
            if seg.get('scheduler'):
                label += f" ({seg['scheduler'][:4]})"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
            # continue to next pid (we draw one rect per pid above)

class MetricsPanel(QWidget):
    """Panel showing real-time metrics with charts and KPI cards."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("metricsPanel")
        self.previous_values: Dict[str, float] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(18)

        title = QLabel("Real-time Analytics")
        title.setStyleSheet("font-size:20px;font-weight:700;color:#0b2447;")
        layout.addWidget(title)

        # KPI cards
        card_grid = QGridLayout()
        card_grid.setHorizontalSpacing(12)
        card_grid.setVerticalSpacing(12)

        self.cards = {
            "waiting_time": StatCard("Avg Waiting", lambda v: f"{v:.2f} s"),
            "turnaround_time": StatCard("Avg Turnaround", lambda v: f"{v:.2f} s"),
            "response_time": StatCard("Avg Response", lambda v: f"{v:.2f} s"),
            "throughput": StatCard("Throughput", lambda v: f"{v:.2f} proc/s"),
            "cpu_utilization": StatCard("CPU Utilization", lambda v: f"{v:.1f} %"),
            "context_switches": StatCard("Context Switches", lambda v: f"{int(v)} total"),
        }

        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for (key, card), pos in zip(self.cards.items(), positions):
            card_grid.addWidget(card, *pos)

        layout.addLayout(card_grid)

        # Charts
        self.cpu_series = QLineSeries()
        self.wait_series = QLineSeries()
        self.cpu_series.setName("CPU %")
        self.wait_series.setName("Avg Waiting (s)")

        self.cpu_chart = QChart()
        self.cpu_chart.addSeries(self.cpu_series)
        self.cpu_chart.createDefaultAxes()
        self.cpu_chart.setTitle("CPU Utilization Over Time")
        self.cpu_chart.legend().setVisible(False)
        self.cpu_chart.setBackgroundBrush(QBrush(QColor("#f7f9fc")))
        self.cpu_axis_x = self.cpu_chart.axes(Qt.Orientation.Horizontal)[0]
        self.cpu_axis_y = self.cpu_chart.axes(Qt.Orientation.Vertical)[0]
        self.cpu_axis_x.setTitleText("Time (s)")
        self.cpu_axis_y.setTitleText("Utilization (%)")
        self.cpu_axis_y.setRange(0, 100)

        cpu_chart_view = QChartView(self.cpu_chart)
        cpu_chart_view.setMinimumHeight(160)
        layout.addWidget(cpu_chart_view)

        self.wait_chart = QChart()
        self.wait_chart.addSeries(self.wait_series)
        self.wait_chart.createDefaultAxes()
        self.wait_chart.setTitle("Average Waiting Time Trend")
        self.wait_chart.legend().setVisible(False)
        self.wait_chart.setBackgroundBrush(QBrush(QColor("#f7f9fc")))
        self.wait_axis_x = self.wait_chart.axes(Qt.Orientation.Horizontal)[0]
        self.wait_axis_y = self.wait_chart.axes(Qt.Orientation.Vertical)[0]
        self.wait_axis_x.setTitleText("Time (s)")
        self.wait_axis_y.setTitleText("Waiting (s)")

        wait_chart_view = QChartView(self.wait_chart)
        wait_chart_view.setMinimumHeight(160)
        layout.addWidget(wait_chart_view)

    def _trend_text(self, key: str, value: float) -> str:
        previous = self.previous_values.get(key)
        self.previous_values[key] = value
        if previous is None:
            return ""
        delta = value - previous
        if abs(delta) < 1e-4:
            return "stable"
        arrow = "▲" if delta > 0 else "▼"
        return f"{arrow} {abs(delta):.2f}"

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update displayed metrics."""
        self.cards["waiting_time"].update_value(metrics["waiting_time"], self._trend_text("waiting_time", metrics["waiting_time"]))
        self.cards["turnaround_time"].update_value(metrics["turnaround_time"], self._trend_text("turnaround_time", metrics["turnaround_time"]))
        self.cards["response_time"].update_value(metrics["response_time"], self._trend_text("response_time", metrics["response_time"]))
        self.cards["throughput"].update_value(metrics["throughput"], self._trend_text("throughput", metrics["throughput"]))
        self.cards["cpu_utilization"].update_value(metrics["cpu_utilization"], self._trend_text("cpu_utilization", metrics["cpu_utilization"]))
        self.cards["context_switches"].update_value(metrics["context_switches"], f"min ctx switches {metrics['min_context_switches']}")

        time_point = metrics["time"]

        # Update CPU chart
        self.cpu_series.append(QPointF(time_point, metrics["cpu_utilization"]))
        if self.cpu_series.count() > 400:
            self.cpu_series.removePoints(0, self.cpu_series.count() - 400)
        self.cpu_axis_x.setRange(max(0.0, time_point - 30), time_point + 1)

        # Update waiting time chart
        self.wait_series.append(QPointF(time_point, metrics["waiting_time"]))
        if self.wait_series.count() > 400:
            self.wait_series.removePoints(0, self.wait_series.count() - 400)
        self.wait_axis_x.setRange(max(0.0, time_point - 30), time_point + 1)

        max_wait = max((p.y() for p in self.wait_series.points()), default=1.0)
        self.wait_axis_y.setRange(0, max(0.5, max_wait * 1.1))


class MainWindow(QMainWindow):
    TICK_DURATION = 0.1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OS Load Simulator")
        self.setMinimumSize(1280, 840)
        self.tick_duration = self.TICK_DURATION

        self.setStyleSheet("""
            QMainWindow {
                background: #eef3fb;
            }
            QPushButton {
                background-color: #1167b1;
                color: #ffffff;
                font-weight: 600;
                padding: 10px 18px;
                border-radius: 8px;
            }
            QPushButton:disabled {
                background-color: #c4d4ea;
                color: #eef2f9;
            }
            QPushButton#chipButton {
                background-color: #ffffff;
                color: #1167b1;
                border: 1px solid #c4d4ea;
                padding: 8px 16px;
            }
            QPushButton#chipButton:checked {
                background-color: #1167b1;
                color: #ffffff;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                padding: 6px 10px;
                border-radius: 6px;
                border: 1px solid #c7d2e4;
                background: #ffffff;
            }
            QFrame#headerFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                             stop:0 #fdfbfb, stop:1 #ebedff);
                border-radius: 18px;
            }
            QFrame#controlFrame {
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid #dbe3f0;
            }
            QFrame#statCard {
                background: #ffffff;
                border-radius: 14px;
                border: 1px solid #d7e0f0;
            }
            QWidget#metricsPanel {
                background: #ffffff;
                border-radius: 18px;
                border: 1px solid #dbe3f0;
            }
            QTabWidget::pane {
                border: 1px solid #dbe3f0;
                border-radius: 12px;
                background: #ffffff;
            }
            QTabBar::tab {
                padding: 10px 18px;
                margin: 3px;
                background: #f1f5ff;
                border-radius: 12px;
                color: #4b5d7d;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #1167b1;
                color: #ffffff;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        # Header block
        header = QFrame()
        header.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 20, 28, 20)
        header_layout.setSpacing(20)

        title_block = QVBoxLayout()
        title_label = QLabel("OS Load Simulator")
        title_label.setStyleSheet("font-size:28px;font-weight:800;color:#0b2447;")
        subtitle_label = QLabel("Visualize and benchmark classic and adaptive CPU scheduling strategies.")
        subtitle_label.setStyleSheet("color:#4b5d7d;font-size:14px;")
        title_block.addWidget(title_label)
        title_block.addWidget(subtitle_label)
        header_layout.addLayout(title_block, stretch=3)

        indicator_block = QVBoxLayout()
        self.scheduler_indicator = QLabel("Scheduler: FCFS")
        self.scheduler_indicator.setStyleSheet("font-size:16px;font-weight:600;color:#1167b1;")
        self.scheduler_history_label = QLabel("")
        self.scheduler_history_label.setStyleSheet("color:#4b5d7d;font-style:italic;")
        indicator_block.addWidget(self.scheduler_indicator)
        indicator_block.addWidget(self.scheduler_history_label)
        header_layout.addLayout(indicator_block, stretch=2)

        self.auto_scheduler_mode = False
        self.auto_scheduler_sequence = ["FCFS", "SJF", "SRTF", "Round Robin", "Priority"]
        self.auto_scheduler_index = 0

        self.auto_mode_button = QPushButton("Auto Scheduler Mode")
        self.auto_mode_button.setCheckable(True)
        self.auto_mode_button.setObjectName("chipButton")
        header_layout.addWidget(self.auto_mode_button, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        main_layout.addWidget(header)

        # Control frame
        control_frame = QFrame()
        control_frame.setObjectName("controlFrame")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(28, 18, 28, 18)
        control_layout.setSpacing(32)

        scheduler_box = QVBoxLayout()
        scheduler_label = QLabel("Scheduler")
        scheduler_label.setStyleSheet("font-weight:600;color:#4b5d7d;")
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["FCFS", "SJF", "SRTF", "Round Robin", "Priority", "Adaptive"])
        scheduler_box.addWidget(scheduler_label)
        scheduler_box.addWidget(self.scheduler_combo)
        control_layout.addLayout(scheduler_box)

        quantum_box = QVBoxLayout()
        quantum_label = QLabel("Time Quantum")
        quantum_label.setStyleSheet("font-weight:600;color:#4b5d7d;")
        self.quantum_spin = QDoubleSpinBox()
        self.quantum_spin.setRange(0.1, 10.0)
        self.quantum_spin.setValue(2.0)
        self.quantum_spin.setSingleStep(0.1)
        quantum_box.addWidget(quantum_label)
        quantum_box.addWidget(self.quantum_spin)
        control_layout.addLayout(quantum_box)

        button_column = QVBoxLayout()
        sim_label = QLabel("Simulation")
        sim_label.setStyleSheet("font-weight:600;color:#4b5d7d;")
        button_bar = QHBoxLayout()
        button_bar.setSpacing(10)
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.step_button = QPushButton("Step")
        self.reset_button = QPushButton("Reset")
        button_bar.addWidget(self.start_button)
        button_bar.addWidget(self.pause_button)
        button_bar.addWidget(self.step_button)
        button_bar.addWidget(self.reset_button)
        button_column.addWidget(sim_label)
        button_column.addLayout(button_bar)
        control_layout.addLayout(button_column)

        process_box = QVBoxLayout()
        process_label = QLabel("Processes")
        process_label.setStyleSheet("font-weight:600;color:#4b5d7d;")
        self.add_process_button = QPushButton("Add Process")
        process_box.addWidget(process_label)
        process_box.addWidget(self.add_process_button)
        control_layout.addLayout(process_box)

        main_layout.addWidget(control_frame)

        # Content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter, stretch=1)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        self.gantt_chart = GanttChart()
        left_layout.addWidget(self.gantt_chart, stretch=1)

        tab_widget = QTabWidget()
        self.process_table = ProcessTable()
        self.context_table = ContextSwitchTable()
        tab_widget.addTab(self.process_table, "Processes")
        tab_widget.addTab(self.context_table, "Context Switches")
        left_layout.addWidget(tab_widget, stretch=1)

        content_splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics_panel = MetricsPanel()
        right_layout.addWidget(self.metrics_panel)
        content_splitter.addWidget(right_widget)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)

        # Connect signals
        self.auto_mode_button.toggled.connect(self.toggle_auto_scheduler_mode)
        self.add_process_button.clicked.connect(self.show_add_process_dialog)
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.step_button.clicked.connect(self.step_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)

        # Setup simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.simulation_step)
        self.sim_time = 0.0
        self.processes = []
        self.exec_log = []
        self.last_running_pid = None

        # Initialize simulation state
        self.reset_simulation()

    def show_add_process_dialog(self):
        """Show dialog to add a new process."""
        dialog = ProcessDialog(self)
        dialog.pid.setValue(len(self.processes) + 1)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            pcb = PCB(
                pid=dialog.pid.value(),
                burst_time=dialog.burst_time.value(),
                arrival_time=dialog.arrival_time.value(),
                priority=dialog.priority.value()
            )
            self.processes.append(pcb)
            self.process_table.update_process(pcb)
            self.gantt_chart.processes = self.processes

    def toggle_auto_scheduler_mode(self, checked):
        """Toggle between manual and auto scheduler modes."""
        self.auto_scheduler_mode = checked
        if checked:
            self.scheduler_combo.setEnabled(False)
            self.create_scheduler(auto=True)
            self.scheduler_history_label.setText("Auto scheduler mode engaged.")
        else:
            self.scheduler_combo.setEnabled(True)
            self.create_scheduler(auto=False)
            self.scheduler_history_label.setText("")

    def create_scheduler(self, auto=False) -> None:
        """Create a scheduler instance."""
        # In auto mode, start with FCFS and let adaptive switching handle the rest
        if auto:
            scheduler_type = "FCFS"
            # Use the adaptive controller to handle policy changes
            self.scheduler = FCFSScheduler()
        else:
            scheduler_type = self.scheduler_combo.currentText()
            if scheduler_type == "FCFS":
                self.scheduler = FCFSScheduler()
            elif scheduler_type == "SJF":
                self.scheduler = SJFScheduler()
            elif scheduler_type == "SRTF":
                self.scheduler = SRTFScheduler()
            elif scheduler_type == "Round Robin":
                self.scheduler = RoundRobinScheduler(
                    time_quantum=self.quantum_spin.value()
                )
            elif scheduler_type == "Priority":
                self.scheduler = PriorityScheduler()
            else:  # Adaptive - this case won't occur in auto mode
                self.scheduler = FCFSScheduler()  # Start with FCFS for adaptive mode
        
        self.scheduler_indicator.setText(f"Scheduler: {scheduler_type}{' (Auto)' if auto else ''}")

    def simulation_step(self):
        """Execute one step of simulation with adaptive scheduler switching."""
        current_time = self.sim_time
        next_time = current_time + self.tick_duration

        # Add arrived processes to scheduler
        for process in self.processes:
            if (process.state == ProcessState.NEW and 
                process.arrival_time <= current_time):
                self.scheduler.add_process(process, sim_time=current_time)
                process.state = ProcessState.READY
                self.process_table.update_process(process)

        # Adaptive scheduler switching every window (e.g., every 2.0 simulated seconds)
        if self.auto_scheduler_mode:
            window_size = 2.0
            if not hasattr(self, 'next_adaptive_window'):
                self.next_adaptive_window = window_size
            if current_time >= self.next_adaptive_window:
                # Prefer dataset-driven decider if available
                try:
                    from ..backend.dataset_decider import get_dataset_decider
                    dataset_decider = get_dataset_decider()
                except Exception:
                    dataset_decider = None

                from ..backend.adaptive_controller import rule_based_decider

                # Only consider currently runnable processes (ready + running)
                active_procs = []
                try:
                    ready_list = list(self.scheduler.ready_queue.get_all_processes())
                except Exception:
                    ready_list = []
                active_procs.extend(ready_list)
                if getattr(self.scheduler, 'current_process', None):
                    if self.scheduler.current_process not in active_procs:
                        active_procs.append(self.scheduler.current_process)

                # use dataset decider first; fallback to rule-based
                if dataset_decider is not None:
                    decision = dataset_decider.decide(self.sim_time, active_procs)
                else:
                    decision = rule_based_decider(self.sim_time, active_procs)
                new_policy = None
                policy_str = str(decision.get('policy', '')).upper()
                if 'SJF' in policy_str and 'SRTF' not in policy_str:
                    new_policy = 'SJF'
                elif 'SRTF' in policy_str:
                    new_policy = 'SRTF'
                elif 'FCFS' in policy_str:
                    new_policy = 'FCFS'
                elif 'RR' in policy_str or 'ROUND' in policy_str:
                    new_policy = 'Round Robin'
                elif 'PRIOR' in policy_str or 'PRIORITY' in policy_str:
                    new_policy = 'PRIORITY'
                reason = decision.get('reason', '')
                # Map policy string to scheduler class
                scheduler_map = {
                    'FCFS': FCFSScheduler,
                    'SJF': SJFScheduler,
                    'SRTF': SRTFScheduler,
                    'Round Robin': RoundRobinScheduler,
                    'PRIORITY': PriorityScheduler,
                }
                if new_policy and scheduler_map.get(new_policy):
                    # Only switch if policy changes
                    if type(self.scheduler) != scheduler_map[new_policy]:
                        current_processes = [p for p in self.scheduler.ready_queue.get_all_processes()]
                        current_running = getattr(self.scheduler, "current_process", None)
                        # Preserve metrics across scheduler switches so averages remain meaningful
                        old_metrics = getattr(self.scheduler, 'metrics', None)
                        new_sched = scheduler_map[new_policy]()
                        if old_metrics is not None:
                            new_sched.metrics = old_metrics
                        self.scheduler = new_sched
                        for p in current_processes:
                            self.scheduler.add_process(p, sim_time=current_time)
                        if current_running and current_running.remaining_time > 0:
                            current_running.state = ProcessState.READY
                            self.scheduler.add_process(current_running, sim_time=current_time)
                        self.scheduler_indicator.setText(f"Scheduler: {new_policy} (Auto)")
                        self.scheduler_history_label.setText(f"Auto-switched to {new_policy} at t={current_time:.1f}s ({reason})")

                self.next_adaptive_window += window_size

        # Get next process from scheduler (pass simulated time)
        prev_proc = self.scheduler.current_process
        current = self.scheduler.schedule(current_time)

        # Detect context switch (prev_proc != current)
        if prev_proc is not current:
            # Close previous exec segment
            if self.exec_log and self.exec_log[-1].get('end') is None:
                self.exec_log[-1]['end'] = current_time

            reason = "dispatch"
            if prev_proc and prev_proc.remaining_time <= 0:
                reason = "completed"
            elif prev_proc and current:
                reason = "quantum expired" if isinstance(self.scheduler, RoundRobinScheduler) else "preempted"
            elif prev_proc and current is None:
                reason = "idle"

            self.context_table.add_event(
                current_time,
                prev_proc.pid if prev_proc else None,
                current.pid if current else None,
                reason,
                type(self.scheduler).__name__,
            )

            # Start a new segment for the newly running process
            if current is not None:
                seg = {
                    'pid': current.pid,
                    'start': current_time,
                    'end': None,
                    'scheduler': type(self.scheduler).__name__,
                    'reason': reason,
                }
                self.exec_log.append(seg)
                self.last_running_pid = current.pid
            else:
                self.last_running_pid = None

        # If there's a running process, advance it
        if current:
            # Update process (advance simulated time slice)
            current.remaining_time = max(0.0, current.remaining_time - self.tick_duration)
            self.process_table.update_process(current)

            # If process completes, close its segment and record completion
            if current.remaining_time <= 1e-9:
                # Mark completion
                current.state = ProcessState.TERMINATED
                try:
                    self.scheduler.complete_process(current, sim_time=next_time)
                except Exception:
                    # Some schedulers may handle completion elsewhere; ignore failures
                    pass

                # Close last segment
                if self.exec_log and self.exec_log[-1].get('end') is None:
                    self.exec_log[-1]['end'] = next_time
                self.last_running_pid = None
                self.scheduler.current_process = None

        # Refresh table entries for any ready or completed processes
        for pcb in self.scheduler.metrics.process_history.values():
            self.process_table.update_process(pcb)

        # Compute minimum context-switches across completed processes
        min_cs = 0
        completed = [p for p in self.scheduler.metrics.process_history.values() if p.stats.completion_time is not None]
        if completed:
            min_cs = min(p.stats.context_switches for p in completed)

        # Update scheduler-internal metrics using simulated time
        try:
            self.scheduler.update_metrics(next_time)
        except Exception:
            pass

        metrics = {
            'waiting_time': self.scheduler.metrics.get_avg_waiting_time(),
            'turnaround_time': self.scheduler.metrics.get_avg_turnaround_time(),
            'response_time': self.scheduler.metrics.get_avg_response_time(),
            'context_switches': self.scheduler.metrics.total_context_switches,
            'min_context_switches': int(min_cs),
            'throughput': self.scheduler.metrics.get_throughput(),
            'time': next_time,
            'cpu_utilization': self.scheduler.metrics.get_cpu_utilization()
        }
        self.metrics_panel.update_metrics(metrics)

        # Update Gantt chart with exec_log
        self.gantt_chart.set_exec_log(self.exec_log)
        self.gantt_chart.update_time(next_time)

        # Check if simulation is complete
        if (all(p.state == ProcessState.TERMINATED for p in self.processes) and
            len(self.processes) > 0):
            self.pause_simulation()

        # Increment simulation time
        self.sim_time = next_time

    def reset_simulation(self):
        """Reset simulation state."""
        self.sim_time = 0.0
        self.processes = []
        self.process_table.setRowCount(0)
        if hasattr(self, "context_table"):
            self.context_table.setRowCount(0)
        self.create_scheduler(auto=self.auto_scheduler_mode)
        self.scheduler_history_label.setText("")
        self.exec_log = []
        self.last_running_pid = None
        self.gantt_chart.set_exec_log([])
        self.metrics_panel.cpu_series.clear()
        self.metrics_panel.wait_series.clear()
        self.metrics_panel.previous_values.clear()
        self.metrics_panel.update_metrics({
            'waiting_time': 0.0,
            'turnaround_time': 0.0,
            'response_time': 0.0,
            'context_switches': 0,
            'min_context_switches': 0,
            'throughput': 0.0,
            'time': 0.0,
            'cpu_utilization': 0.0
        })

    def start_simulation(self):
        """Start or resume simulation."""
        self.sim_timer.start(100)  # 100ms intervals
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)

    def pause_simulation(self):
        """Pause simulation."""
        self.sim_timer.stop()
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)

    def step_simulation(self):
        """Step through simulation."""
        self.simulation_step()
