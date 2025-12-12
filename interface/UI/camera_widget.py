"""
工业视觉检测系统 - 摄像头组件
负责摄像头控制、视频显示和基本图像处理
"""

import logging
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QCheckBox,
    QSlider, QSpinBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QFrame, QFileDialog, QMessageBox
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QTimer, QSize, 
    QThread, QMutex, QWaitCondition
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor,
    QFont, QFontMetrics
)

class VideoDisplayWidget(QFrame):
    """视频显示组件"""
    
    # 信号：点击了显示区域
    clicked = pyqtSignal(int, int)  # x, y坐标
    
    def __init__(self):
        """初始化视频显示组件"""
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setLineWidth(2)
        
        # 视频图像
        self.current_frame = None
        self.display_frame = None
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        
        # 显示设置
        self.show_fps = True
        self.show_timestamp = True
        self.show_grid = False
        self.show_crosshair = False
        self.display_mode = "original"  # original, grayscale, edges, binary
        
        # 标定数据
        self.calibration_data = {
            'pixel_per_mm': 1.0,
            'reference_length': 10.0,  # mm
            'reference_pixels': 100  # pixels
        }
        
        # ROI区域
        self.roi_rect = None
        self.is_drawing_roi = False
        self.roi_start_pos = None
        
        # 测量线
        self.measure_line = None
        self.is_measuring = False
        self.measure_start_pos = None
        
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        
        # 更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(33)  # ~30 FPS更新
    
    def set_frame(self, frame):
        """设置当前帧"""
        if frame is not None and frame.size > 0:
            self.current_frame = frame.copy()
            self.process_frame_for_display()
    
    def process_frame_for_display(self):
        """处理帧以进行显示"""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # 根据显示模式处理图像
        if self.display_mode == "grayscale" and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == "edges":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif self.display_mode == "binary":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 缩放和平移
        if self.zoom_factor != 1.0:
            h, w = frame.shape[:2]
            new_size = (int(w * self.zoom_factor), int(h * self.zoom_factor))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        
        # 转换为QPixmap用于显示
        self.display_frame = self.convert_cv_to_qpixmap(frame)
    
    def convert_cv_to_qpixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        if cv_img is None:
            return QPixmap()
        
        h, w = cv_img.shape[:2]
        if len(cv_img.shape) == 2:  # 灰度图
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, 
                          QImage.Format.Format_Grayscale8)
        else:  # 彩色图
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, 
                          QImage.Format.Format_RGB888)
        
        return QPixmap.fromImage(q_img)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        # 绘制图像
        if self.display_frame and not self.display_frame.isNull():
            # 计算居中位置
            pixmap_rect = self.display_frame.rect()
            view_rect = self.rect()
            
            # 计算缩放以保持宽高比
            pixmap_rect = pixmap_rect.scaled(view_rect.size(), 
                                           Qt.AspectRatioMode.KeepAspectRatio)
            
            # 居中
            x = (view_rect.width() - pixmap_rect.width()) // 2
            y = (view_rect.height() - pixmap_rect.height()) // 2
            pixmap_rect.moveTo(x, y)
            
            # 绘制图像
            painter.drawPixmap(pixmap_rect, self.display_frame, 
                             self.display_frame.rect())
            
            # 绘制ROI区域
            if self.roi_rect:
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawRect(*self.roi_rect)
                
                # 显示ROI信息
                if self.roi_rect[2] > 0 and self.roi_rect[3] > 0:
                    info_text = f"ROI: {self.roi_rect[2]}x{self.roi_rect[3]}"
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.setFont(QFont("Arial", 10))
                    painter.drawText(self.roi_rect[0] + 5, 
                                   self.roi_rect[1] + 20, info_text)
            
            # 绘制测量线
            if self.measure_line:
                x1, y1, x2, y2 = self.measure_line
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(x1, y1, x2, y2)
                
                # 绘制端点
                painter.setBrush(QColor(255, 0, 0))
                painter.drawEllipse(x1 - 3, y1 - 3, 6, 6)
                painter.drawEllipse(x2 - 3, y2 - 3, 6, 6)
                
                # 计算并显示距离
                dx = x2 - x1
                dy = y2 - y1
                pixel_distance = np.sqrt(dx*dx + dy*dy)
                mm_distance = pixel_distance / self.calibration_data['pixel_per_mm']
                
                # 在线的中点显示距离
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                distance_text = f"{mm_distance:.2f} mm ({pixel_distance:.1f} px)"
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setFont(QFont("Arial", 10))
                
                # 计算文本大小并绘制背景
                font_metrics = QFontMetrics(painter.font())
                text_width = font_metrics.horizontalAdvance(distance_text)
                text_height = font_metrics.height()
                
                bg_rect = (mid_x - text_width//2 - 5, mid_y - text_height - 5,
                          text_width + 10, text_height + 5)
                painter.fillRect(*bg_rect, QColor(0, 0, 0, 180))
                painter.drawText(mid_x - text_width//2, mid_y - 5, distance_text)
            
            # 绘制网格
            if self.show_grid:
                painter.setPen(QPen(QColor(100, 100, 100, 100), 1))
                grid_size = 50
                for x in range(pixmap_rect.left(), pixmap_rect.right(), grid_size):
                    painter.drawLine(x, pixmap_rect.top(), x, pixmap_rect.bottom())
                for y in range(pixmap_rect.top(), pixmap_rect.bottom(), grid_size):
                    painter.drawLine(pixmap_rect.left(), y, pixmap_rect.right(), y)
            
            # 绘制十字准线
            if self.show_crosshair:
                center_x = pixmap_rect.center().x()
                center_y = pixmap_rect.center().y()
                painter.setPen(QPen(QColor(255, 255, 0, 150), 1))
                painter.drawLine(center_x, pixmap_rect.top(), 
                               center_x, pixmap_rect.bottom())
                painter.drawLine(pixmap_rect.left(), center_y, 
                               pixmap_rect.right(), center_y)
            
            # 绘制信息覆盖层
            self.draw_overlay(painter, pixmap_rect)
    
    def draw_overlay(self, painter, pixmap_rect):
        """绘制信息覆盖层"""
        # 绘制边框
        painter.setPen(QPen(QColor(100, 100, 255), 2))
        painter.drawRect(pixmap_rect)
        
        # 绘制时间戳和FPS
        if self.show_timestamp or self.show_fps:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_text = f"{timestamp}"
            
            # TODO: 从CameraManager获取实际FPS
            if self.show_fps:
                info_text += f" | FPS: 30.0"
            
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Arial", 10))
            
            # 在左上角绘制
            font_metrics = QFontMetrics(painter.font())
            text_height = font_metrics.height()
            
            # 绘制背景
            bg_rect = (pixmap_rect.left() + 5, pixmap_rect.top() + 5,
                      font_metrics.horizontalAdvance(info_text) + 10,
                      text_height + 5)
            painter.fillRect(*bg_rect, QColor(0, 0, 0, 150))
            
            # 绘制文本
            painter.drawText(pixmap_rect.left() + 10, 
                           pixmap_rect.top() + 10 + font_metrics.ascent(),
                           info_text)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            self.clicked.emit(pos.x(), pos.y())
            
            # ROI绘制模式
            if self.is_drawing_roi:
                self.roi_start_pos = pos
                self.roi_rect = (pos.x(), pos.y(), 0, 0)
            # 测量模式
            elif self.is_measuring:
                self.measure_start_pos = pos
                self.measure_line = (pos.x(), pos.y(), pos.x(), pos.y())
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        # 更新ROI绘制
        if self.is_drawing_roi and self.roi_start_pos:
            start_pos = self.roi_start_pos
            current_pos = event.pos()
            
            x = min(start_pos.x(), current_pos.x())
            y = min(start_pos.y(), current_pos.y())
            width = abs(current_pos.x() - start_pos.x())
            height = abs(current_pos.y() - start_pos.y())
            
            self.roi_rect = (x, y, width, height)
            self.update()
        
        # 更新测量线
        elif self.is_measuring and self.measure_start_pos:
            x1, y1 = self.measure_start_pos.x(), self.measure_start_pos.y()
            x2, y2 = event.pos().x(), event.pos().y()
            self.measure_line = (x1, y1, x2, y2)
            self.update()
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_drawing_roi:
                self.is_drawing_roi = False
            elif self.is_measuring:
                self.is_measuring = False
        
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """鼠标滚轮事件 - 缩放控制"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor = min(self.zoom_factor * 1.1, 5.0)
        else:
            self.zoom_factor = max(self.zoom_factor / 1.1, 0.1)
        
        self.process_frame_for_display()
        self.update()
        event.accept()
    
    def reset_view(self):
        """重置视图"""
        self.zoom_factor = 1.0
        self.pan_offset = (0, 0)
        self.process_frame_for_display()
        self.update()
    
    def set_display_mode(self, mode):
        """设置显示模式"""
        self.display_mode = mode
        self.process_frame_for_display()
        self.update()
    
    def start_roi_selection(self):
        """开始ROI选择"""
        self.is_drawing_roi = True
        self.roi_rect = None
    
    def start_measurement(self):
        """开始测量"""
        self.is_measuring = True
        self.measure_line = None
    
    def clear_overlays(self):
        """清除覆盖层"""
        self.roi_rect = None
        self.measure_line = None
        self.update()

class CameraControlPanel(QGroupBox):
    """摄像头控制面板"""
    
    # 信号
    camera_settings_changed = pyqtSignal(dict)
    capture_requested = pyqtSignal()
    recording_toggled = pyqtSignal(bool)
    snapshot_requested = pyqtSignal()
    
    def __init__(self):
        """初始化摄像头控制面板"""
        super().__init__("摄像头控制")
        
        self.logger = logging.getLogger(__name__)
        self.init_ui()
        self.set_camera_settings()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 摄像头选择
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("摄像头:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2", "摄像头 3"])
        camera_layout.addWidget(self.camera_combo)
        
        self.refresh_cameras_btn = QPushButton("刷新")
        self.refresh_cameras_btn.setFixedWidth(60)
        camera_layout.addWidget(self.refresh_cameras_btn)
        
        layout.addLayout(camera_layout)
        
        # 分辨率选择
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("分辨率:"))
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x480", "800x600", "1024x768", 
            "1280x720", "1920x1080", "自定义"
        ])
        resolution_layout.addWidget(self.resolution_combo)
        layout.addLayout(resolution_layout)
        
        # 帧率控制
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("帧率:"))
        
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(1, 60)
        self.fps_slider.setValue(30)
        fps_layout.addWidget(self.fps_slider)
        
        self.fps_label = QLabel("30 FPS")
        self.fps_label.setFixedWidth(50) 
        fps_layout.addWidget(self.fps_label)
        layout.addLayout(fps_layout)
        
        # 图像设置
        image_group = QGroupBox("图像设置")
        image_layout = QFormLayout()
        
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        image_layout.addRow("亮度:", self.brightness_slider)
        
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        image_layout.addRow("对比度:", self.contrast_slider)
        
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        image_layout.addRow("饱和度:", self.saturation_slider)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # 高级设置
        advanced_group = QGroupBox("高级设置")
        advanced_layout = QVBoxLayout()
        
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("曝光:"))
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(-10, 10)
        self.exposure_spin.setSingleStep(0.1)
        self.exposure_spin.setValue(0)
        exposure_layout.addWidget(self.exposure_spin)
        advanced_layout.addLayout(exposure_layout)
        
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("增益:"))
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0, 10)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1)
        gain_layout.addWidget(self.gain_spin)
        advanced_layout.addLayout(gain_layout)
        
        self.auto_exposure_check = QCheckBox("自动曝光")
        self.auto_exposure_check.setChecked(True)
        advanced_layout.addWidget(self.auto_exposure_check)
        
        self.auto_whitebalance_check = QCheckBox("自动白平衡")
        self.auto_whitebalance_check.setChecked(True)
        advanced_layout.addWidget(self.auto_whitebalance_check)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # 控制按钮
        button_layout = QGridLayout()
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        button_layout.addWidget(self.connect_btn, 0, 0)
        
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setEnabled(False)
        button_layout.addWidget(self.disconnect_btn, 0, 1)
        
        self.capture_btn = QPushButton("拍照")
        button_layout.addWidget(self.capture_btn, 1, 0)
        
        self.record_btn = QPushButton("录制")
        self.record_btn.setCheckable(True)
        button_layout.addWidget(self.record_btn, 1, 1)
        
        self.snapshot_btn = QPushButton("快照")
        button_layout.addWidget(self.snapshot_btn, 2, 0, 1, 2)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # 连接信号
        self.connect_signals()
    
    def connect_signals(self):
        """连接信号"""
        self.fps_slider.valueChanged.connect(
            lambda v: self.fps_label.setText(f"{v} FPS")
        )
        
        # 图像设置变化
        for slider in [self.brightness_slider, self.contrast_slider, 
                      self.saturation_slider]:
            slider.valueChanged.connect(self.on_settings_changed)
        
        # 高级设置变化
        self.exposure_spin.valueChanged.connect(self.on_settings_changed)
        self.gain_spin.valueChanged.connect(self.on_settings_changed)
        self.auto_exposure_check.stateChanged.connect(self.on_settings_changed)
        self.auto_whitebalance_check.stateChanged.connect(self.on_settings_changed)
        
        # 按钮信号
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.disconnect_btn.clicked.connect(self.on_disconnect_clicked)
        self.capture_btn.clicked.connect(self.capture_requested.emit)
        self.record_btn.toggled.connect(self.recording_toggled.emit)
        self.snapshot_btn.clicked.connect(self.snapshot_requested.emit)
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)
    
    def on_settings_changed(self):
        """设置变化处理"""
        settings = self.get_camera_settings()
        self.camera_settings_changed.emit(settings)
    
    def get_camera_settings(self):
        """获取摄像头设置"""
        # 解析分辨率
        resolution_str = self.resolution_combo.currentText()
        if "x" in resolution_str:
            width, height = map(int, resolution_str.split("x"))
        else:
            width, height = 640, 480
        
        return {
            'camera_index': self.camera_combo.currentIndex(),
            'resolution': (width, height),
            'fps': self.fps_slider.value(),
            'brightness': self.brightness_slider.value(),
            'contrast': self.contrast_slider.value(),
            'saturation': self.saturation_slider.value(),
            'exposure': self.exposure_spin.value(),
            'gain': self.gain_spin.value(),
            'auto_exposure': self.auto_exposure_check.isChecked(),
            'auto_whitebalance': self.auto_whitebalance_check.isChecked()
        }
    
    def set_camera_settings(self, settings=None):
        """设置摄像头设置"""
        if settings:
            # TODO: 根据设置更新UI控件
            pass
    
    def on_connect_clicked(self):
        """连接按钮点击"""
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.refresh_cameras_btn.setEnabled(False)
        
        # 发送连接请求
        settings = self.get_camera_settings()
        self.camera_settings_changed.emit({**settings, 'action': 'connect'})
    
    def on_disconnect_clicked(self):
        """断开按钮点击"""
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.refresh_cameras_btn.setEnabled(True)
        
        # 发送断开请求
        self.camera_settings_changed.emit({'action': 'disconnect'})
    
    def refresh_cameras(self):
        """刷新摄像头列表"""
        self.camera_combo.clear()
        self.camera_combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2", "摄像头 3"])
        self.logger.info("刷新摄像头列表")

class CameraWidget(QWidget):
    """摄像头主组件"""
    
    # 信号
    camera_selected = pyqtSignal(int)
    camera_connected = pyqtSignal()
    camera_disconnected = pyqtSignal()
    
    def __init__(self, camera_manager):
        """初始化摄像头组件
        
        Args:
            camera_manager: 摄像头管理器实例
        """
        super().__init__()
        
        self.camera_manager = camera_manager
        self.logger = logging.getLogger(__name__)
        
        # 状态变量
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.snapshot_count = 0
        self._camera_connected = False # 摄像头连接状态
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 初始化帧定时器
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frame)
        self.frame_timer.start(33)  # ~30 FPS
        
        self.logger.info("摄像头组件初始化完成")
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # 左侧：视频显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频显示
        self.video_display = VideoDisplayWidget()
        left_layout.addWidget(self.video_display, 1)
        
        # 显示控制工具栏
        self.create_display_toolbar(left_layout)
        
        main_layout.addWidget(left_widget, 3)  # 3/4宽度
        
        # 右侧：控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 摄像头控制面板
        self.control_panel = CameraControlPanel()
        right_layout.addWidget(self.control_panel)
        
        # 标定面板
        self.create_calibration_panel(right_layout)
        
        # 图像处理面板
        self.create_image_processing_panel(right_layout)
        
        right_layout.addStretch()
        main_layout.addWidget(right_widget, 1)  # 1/4宽度
        
        self.setLayout(main_layout)
    
    def create_display_toolbar(self, parent_layout):
        """创建显示控制工具栏"""
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        toolbar.setMaximumHeight(40)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # 显示模式
        layout.addWidget(QLabel("显示模式:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["原始", "灰度", "边缘", "二值化"])
        self.display_mode_combo.currentTextChanged.connect(
            lambda t: self.video_display.set_display_mode(
                {"原始": "original", "灰度": "grayscale", 
                 "边缘": "edges", "二值化": "binary"}[t]
            )
        )
        layout.addWidget(self.display_mode_combo)
        
        layout.addSpacing(10)
        
        # 显示选项
        self.show_grid_check = QCheckBox("网格")
        self.show_grid_check.toggled.connect(
            lambda c: setattr(self.video_display, 'show_grid', c)
        )
        layout.addWidget(self.show_grid_check)
        
        self.show_crosshair_check = QCheckBox("十字线")
        self.show_crosshair_check.toggled.connect(
            lambda c: setattr(self.video_display, 'show_crosshair', c)
        )
        layout.addWidget(self.show_crosshair_check)
        
        self.show_fps_check = QCheckBox("FPS")
        self.show_fps_check.setChecked(True)
        self.show_fps_check.toggled.connect(
            lambda c: setattr(self.video_display, 'show_fps', c)
        )
        layout.addWidget(self.show_fps_check)
        
        layout.addSpacing(10)
        
        # 工具按钮
        self.roi_btn = QPushButton("ROI")
        self.roi_btn.clicked.connect(self.video_display.start_roi_selection)
        layout.addWidget(self.roi_btn)
        
        self.measure_btn = QPushButton("测量")
        self.measure_btn.clicked.connect(self.video_display.start_measurement)
        layout.addWidget(self.measure_btn)
        
        self.clear_btn = QPushButton("清除")
        self.clear_btn.clicked.connect(self.video_display.clear_overlays)
        layout.addWidget(self.clear_btn)
        
        self.zoom_reset_btn = QPushButton("重置缩放")
        self.zoom_reset_btn.clicked.connect(self.video_display.reset_view)
        layout.addWidget(self.zoom_reset_btn)
        
        layout.addStretch()
        
        # 录制状态显示
        self.recording_label = QLabel()
        self.recording_label.setStyleSheet("color: #ff0000; font-weight: bold;")
        layout.addWidget(self.recording_label)
        
        parent_layout.addWidget(toolbar)
    
    def create_calibration_panel(self, parent_layout):
        """创建标定面板"""
        cal_group = QGroupBox("标定设置")
        cal_layout = QVBoxLayout()
        
        # 参考长度输入
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("参考长度 (mm):"))
        self.ref_length_spin = QDoubleSpinBox()
        self.ref_length_spin.setRange(0.1, 1000)
        self.ref_length_spin.setValue(10.0)
        self.ref_length_spin.setDecimals(2)
        ref_layout.addWidget(self.ref_length_spin)
        cal_layout.addLayout(ref_layout)
        
        # 参考像素输入
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("参考像素:"))
        self.ref_pixel_spin = QSpinBox()
        self.ref_pixel_spin.setRange(1, 10000)
        self.ref_pixel_spin.setValue(100)
        pixel_layout.addWidget(self.ref_pixel_spin)
        cal_layout.addLayout(pixel_layout)
        
        # 标定按钮
        self.calibrate_btn = QPushButton("标定")
        self.calibrate_btn.clicked.connect(self.perform_calibration)
        cal_layout.addWidget(self.calibrate_btn)
        
        # 标定结果显示
        self.calibration_label = QLabel("像素/毫米: 1.00")
        self.calibration_label.setStyleSheet("font-weight: bold;")
        cal_layout.addWidget(self.calibration_label)
        
        cal_group.setLayout(cal_layout)
        parent_layout.addWidget(cal_group)
    
    def create_image_processing_panel(self, parent_layout):
        """创建图像处理面板"""
        proc_group = QGroupBox("图像处理")
        proc_layout = QVBoxLayout()
        
        # 图像调整
        adjust_layout = QGridLayout()
        
        adjust_layout.addWidget(QLabel("亮度:"), 0, 0)
        self.brightness_adjust = QSlider(Qt.Orientation.Horizontal)
        self.brightness_adjust.setRange(-100, 100)
        self.brightness_adjust.setValue(0)
        adjust_layout.addWidget(self.brightness_adjust, 0, 1)
        
        adjust_layout.addWidget(QLabel("对比度:"), 1, 0)
        self.contrast_adjust = QSlider(Qt.Orientation.Horizontal)
        self.contrast_adjust.setRange(-100, 100)
        self.contrast_adjust.setValue(0)
        adjust_layout.addWidget(self.contrast_adjust, 1, 1)
        
        adjust_layout.addWidget(QLabel("伽马:"), 2, 0)
        self.gamma_adjust = QSlider(Qt.Orientation.Horizontal)
        self.gamma_adjust.setRange(10, 300)
        self.gamma_adjust.setValue(100)
        adjust_layout.addWidget(self.gamma_adjust, 2, 1)
        
        proc_layout.addLayout(adjust_layout)
        
        # 边缘检测
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(QLabel("边缘检测:"))
        self.edge_threshold1 = QSpinBox()
        self.edge_threshold1.setRange(0, 255)
        self.edge_threshold1.setValue(50)
        edge_layout.addWidget(self.edge_threshold1)
        
        self.edge_threshold2 = QSpinBox()
        self.edge_threshold2.setRange(0, 255)
        self.edge_threshold2.setValue(150)
        edge_layout.addWidget(self.edge_threshold2)
        
        proc_layout.addLayout(edge_layout)
        
        # 处理按钮
        self.apply_processing_btn = QPushButton("应用处理")
        self.apply_processing_btn.clicked.connect(self.apply_image_processing)
        proc_layout.addWidget(self.apply_processing_btn)
        
        self.reset_processing_btn = QPushButton("重置处理")
        self.reset_processing_btn.clicked.connect(self.reset_image_processing)
        proc_layout.addWidget(self.reset_processing_btn)
        
        proc_group.setLayout(proc_layout)
        parent_layout.addWidget(proc_group)
    
    def connect_signals(self):
        """连接信号"""
        # 摄像头管理器信号
        if hasattr(self.camera_manager, 'frame_received'):
            self.camera_manager.frame_received.connect(self.on_frame_received)
        
        # 控制面板信号
        self.control_panel.camera_settings_changed.connect(
            self.on_camera_settings_changed
        )
        self.control_panel.capture_requested.connect(self.capture_image)
        self.control_panel.recording_toggled.connect(self.toggle_recording)
        self.control_panel.snapshot_requested.connect(self.take_snapshot)
        
        # 视频显示信号
        self.video_display.clicked.connect(self.on_video_clicked)
    
    def on_camera_settings_changed(self, settings):
        """摄像头设置变化处理"""
        action = settings.get('action')
        
        if action == 'connect':
            camera_index = settings.get('camera_index', 0)
            self.connect_camera(camera_index)
        elif action == 'disconnect':
            self.disconnect_camera()
        else:
            # 更新摄像头参数
            self.update_camera_parameters(settings)
    
    def connect_camera(self, camera_index):
        """连接摄像头"""
        try:
            if self.camera_manager.connect_camera(camera_index):
                self._camera_connected = True
                self.camera_connected.emit()
                self.camera_selected.emit(camera_index)
                self.logger.info(f"摄像头 {camera_index} 连接成功")
            else:
                self.logger.error(f"摄像头 {camera_index} 连接失败")
                self._camera_connected = False
        except Exception as e:
            self.logger.error(f"摄像头连接错误: {e}")
            QMessageBox.critical(self, "摄像头错误", f"连接失败: {str(e)}")
            self._camera_connected = False
    
    def disconnect_camera(self):
        """断开摄像头"""
        try:
            self.camera_manager.disconnect_camera()
            self._camera_connected = False
            self.camera_disconnected.emit()
            self.logger.info("摄像头已断开")
        except Exception as e:
            self.logger.error(f"摄像头断开错误: {e}")
            self._camera_connected = False
    
    def update_camera_parameters(self, settings):
        """更新摄像头参数"""
        # TODO: 实现摄像头参数更新
        self.logger.debug(f"更新摄像头参数: {settings}")
    
    @pyqtSlot(object)
    def on_frame_received(self, frame_info):
        """接收到新帧"""
        frame = frame_info.get('frame')
        if frame is not None:
            # 处理图像（如果需要）
            processed_frame = self.process_frame(frame)
            
            # 更新显示
            self.video_display.set_frame(processed_frame)
            
            # 录制视频
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(processed_frame)
    
    def process_frame(self, frame):
        """处理图像帧"""
        if frame is None:
            return None
        
        processed = frame.copy()
        
        # 应用图像调整
        brightness = self.brightness_adjust.value() / 100.0
        contrast = self.contrast_adjust.value() / 100.0
        
        if brightness != 0 or contrast != 0:
            processed = cv2.convertScaleAbs(processed, alpha=1 + contrast, 
                                          beta=255 * brightness)
        
        # 应用伽马校正
        gamma = self.gamma_adjust.value() / 100.0
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            processed = cv2.LUT(processed, table)
        
        return processed
    
    def update_frame(self):
        """更新帧显示"""
        # 如果摄像头管理器有新的帧，会通过信号传递
        # 这里主要处理非实时更新或手动刷新
        pass
    
    def capture_image(self):
        """捕获图像"""
        if not self.camera_manager.is_connected():
            QMessageBox.warning(self, "警告", "请先连接摄像头")
            return
        
        try:
            frame = self.camera_manager.capture_frame()
            if frame is not None:
                # 保存图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                save_path = Path("captures") / filename
                save_path.parent.mkdir(exist_ok=True)
                
                cv2.imwrite(str(save_path), frame)
                self.logger.info(f"图像已保存: {save_path}")
                QMessageBox.information(self, "成功", f"图像已保存到: {save_path}")
            else:
                QMessageBox.warning(self, "警告", "捕获图像失败")
        except Exception as e:
            self.logger.error(f"捕获图像错误: {e}")
            QMessageBox.critical(self, "错误", f"捕获失败: {str(e)}")
    
    def toggle_recording(self, start):
        """切换录制状态"""
        if start:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录制视频"""
        if not self.is_camera_connected():
            QMessageBox.warning(self, "警告", "请先连接摄像头")
            self.control_panel.record_btn.setChecked(False)
            return
        
        try:
            # 获取摄像头信息
            camera_info = self.camera_manager.get_camera_info()
            if not camera_info:
                raise ValueError("无法获取摄像头信息")
            
            # 创建视频写入器
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"
            save_path = Path("recordings") / filename
            save_path.parent.mkdir(exist_ok=True)
            
            fps = camera_info.get('fps', 30)
            width = camera_info.get('width', 640)
            height = camera_info.get('height', 480)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                str(save_path), fourcc, fps, (width, height)
            )
            
            if not self.video_writer.isOpened():
                raise ValueError("无法创建视频文件")
            
            self.is_recording = True
            self.recording_start_time = datetime.now()
            
            # 更新UI
            self.recording_label.setText("● 录制中")
            self.logger.info(f"开始录制视频: {save_path}")
            
        except Exception as e:
            self.logger.error(f"开始录制错误: {e}")
            QMessageBox.critical(self, "错误", f"开始录制失败: {str(e)}")
            self.control_panel.record_btn.setChecked(False)

    def is_camera_connected(self):
        """检查摄像头是否连接"""
        return self._camera_connected
    
    def stop_recording(self):
        """停止录制视频"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        
        # 计算录制时长
        if self.recording_start_time:
            duration = (datetime.now() - self.recording_start_time).total_seconds()
            self.logger.info(f"停止录制，时长: {duration:.1f}秒")
        
        # 更新UI
        self.recording_label.setText("")
        self.logger.info("录制已停止")
    
    def take_snapshot(self):
        """拍摄快照"""
        if not self.camera_manager.is_connected():
            return
        
        frame = self.camera_manager.capture_frame()
        if frame is not None:
            self.snapshot_count += 1
            self.video_display.set_frame(frame)
            self.logger.info(f"快照 #{self.snapshot_count} 已拍摄")
    
    def perform_calibration(self):
        """执行标定"""
        ref_length = self.ref_length_spin.value()
        ref_pixels = self.ref_pixel_spin.value()
        
        if ref_length <= 0 or ref_pixels <= 0:
            QMessageBox.warning(self, "警告", "参考长度和像素必须大于0")
            return
        
        pixel_per_mm = ref_pixels / ref_length
        self.video_display.calibration_data['pixel_per_mm'] = pixel_per_mm
        self.video_display.calibration_data['reference_length'] = ref_length
        self.video_display.calibration_data['reference_pixels'] = ref_pixels
        
        self.calibration_label.setText(f"像素/毫米: {pixel_per_mm:.2f}")
        self.logger.info(f"标定完成: {pixel_per_mm:.2f} 像素/毫米")
    
    def apply_image_processing(self):
        """应用图像处理"""
        # 处理已在process_frame中实时应用
        self.logger.debug("图像处理已应用")
    
    def reset_image_processing(self):
        """重置图像处理"""
        self.brightness_adjust.setValue(0)
        self.contrast_adjust.setValue(0)
        self.gamma_adjust.setValue(100)
        self.edge_threshold1.setValue(50)
        self.edge_threshold2.setValue(150)
        
        self.logger.debug("图像处理已重置")
    
    def on_video_clicked(self, x, y):
        """视频区域点击处理"""
        self.logger.debug(f"视频点击位置: ({x}, {y})")
        # TODO: 可以添加点击交互功能
    
    def closeEvent(self, event):
        """关闭事件"""
        # 停止录制
        if self.is_recording:
            self.stop_recording()
        
        # 停止帧定时器
        if self.frame_timer.isActive():
            self.frame_timer.stop()
        
        super().closeEvent(event)