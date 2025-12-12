#!/usr/bin/env python3
"""
摄像头UI测试程序
使用PyQt6的简单界面测试摄像头显示
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QComboBox,
    QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np

class SimpleCameraWidget(QWidget):
    """简单的摄像头显示组件"""
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.current_frame = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 摄像头选择
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("摄像头设备:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2", "摄像头 3"])
        control_layout.addWidget(self.camera_combo)
        
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.connect_camera)
        control_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self.disconnect_camera)
        control_layout.addWidget(self.disconnect_btn)
        
        self.capture_btn = QPushButton("拍照")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_image)
        control_layout.addWidget(self.capture_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        layout.addWidget(self.video_label, 1)
        
        # 信息显示
        self.info_label = QLabel("未连接")
        self.info_label.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(self.info_label)
        
        # 定时器用于更新帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def connect_camera(self):
        """连接摄像头"""
        device_id = self.camera_combo.currentIndex()
        
        try:
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                QMessageBox.warning(self, "警告", f"无法打开摄像头 {device_id}")
                return
            
            # 获取摄像头信息
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.info_label.setText(f"已连接: {width}x{height} @ {fps:.1f} FPS")
            
            # 更新按钮状态
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
            
            # 启动定时器
            self.timer.start(30)  # 约30 FPS
            
            logger.info(f"摄像头 {device_id} 连接成功")
            
        except Exception as e:
            logger.error(f"连接摄像头失败: {e}")
            QMessageBox.critical(self, "错误", f"连接失败: {str(e)}")
    
    def disconnect_camera(self):
        """断开摄像头"""
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
        
        # 清除显示
        self.video_label.clear()
        self.video_label.setText("摄像头已断开")
        
        # 更新按钮状态
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        
        self.info_label.setText("未连接")
        logger.info("摄像头已断开")
    
    def update_frame(self):
        """更新帧显示"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_frame(frame)
    
    def display_frame(self, frame):
        """显示帧"""
        if frame is None:
            return
        
        # 转换颜色空间 BGR -> RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 获取图像尺寸
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # 创建QImage
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 缩放图像以适应标签
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def capture_image(self):
        """捕获图像"""
        if self.current_frame is not None:
            # 保存图像
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_capture_{timestamp}.jpg"
            
            cv2.imwrite(filename, self.current_frame)
            logger.info(f"图像已保存: {filename}")
            
            QMessageBox.information(self, "成功", f"图像已保存到:\n{filename}")
        else:
            QMessageBox.warning(self, "警告", "没有可保存的图像")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.disconnect_camera()
        super().closeEvent(event)

class TestCameraWindow(QMainWindow):
    """测试窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摄像头测试程序")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # 添加摄像头测试组件
        self.camera_widget = SimpleCameraWidget()
        layout.addWidget(self.camera_widget)
        
        # 添加信息面板
        info_group = QGroupBox("测试信息")
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            "这是一个简单的摄像头测试程序，用于验证摄像头基本功能。\n"
            "1. 选择摄像头设备\n"
            "2. 点击'连接'按钮\n"
            "3. 如果连接成功，视频将显示在上方\n"
            "4. 点击'拍照'可以保存当前帧\n"
            "5. 点击'断开'可以断开摄像头连接"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        logger.info("摄像头测试窗口创建完成")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("摄像头测试")
    
    window = TestCameraWindow()
    window.show()
    
    logger.info("摄像头测试程序启动")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())