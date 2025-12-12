"""
工业视觉检测系统 - 主窗口
主窗口负责集成各个组件和整体布局
"""
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QStatusBar, QMenuBar,
    QMenu, QMessageBox, QToolBar, QLabel, QFrame,
    QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSettings, QSize
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QFont
from UI.camera_widget import CameraWidget
from UI.model_widget import ModelWidget
from UI.analysis_widget import AnalysisWidget
from UI.styles import get_style_sheet
from core.camera_manager import CameraManager
from core.model_manager import ModelManager
from core.data_manager import DataManager

class MainWindow(QMainWindow):
    """主窗口类"""
    
    # 信号定义
    camera_connected = pyqtSignal(str)
    camera_disconnected = pyqtSignal()
    model_loaded = pyqtSignal(str)
    inference_started = pyqtSignal()
    inference_stopped = pyqtSignal()
    
    def __init__(self, settings):
        """初始化主窗口
        
        Args:
            settings: 配置管理器
        """
        super().__init__()
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # 初始化管理器
        self.camera_manager = CameraManager()
        self.model_manager = ModelManager()
        self.data_manager = DataManager()
        
        # 初始化UI组件
        self.init_ui()
        
        # 连接信号与槽
        self.connect_signals()
        
        # 加载设置
        self.load_settings()
        
        self.logger.info("主窗口初始化完成")

    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口属性
        self.setWindowTitle("工业视觉检测系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置样式
        self.setStyleSheet(get_style_sheet())
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建工具栏
        self.create_tool_bar()
        
        # 创建主内容区域
        self.create_main_content(main_layout)
        
        # 创建状态栏
        self.create_status_bar()

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        # 打开项目
        open_project_action = QAction(QIcon("resources/icons/open.png"), "打开项目(&O)", self)
        open_project_action.setShortcut(QKeySequence("Ctrl+O"))
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        # 保存项目
        save_project_action = QAction(QIcon("resources/icons/save.png"), "保存项目(&S)", self)
        save_project_action.setShortcut(QKeySequence("Ctrl+S"))
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)
        
        file_menu.addSeparator()
        
        # 导入模型
        import_model_action = QAction("导入模型(&I)", self)
        import_model_action.triggered.connect(self.import_model)
        file_menu.addAction(import_model_action)
        
        # 导出数据
        export_data_action = QAction("导出数据(&E)", self)
        export_data_action.triggered.connect(self.export_data)
        file_menu.addAction(export_data_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction(QIcon("resources/icons/exit.png"), "退出(&X)", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        # 显示/隐藏工具栏
        self.toggle_toolbar_action = QAction("显示工具栏", self, checkable=True)
        self.toggle_toolbar_action.setChecked(True)
        self.toggle_toolbar_action.triggered.connect(self.toggle_toolbar)
        view_menu.addAction(self.toggle_toolbar_action)
        
        # 显示/隐藏状态栏
        self.toggle_statusbar_action = QAction("显示状态栏", self, checkable=True)
        self.toggle_statusbar_action.setChecked(True)
        self.toggle_statusbar_action.triggered.connect(self.toggle_statusbar)
        view_menu.addAction(self.toggle_statusbar_action)
        
        view_menu.addSeparator()
        
        # 全屏模式
        fullscreen_action = QAction("全屏(&F)", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # 操作菜单
        operation_menu = menubar.addMenu("操作(&O)")
        
        # 连接摄像头
        connect_camera_action = QAction(QIcon("resources/icons/camera.png"), "连接摄像头(&C)", self)
        connect_camera_action.triggered.connect(self.connect_camera)
        operation_menu.addAction(connect_camera_action)
        
        # 断开摄像头
        disconnect_camera_action = QAction("断开摄像头(&D)", self)
        disconnect_camera_action.triggered.connect(self.disconnect_camera)
        operation_menu.addAction(disconnect_camera_action)
        
        operation_menu.addSeparator()
        
        # 开始检测
        self.start_inference_action = QAction(QIcon("resources/icons/start.png"), "开始检测(&S)", self)
        self.start_inference_action.triggered.connect(self.start_inference)
        operation_menu.addAction(self.start_inference_action)
        
        # 停止检测
        self.stop_inference_action = QAction(QIcon("resources/icons/stop.png"), "停止检测(&T)", self)
        self.stop_inference_action.triggered.connect(self.stop_inference)
        self.stop_inference_action.setEnabled(False)
        operation_menu.addAction(self.stop_inference_action)
        
        operation_menu.addSeparator()
        
        # 单帧捕获
        single_frame_action = QAction("单帧捕获(&F)", self)
        single_frame_action.triggered.connect(self.capture_single_frame)
        operation_menu.addAction(single_frame_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        # 用户手册
        manual_action = QAction("用户手册(&M)", self)
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)
        
        # 关于
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_tool_bar(self):
        """创建工具栏"""
        self.toolbar = self.addToolBar("主工具栏")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(24, 24))
        
        # 摄像头连接
        self.toolbar.addAction(self.findChild(QAction, "连接摄像头"))
        
        # 检测控制
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.findChild(QAction, "开始检测"))
        self.toolbar.addAction(self.findChild(QAction, "停止检测"))
        
        # 文件操作
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.findChild(QAction, "打开项目"))
        self.toolbar.addAction(self.findChild(QAction, "保存项目"))

    def create_main_content(self, parent_layout):
        """创建主内容区域"""
        # 创建水平分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧面板（摄像头视图）
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧面板（控制和分析）
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割器比例
        splitter.setSizes([900, 500])
        
        parent_layout.addWidget(splitter)

    def create_left_panel(self):
        """创建左侧面板（摄像头视图）"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        # 摄像头标题
        camera_label = QLabel("摄像头视图")
        camera_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        left_layout.addWidget(camera_label)
        
        # 摄像头组件
        self.camera_widget = CameraWidget(self.camera_manager)
        left_layout.addWidget(self.camera_widget, 1)  # 拉伸因子为1
        
        # 摄像头信息面板
        self.create_camera_info_panel(left_layout)
        
        return left_widget

    def create_camera_info_panel(self, parent_layout):
        """创建摄像头信息面板"""
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 5, 10, 5)
        
        # 摄像头状态
        self.camera_status_label = QLabel("状态: 未连接")
        self.camera_status_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.camera_status_label)
        
        # 帧率信息
        self.fps_label = QLabel("FPS: 0")
        info_layout.addWidget(self.fps_label)
        
        # 分辨率信息
        self.resolution_label = QLabel("分辨率: N/A")
        info_layout.addWidget(self.resolution_label)
        
        info_layout.addStretch()
        
        parent_layout.addWidget(info_frame)

    def create_right_panel(self):
        """创建右侧面板（控制和分析）"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # 模型选项卡
        self.model_widget = ModelWidget(self.model_manager)
        tab_widget.addTab(self.model_widget, "模型配置")
        
        # 分析选项卡
        self.analysis_widget = AnalysisWidget(self.data_manager)
        tab_widget.addTab(self.analysis_widget, "数据分析")
        
        # 设置选项卡
        self.settings_widget = self.create_settings_widget()
        tab_widget.addTab(self.settings_widget, "系统设置")
        
        right_layout.addWidget(tab_widget)
        
        return right_widget

    def create_settings_widget(self):
        """创建设置选项卡"""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # 这里可以添加各种设置控件
        # 目前先放一个占位标签
        placeholder_label = QLabel("系统设置")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        settings_layout.addWidget(placeholder_label)
        
        settings_layout.addStretch()
        
        return settings_widget

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 添加状态标签
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        # 添加系统信息
        self.status_bar.addPermanentWidget(QLabel("工业视觉检测系统 v1.0"))
        
        # 添加连接状态
        self.connection_status = QLabel("未连接")
        self.connection_status.setStyleSheet("color: #f00;")
        self.status_bar.addPermanentWidget(self.connection_status)

    def connect_signals(self):
        """连接信号与槽"""
        # 摄像头连接信号
        self.camera_connected.connect(self.on_camera_connected)
        self.camera_disconnected.connect(self.on_camera_disconnected)
        
        # 摄像头管理器的信号
        self.camera_manager.frame_received.connect(self.on_frame_received)
        self.camera_manager.camera_error.connect(self.on_camera_error)
        
        # 模型管理器的信号
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        self.model_manager.inference_completed.connect(self.on_inference_completed)
        
        # 数据管理器的信号
        self.data_manager.data_updated.connect(self.on_data_updated)
        
        # 子组件的信号
        self.camera_widget.camera_selected.connect(self.on_camera_selected)
        self.model_widget.model_selected.connect(self.on_model_selected)

    def load_settings(self):
        """加载窗口设置"""
        settings = QSettings("Industrial Vision", "Industrial Vision System")
        
        # 恢复窗口几何状态
        geometry = settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # 恢复窗口状态
        state = settings.value("window_state")
        if state:
            self.restoreState(state)
        
        # 恢复工具栏和状态栏状态
        show_toolbar = settings.value("show_toolbar", True, type=bool)
        self.toggle_toolbar_action.setChecked(show_toolbar)
        self.toolbar.setVisible(show_toolbar)
        
        show_statusbar = settings.value("show_statusbar", True, type=bool)
        self.toggle_statusbar_action.setChecked(show_statusbar)
        self.status_bar.setVisible(show_statusbar)

    def save_settings(self):
        """保存窗口设置"""
        settings = QSettings("Industrial Vision", "Industrial Vision System")
        
        # 保存窗口几何状态
        settings.setValue("window_geometry", self.saveGeometry())
        settings.setValue("window_state", self.saveState())
        settings.setValue("show_toolbar", self.toolbar.isVisible())
        settings.setValue("show_statusbar", self.status_bar.isVisible())

    # ====================== 事件处理函数 ======================
    
    def on_camera_connected(self, camera_name):
        """摄像头连接成功处理"""
        self.camera_status_label.setText(f"状态: 已连接 ({camera_name})")
        self.camera_status_label.setStyleSheet("color: #0a0;")
        self.connection_status.setText("已连接")
        self.connection_status.setStyleSheet("color: #0a0;")
        self.status_label.setText(f"摄像头 {camera_name} 连接成功")
        
        # 更新摄像头信息
        camera_info = self.camera_manager.get_camera_info()
        if camera_info:
            self.resolution_label.setText(f"分辨率: {camera_info.get('width', 'N/A')}x{camera_info.get('height', 'N/A')}")

    def on_camera_disconnected(self):
        """摄像头断开连接处理"""
        self.camera_status_label.setText("状态: 未连接")
        self.camera_status_label.setStyleSheet("color: #888;")
        self.connection_status.setText("未连接")
        self.connection_status.setStyleSheet("color: #f00;")
        self.status_label.setText("摄像头已断开")
        self.resolution_label.setText("分辨率: N/A")
        self.fps_label.setText("FPS: 0")

    def on_frame_received(self, frame_info):
        """接收到新帧处理"""
        # 更新FPS显示
        fps = frame_info.get('fps', 0)
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_camera_error(self, error_message):
        """摄像头错误处理"""
        self.status_label.setText(f"摄像头错误: {error_message}")
        QMessageBox.warning(self, "摄像头错误", error_message)

    def on_model_loaded(self, model_name):
        """模型加载成功处理"""
        self.status_label.setText(f"模型 {model_name} 加载成功")
        self.model_loaded.emit(model_name)

    def on_inference_completed(self, result):
        """推理完成处理"""
        # 更新分析部件
        self.analysis_widget.update_results(result)
        
        # 保存数据
        self.data_manager.save_inference_result(result)

    def on_data_updated(self, data_info):
        """数据更新处理"""
        self.status_label.setText(f"数据已更新: {data_info}")

    def on_camera_selected(self, camera_index):
        """摄像头选择处理"""
        self.camera_manager.select_camera(camera_index)
        self.status_label.setText(f"选择摄像头 {camera_index}")

    def on_model_selected(self, model_path):
        """模型选择处理"""
        self.model_manager.load_model(model_path)

    # ====================== 菜单功能实现 ======================
    
    def open_project(self):
        """打开项目"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开项目", "", "项目文件 (*.ivproj)"
        )
        if file_path:
            self.status_label.setText(f"打开项目: {file_path}")
            # TODO: 实现项目加载逻辑

    def save_project(self):
        """保存项目"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "项目文件 (*.ivproj)"
        )
        if file_path:
            self.status_label.setText(f"保存项目: {file_path}")
            # TODO: 实现项目保存逻辑

    def import_model(self):
        """导入模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入模型", "", "模型文件 (*.onnx *.pt *.pb);;所有文件 (*.*)"
        )
        if file_path:
            self.status_label.setText(f"导入模型: {file_path}")
            self.model_manager.load_model(file_path)

    def export_data(self):
        """导出数据"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出数据", "", "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*.*)"
        )
        if file_path:
            self.status_label.setText(f"导出数据: {file_path}")
            self.data_manager.export_data(file_path)

    def connect_camera(self):
        """连接摄像头"""
        # 获取可用摄像头列表
        cameras = self.camera_manager.list_available_cameras()
        
        if not cameras:
            QMessageBox.warning(self, "警告", "未检测到可用摄像头")
            return
        
        # 选择摄像头（这里简单选择第一个）
        if self.camera_manager.connect_camera(0):
            self.camera_connected.emit(f"摄像头 {0}")
        else:
            QMessageBox.critical(self, "错误", "摄像头连接失败")

    def disconnect_camera(self):
        """断开摄像头"""
        if self.camera_manager.disconnect_camera():
            self.camera_disconnected.emit()
        else:
            QMessageBox.warning(self, "警告", "摄像头断开失败")

    def start_inference(self):
        """开始检测"""
        if not self.camera_manager.is_connected():
            QMessageBox.warning(self, "警告", "请先连接摄像头")
            return
        
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        # 启动推理
        if self.model_manager.start_inference(self.camera_manager):
            self.start_inference_action.setEnabled(False)
            self.stop_inference_action.setEnabled(True)
            self.inference_started.emit()
            self.status_label.setText("检测已开始")
        else:
            QMessageBox.critical(self, "错误", "检测启动失败")

    def stop_inference(self):
        """停止检测"""
        self.model_manager.stop_inference()
        self.start_inference_action.setEnabled(True)
        self.stop_inference_action.setEnabled(False)
        self.inference_stopped.emit()
        self.status_label.setText("检测已停止")

    def capture_single_frame(self):
        """单帧捕获"""
        if not self.camera_manager.is_connected():
            QMessageBox.warning(self, "警告", "请先连接摄像头")
            return
        
        frame = self.camera_manager.capture_frame()
        if frame is not None:
            self.status_label.setText("单帧捕获成功")
            # TODO: 显示捕获的帧

    def toggle_toolbar(self, checked):
        """切换工具栏显示"""
        self.toolbar.setVisible(checked)

    def toggle_statusbar(self, checked):
        """切换状态栏显示"""
        self.status_bar.setVisible(checked)

    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_manual(self):
        """显示用户手册"""
        QMessageBox.information(self, "用户手册", "工业视觉检测系统用户手册\n\n版本: 1.0\n\n功能说明:\n1. 连接摄像头\n2. 加载模型\n3. 开始检测\n4. 查看分析结果")

    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h3>工业视觉检测系统</h3>
        <p>版本: 1.0</p>
        <p>基于PyQt6的工业流水线视觉检测系统</p>
        <p>功能特点:</p>
        <ul>
            <li>多摄像头支持</li>
            <li>多种深度学习模型</li>
            <li>实时数据分析</li>
            <li>结果导出功能</li>
        </ul>
        <p>© 2024 工业视觉检测系统</p>
        """
        QMessageBox.about(self, "关于", about_text)

    # ====================== 窗口事件重写 ======================
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止所有运行中的任务
        if self.model_manager.is_inference_running():
            self.model_manager.stop_inference()
        
        # 断开摄像头
        if self.camera_manager.is_connected():
            self.camera_manager.disconnect_camera()
        
        # 保存设置
        self.save_settings()
        
        # 确认关闭
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出工业视觉检测系统吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
            self.logger.info("应用程序正常退出")
        else:
            event.ignore()

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # 按ESC退出全屏
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        self.status_label.setText("系统已就绪")
        self.logger.info("主窗口显示完成")