"""
工业视觉检测系统 - 模型组件
负责模型管理、配置和推理设置
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QFrame, QFileDialog, QMessageBox,
    QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QSplitter, QProgressBar,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QToolButton, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QTimer, QSize, 
    QThread, QModelIndex, QItemSelectionModel
)
from PyQt6.QtGui import (
    QFont, QColor, QIcon, QPixmap, QAction
)

class ModelInfoWidget(QGroupBox):
    """模型信息显示组件"""
    
    def __init__(self):
        """初始化模型信息组件"""
        super().__init__("模型信息")
        
        self.init_ui()
        self.current_model_info = {}
    
    def init_ui(self):
        """初始化UI"""
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # 模型名称
        self.name_label = QLabel("未加载")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addRow("名称:", self.name_label)
        
        # 模型类型
        self.type_label = QLabel("N/A")
        layout.addRow("类型:", self.type_label)
        
        # 模型大小
        self.size_label = QLabel("N/A")
        layout.addRow("大小:", self.size_label)
        
        # 输入尺寸
        self.input_size_label = QLabel("N/A")
        layout.addRow("输入尺寸:", self.input_size_label)
        
        # 输出维度
        self.output_dims_label = QLabel("N/A")
        layout.addRow("输出维度:", self.output_dims_label)
        
        # 精度
        self.precision_label = QLabel("N/A")
        layout.addRow("精度:", self.precision_label)
        
        # 框架
        self.framework_label = QLabel("N/A")
        layout.addRow("框架:", self.framework_label)
        
        # 创建时间
        self.created_label = QLabel("N/A")
        layout.addRow("创建时间:", self.created_label)
        
        # 修改时间
        self.modified_label = QLabel("N/A")
        layout.addRow("修改时间:", self.modified_label)
        
        # 描述
        self.description_label = QLabel("无描述")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #666;")
        layout.addRow("描述:", self.description_label)
        
        self.setLayout(layout)
    
    def update_info(self, model_info):
        """更新模型信息"""
        self.current_model_info = model_info or {}
        
        # 更新显示
        self.name_label.setText(self.current_model_info.get('name', '未加载'))
        self.type_label.setText(self.current_model_info.get('type', 'N/A'))
        self.size_label.setText(self.current_model_info.get('size', 'N/A'))
        self.input_size_label.setText(self.current_model_info.get('input_size', 'N/A'))
        self.output_dims_label.setText(str(self.current_model_info.get('output_dims', 'N/A')))
        self.precision_label.setText(self.current_model_info.get('precision', 'N/A'))
        self.framework_label.setText(self.current_model_info.get('framework', 'N/A'))
        self.created_label.setText(self.current_model_info.get('created', 'N/A'))
        self.modified_label.setText(self.current_model_info.get('modified', 'N/A'))
        
        description = self.current_model_info.get('description', '')
        if description:
            self.description_label.setText(description)
        else:
            self.description_label.setText("无描述")
    
    def clear_info(self):
        """清除模型信息"""
        self.current_model_info = {}
        self.name_label.setText("未加载")
        self.type_label.setText("N/A")
        self.size_label.setText("N/A")
        self.input_size_label.setText("N/A")
        self.output_dims_label.setText("N/A")
        self.precision_label.setText("N/A")
        self.framework_label.setText("N/A")
        self.created_label.setText("N/A")
        self.modified_label.setText("N/A")
        self.description_label.setText("无描述")

class ModelPerformanceWidget(QGroupBox):
    """模型性能监控组件"""
    
    def __init__(self):
        """初始化性能监控组件"""
        super().__init__("性能监控")
        
        self.performance_data = {
            'inference_time': [],
            'memory_usage': [],
            'fps': [],
            'accuracy': [],
            'last_update': None
        }
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QGridLayout()
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(10)
        
        # 推理时间
        layout.addWidget(QLabel("推理时间:"), 0, 0)
        self.inference_time_label = QLabel("0.0 ms")
        self.inference_time_label.setStyleSheet("font-weight: bold; color: #0078D7;")
        layout.addWidget(self.inference_time_label, 0, 1)
        
        # 内存使用
        layout.addWidget(QLabel("内存使用:"), 0, 2)
        self.memory_label = QLabel("0.0 MB")
        self.memory_label.setStyleSheet("font-weight: bold; color: #0078D7;")
        layout.addWidget(self.memory_label, 0, 3)
        
        # 推理速度
        layout.addWidget(QLabel("推理速度:"), 1, 0)
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setStyleSheet("font-weight: bold; color: #0078D7;")
        layout.addWidget(self.fps_label, 1, 1)
        
        # 准确率
        layout.addWidget(QLabel("准确率:"), 1, 2)
        self.accuracy_label = QLabel("0.0 %")
        self.accuracy_label.setStyleSheet("font-weight: bold; color: #0078D7;")
        layout.addWidget(self.accuracy_label, 1, 3)
        
        # GPU使用率
        layout.addWidget(QLabel("GPU使用:"), 2, 0)
        self.gpu_label = QLabel("N/A")
        self.gpu_label.setStyleSheet("color: #888;")
        layout.addWidget(self.gpu_label, 2, 1)
        
        # 批次大小
        layout.addWidget(QLabel("批次大小:"), 2, 2)
        self.batch_label = QLabel("1")
        self.batch_label.setStyleSheet("color: #888;")
        layout.addWidget(self.batch_label, 2, 3)
        
        # 进度条
        layout.addWidget(QLabel("负载:"), 3, 0)
        self.performance_bar = QProgressBar()
        self.performance_bar.setRange(0, 100)
        self.performance_bar.setValue(0)
        self.performance_bar.setTextVisible(True)
        layout.addWidget(self.performance_bar, 3, 1, 1, 3)
        
        # 历史数据按钮
        self.history_button = QPushButton("查看历史")
        self.history_button.setEnabled(False)
        layout.addWidget(self.history_button, 4, 0, 1, 4)
        
        self.setLayout(layout)
    
    def update_performance(self, performance_data):
        """更新性能数据"""
        # 更新当前数据
        inference_time = performance_data.get('inference_time', 0)
        memory_usage = performance_data.get('memory_usage', 0)
        fps = performance_data.get('fps', 0)
        accuracy = performance_data.get('accuracy', 0)
        gpu_usage = performance_data.get('gpu_usage', 0)
        batch_size = performance_data.get('batch_size', 1)
        
        # 更新显示
        self.inference_time_label.setText(f"{inference_time:.1f} ms")
        self.memory_label.setText(f"{memory_usage:.1f} MB")
        self.fps_label.setText(f"{fps:.1f} FPS")
        self.accuracy_label.setText(f"{accuracy:.1f} %")
        
        if gpu_usage is not None:
            self.gpu_label.setText(f"{gpu_usage:.1f} %")
        else:
            self.gpu_label.setText("N/A")
        
        self.batch_label.setText(str(batch_size))
        
        # 计算负载百分比（基于推理时间和目标FPS）
        target_fps = 30  # 目标FPS
        max_inference_time = 1000 / target_fps if target_fps > 0 else 100
        load_percentage = min(100, (inference_time / max_inference_time) * 100)
        self.performance_bar.setValue(int(load_percentage))
        
        # 存储历史数据
        self.performance_data['inference_time'].append(inference_time)
        self.performance_data['memory_usage'].append(memory_usage)
        self.performance_data['fps'].append(fps)
        self.performance_data['accuracy'].append(accuracy)
        self.performance_data['last_update'] = datetime.now()
        
        # 限制历史数据长度
        max_history = 100
        for key in ['inference_time', 'memory_usage', 'fps', 'accuracy']:
            if len(self.performance_data[key]) > max_history:
                self.performance_data[key] = self.performance_data[key][-max_history:]
    
    def reset_performance(self):
        """重置性能数据"""
        self.inference_time_label.setText("0.0 ms")
        self.memory_label.setText("0.0 MB")
        self.fps_label.setText("0.0 FPS")
        self.accuracy_label.setText("0.0 %")
        self.gpu_label.setText("N/A")
        self.batch_label.setText("1")
        self.performance_bar.setValue(0)
        
        self.performance_data = {
            'inference_time': [],
            'memory_usage': [],
            'fps': [],
            'accuracy': [],
            'last_update': None
        }

class ModelParametersWidget(QGroupBox):
    """模型参数配置组件"""
    
    # 信号：参数变化
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self):
        """初始化参数配置组件"""
        super().__init__("推理参数")
        
        self.parameters = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QGridLayout()
        layout.setHorizontalSpacing(15)
        layout.setVerticalSpacing(10)
        
        row = 0
        
        # 置信度阈值
        layout.addWidget(QLabel("置信度阈值:"), row, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.confidence_spin, row, 1)
        row += 1
        
        # IOU阈值
        layout.addWidget(QLabel("IOU阈值:"), row, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setDecimals(2)
        self.iou_spin.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.iou_spin, row, 1)
        row += 1
        
        # 批次大小
        layout.addWidget(QLabel("批次大小:"), row, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(1)
        self.batch_spin.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.batch_spin, row, 1)
        row += 1
        
        # 输入尺寸
        layout.addWidget(QLabel("输入宽度:"), row, 0)
        self.input_width_spin = QSpinBox()
        self.input_width_spin.setRange(32, 4096)
        self.input_width_spin.setValue(640)
        self.input_width_spin.setSingleStep(32)
        self.input_width_spin.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.input_width_spin, row, 1)
        
        layout.addWidget(QLabel("输入高度:"), row, 2)
        self.input_height_spin = QSpinBox()
        self.input_height_spin.setRange(32, 4096)
        self.input_height_spin.setValue(640)
        self.input_height_spin.setSingleStep(32)
        self.input_height_spin.valueChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.input_height_spin, row, 3)
        row += 1
        
        # 预处理参数
        layout.addWidget(QLabel("标准化均值 (R,G,B):"), row, 0, 1, 2)
        
        mean_layout = QHBoxLayout()
        self.mean_r_spin = QDoubleSpinBox()
        self.mean_r_spin.setRange(0.0, 1.0)
        self.mean_r_spin.setSingleStep(0.01)
        self.mean_r_spin.setValue(0.0)
        self.mean_r_spin.setDecimals(3)
        self.mean_r_spin.valueChanged.connect(self.on_parameter_changed)
        mean_layout.addWidget(self.mean_r_spin)
        
        self.mean_g_spin = QDoubleSpinBox()
        self.mean_g_spin.setRange(0.0, 1.0)
        self.mean_g_spin.setSingleStep(0.01)
        self.mean_g_spin.setValue(0.0)
        self.mean_g_spin.setDecimals(3)
        self.mean_g_spin.valueChanged.connect(self.on_parameter_changed)
        mean_layout.addWidget(self.mean_g_spin)
        
        self.mean_b_spin = QDoubleSpinBox()
        self.mean_b_spin.setRange(0.0, 1.0)
        self.mean_b_spin.setSingleStep(0.01)
        self.mean_b_spin.setValue(0.0)
        self.mean_b_spin.setDecimals(3)
        self.mean_b_spin.valueChanged.connect(self.on_parameter_changed)
        mean_layout.addWidget(self.mean_b_spin)
        
        layout.addLayout(mean_layout, row, 2, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("标准化标准差 (R,G,B):"), row, 0, 1, 2)
        
        std_layout = QHBoxLayout()
        self.std_r_spin = QDoubleSpinBox()
        self.std_r_spin.setRange(0.0, 1.0)
        self.std_r_spin.setSingleStep(0.01)
        self.std_r_spin.setValue(1.0)
        self.std_r_spin.setDecimals(3)
        self.std_r_spin.valueChanged.connect(self.on_parameter_changed)
        std_layout.addWidget(self.std_r_spin)
        
        self.std_g_spin = QDoubleSpinBox()
        self.std_g_spin.setRange(0.0, 1.0)
        self.std_g_spin.setSingleStep(0.01)
        self.std_g_spin.setValue(1.0)
        self.std_g_spin.setDecimals(3)
        self.std_g_spin.valueChanged.connect(self.on_parameter_changed)
        std_layout.addWidget(self.std_g_spin)
        
        self.std_b_spin = QDoubleSpinBox()
        self.std_b_spin.setRange(0.0, 1.0)
        self.std_b_spin.setSingleStep(0.01)
        self.std_b_spin.setValue(1.0)
        self.std_b_spin.setDecimals(3)
        self.std_b_spin.valueChanged.connect(self.on_parameter_changed)
        std_layout.addWidget(self.std_b_spin)
        
        layout.addLayout(std_layout, row, 2, 1, 2)
        row += 1
        
        # 后处理选项
        self.nms_checkbox = QCheckBox("启用非极大值抑制 (NMS)")
        self.nms_checkbox.setChecked(True)
        self.nms_checkbox.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.nms_checkbox, row, 0, 1, 2)
        
        self.augment_checkbox = QCheckBox("启用测试时增强 (TTA)")
        self.augment_checkbox.setChecked(False)
        self.augment_checkbox.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.augment_checkbox, row, 2, 1, 2)
        row += 1
        
        # GPU加速
        self.gpu_checkbox = QCheckBox("启用GPU加速")
        self.gpu_checkbox.setChecked(True)
        self.gpu_checkbox.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.gpu_checkbox, row, 0, 1, 2)
        
        # 半精度推理
        self.half_checkbox = QCheckBox("启用半精度推理 (FP16)")
        self.half_checkbox.setChecked(False)
        self.half_checkbox.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.half_checkbox, row, 2, 1, 2)
        row += 1
        
        # 重置按钮
        self.reset_button = QPushButton("重置参数")
        self.reset_button.clicked.connect(self.reset_parameters)
        layout.addWidget(self.reset_button, row, 0, 1, 4)
        
        self.setLayout(layout)
        self.update_parameters()
    
    def on_parameter_changed(self):
        """参数变化处理"""
        self.update_parameters()
        self.parameters_changed.emit(self.parameters)
    
    def update_parameters(self):
        """更新参数字典"""
        self.parameters = {
            'confidence_threshold': self.confidence_spin.value(),
            'iou_threshold': self.iou_spin.value(),
            'batch_size': self.batch_spin.value(),
            'input_width': self.input_width_spin.value(),
            'input_height': self.input_height_spin.value(),
            'normalize_mean': [
                self.mean_r_spin.value(),
                self.mean_g_spin.value(),
                self.mean_b_spin.value()
            ],
            'normalize_std': [
                self.std_r_spin.value(),
                self.std_g_spin.value(),
                self.std_b_spin.value()
            ],
            'enable_nms': self.nms_checkbox.isChecked(),
            'enable_tta': self.augment_checkbox.isChecked(),
            'use_gpu': self.gpu_checkbox.isChecked(),
            'use_half': self.half_checkbox.isChecked()
        }
    
    def reset_parameters(self):
        """重置参数为默认值"""
        self.confidence_spin.setValue(0.5)
        self.iou_spin.setValue(0.45)
        self.batch_spin.setValue(1)
        self.input_width_spin.setValue(640)
        self.input_height_spin.setValue(640)
        
        self.mean_r_spin.setValue(0.0)
        self.mean_g_spin.setValue(0.0)
        self.mean_b_spin.setValue(0.0)
        
        self.std_r_spin.setValue(1.0)
        self.std_g_spin.setValue(1.0)
        self.std_b_spin.setValue(1.0)
        
        self.nms_checkbox.setChecked(True)
        self.augment_checkbox.setChecked(False)
        self.gpu_checkbox.setChecked(True)
        self.half_checkbox.setChecked(False)
        
        self.update_parameters()
        self.parameters_changed.emit(self.parameters)
    
    def set_parameters(self, parameters):
        """设置参数值"""
        if not parameters:
            return
        
        if 'confidence_threshold' in parameters:
            self.confidence_spin.setValue(parameters['confidence_threshold'])
        
        if 'iou_threshold' in parameters:
            self.iou_spin.setValue(parameters['iou_threshold'])
        
        if 'batch_size' in parameters:
            self.batch_spin.setValue(parameters['batch_size'])
        
        if 'input_width' in parameters:
            self.input_width_spin.setValue(parameters['input_width'])
        
        if 'input_height' in parameters:
            self.input_height_spin.setValue(parameters['input_height'])
        
        if 'normalize_mean' in parameters and len(parameters['normalize_mean']) >= 3:
            self.mean_r_spin.setValue(parameters['normalize_mean'][0])
            self.mean_g_spin.setValue(parameters['normalize_mean'][1])
            self.mean_b_spin.setValue(parameters['normalize_mean'][2])
        
        if 'normalize_std' in parameters and len(parameters['normalize_std']) >= 3:
            self.std_r_spin.setValue(parameters['normalize_std'][0])
            self.std_g_spin.setValue(parameters['normalize_std'][1])
            self.std_b_spin.setValue(parameters['normalize_std'][2])
        
        if 'enable_nms' in parameters:
            self.nms_checkbox.setChecked(parameters['enable_nms'])
        
        if 'enable_tta' in parameters:
            self.augment_checkbox.setChecked(parameters['enable_tta'])
        
        if 'use_gpu' in parameters:
            self.gpu_checkbox.setChecked(parameters['use_gpu'])
        
        if 'use_half' in parameters:
            self.half_checkbox.setChecked(parameters['use_half'])
        
        self.update_parameters()

class ModelListWidget(QGroupBox):
    """模型列表组件"""
    
    # 信号：模型选择变化
    model_selected = pyqtSignal(str)
    model_double_clicked = pyqtSignal(str)
    
    def __init__(self, model_manager):
        """初始化模型列表组件
        
        Args:
            model_manager: 模型管理器实例
        """
        super().__init__("模型列表")
        
        self.model_manager = model_manager
        self.models = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 搜索框
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索模型...")
        self.search_input.textChanged.connect(self.filter_models)
        search_layout.addWidget(self.search_input)
        
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setFixedWidth(60)
        self.refresh_button.clicked.connect(self.refresh_models)
        search_layout.addWidget(self.refresh_button)
        
        layout.addLayout(search_layout)
        
        # 模型列表
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(["名称", "类型", "大小", "状态"])
        self.model_tree.setColumnWidth(0, 200)
        self.model_tree.setColumnWidth(1, 100)
        self.model_tree.setColumnWidth(2, 80)
        self.model_tree.setColumnWidth(3, 60)
        
        self.model_tree.itemSelectionChanged.connect(self.on_selection_changed)
        self.model_tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        layout.addWidget(self.model_tree)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("添加模型")
        self.add_button.clicked.connect(self.add_model)
        button_layout.addWidget(self.add_button)
        
        self.remove_button = QPushButton("移除模型")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self.remove_model)
        button_layout.addWidget(self.remove_button)
        
        self.test_button = QPushButton("测试模型")
        self.test_button.setEnabled(False)
        self.test_button.clicked.connect(self.test_model)
        button_layout.addWidget(self.test_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 加载模型列表
        self.refresh_models()
    
    def refresh_models(self):
        """刷新模型列表"""
        """刷新模型列表"""
        self.model_tree.clear()
        models_data = self.model_manager.get_available_models()
    
        # 检查返回的数据类型
        if isinstance(models_data, list):
            # 如果是列表，转换为字典，使用索引作为键
            self.models = {}
            for idx, model_info in enumerate(models_data):
                model_id = model_info.get('id', f"model_{idx}")
                self.models[model_id] = model_info
        elif isinstance(models_data, dict):
            self.models = models_data
        else:
            self.models = {}
            logging.warning(f"模型管理器返回了未知的数据类型: {type(models_data)}")
            return
    
        # 按类别分组
        categories = {}
        for model_id, model_info in self.models.items():
            category = model_info.get('category', '未分类')
            if category not in categories:
                categories[category] = []
            categories[category].append((model_id, model_info))
        
        # 添加模型到树形列表
        for category, model_list in categories.items():
            category_item = QTreeWidgetItem(self.model_tree, [category])
            category_item.setExpanded(True)
            
            for model_id, model_info in model_list:
                model_item = QTreeWidgetItem(category_item)
                
                # 名称
                model_item.setText(0, model_info.get('name', model_id))
                model_item.setData(0, Qt.ItemDataRole.UserRole, model_id)
                
                # 类型
                model_type = model_info.get('type', 'N/A')
                model_item.setText(1, model_type)
                
                # 大小
                size = model_info.get('size', 'N/A')
                model_item.setText(2, size)
                
                # 状态
                status = "已加载" if model_info.get('loaded', False) else "未加载"
                model_item.setText(3, status)
                
                if model_info.get('loaded', False):
                    model_item.setForeground(3, QColor(0, 128, 0))  # 绿色
                else:
                    model_item.setForeground(3, QColor(128, 128, 128))  # 灰色
        
        self.model_tree.expandAll()
    
    def filter_models(self, text):
        """过滤模型列表"""
        for i in range(self.model_tree.topLevelItemCount()):
            category_item = self.model_tree.topLevelItem(i)
            category_visible = False
            
            for j in range(category_item.childCount()):
                model_item = category_item.child(j)
                model_name = model_item.text(0).lower()
                
                if text.lower() in model_name:
                    model_item.setHidden(False)
                    category_visible = True
                else:
                    model_item.setHidden(True)
            
            category_item.setHidden(not category_visible)
    
    def on_selection_changed(self):
        """选择变化处理"""
        selected_items = self.model_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            model_id = item.data(0, Qt.ItemDataRole.UserRole)
            if model_id:
                self.model_selected.emit(model_id)
                self.remove_button.setEnabled(True)
                self.test_button.setEnabled(True)
                return
        
        self.remove_button.setEnabled(False)
        self.test_button.setEnabled(False)
    
    def on_item_double_clicked(self, item, column):
        """双击项目处理"""
        model_id = item.data(0, Qt.ItemDataRole.UserRole)
        if model_id:
            self.model_double_clicked.emit(model_id)
    
    def add_model(self):
        """添加模型"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择模型文件", "",
            "模型文件 (*.onnx *.pt *.pb *.tflite);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 添加到模型管理器
                success = self.model_manager.add_model(file_path)
                if success:
                    self.refresh_models()
                    QMessageBox.information(self, "成功", "模型添加成功")
                else:
                    QMessageBox.warning(self, "警告", "模型添加失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"添加模型时出错: {str(e)}")
    
    def remove_model(self):
        """移除模型"""
        selected_items = self.model_tree.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        model_id = item.data(0, Qt.ItemDataRole.UserRole)
        
        if not model_id:
            return
        
        # 确认对话框
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除模型 '{item.text(0)}' 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                success = self.model_manager.remove_model(model_id)
                if success:
                    self.refresh_models()
                    QMessageBox.information(self, "成功", "模型删除成功")
                else:
                    QMessageBox.warning(self, "警告", "模型删除失败")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除模型时出错: {str(e)}")
    
    def test_model(self):
        """测试模型"""
        selected_items = self.model_tree.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        model_id = item.data(0, Qt.ItemDataRole.UserRole)
        
        if not model_id:
            return
        
        # 发出测试信号
        # 主组件会处理测试逻辑
        QMessageBox.information(self, "测试", f"开始测试模型: {item.text(0)}")
        # TODO: 实现实际测试逻辑

class ModelWidget(QWidget):
    """模型主组件"""
    
    # 信号
    model_selected = pyqtSignal(str)
    inference_parameters_changed = pyqtSignal(dict)
    model_test_requested = pyqtSignal(str)
    
    def __init__(self, model_manager):
        """初始化模型组件
        
        Args:
            model_manager: 模型管理器实例
        """
        super().__init__()
        
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 加载模型列表
        self.refresh_model_info()
        
        self.logger.info("模型组件初始化完成")
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # 创建主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：模型列表和配置
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # 模型列表
        self.model_list_widget = ModelListWidget(self.model_manager)
        left_layout.addWidget(self.model_list_widget, 2)  # 2/3高度
        
        # 参数配置
        self.parameters_widget = ModelParametersWidget()
        left_layout.addWidget(self.parameters_widget, 1)  # 1/3高度
        
        splitter.addWidget(left_widget)
        
        # 右侧：模型信息和性能
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # 模型信息
        self.info_widget = ModelInfoWidget()
        right_layout.addWidget(self.info_widget)
        
        # 性能监控
        self.performance_widget = ModelPerformanceWidget()
        right_layout.addWidget(self.performance_widget)
        
        # 测试面板
        self.create_test_panel(right_layout)
        
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def create_test_panel(self, parent_layout):
        """创建测试面板"""
        test_group = QGroupBox("模型测试")
        test_layout = QVBoxLayout()
        
        # 测试图像选择
        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("测试图像:"))
        
        self.test_image_input = QLineEdit()
        self.test_image_input.setPlaceholderText("选择测试图像...")
        self.test_image_input.setReadOnly(True)
        image_layout.addWidget(self.test_image_input)
        
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_test_image)
        image_layout.addWidget(self.browse_button)
        
        test_layout.addLayout(image_layout)
        
        # 测试按钮
        button_layout = QHBoxLayout()
        
        self.test_single_button = QPushButton("单张测试")
        self.test_single_button.setEnabled(False)
        self.test_single_button.clicked.connect(self.test_single_image)
        button_layout.addWidget(self.test_single_button)
        
        self.test_batch_button = QPushButton("批量测试")
        self.test_batch_button.setEnabled(False)
        self.test_batch_button.clicked.connect(self.test_batch_images)
        button_layout.addWidget(self.test_batch_button)
        
        test_layout.addLayout(button_layout)
        
        # 测试结果
        self.test_result_text = QTextEdit()
        self.test_result_text.setReadOnly(True)
        self.test_result_text.setMaximumHeight(150)
        self.test_result_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                font-family: monospace;
            }
        """)
        test_layout.addWidget(self.test_result_text)
        
        test_group.setLayout(test_layout)
        parent_layout.addWidget(test_group)
    
    def connect_signals(self):
        """连接信号"""
        # 模型列表信号
        self.model_list_widget.model_selected.connect(self.on_model_selected)
        self.model_list_widget.model_double_clicked.connect(self.on_model_double_clicked)
        
        # 参数配置信号
        self.parameters_widget.parameters_changed.connect(self.on_parameters_changed)
        
        # # 模型管理器信号
        # if hasattr(self.model_manager, 'model_loaded'):
        #     self.model_manager.model_loaded.connect(self.on_model_loaded)
        
        # if hasattr(self.model_manager, 'inference_completed'):
        #     self.model_manager.inference_completed.connect(self.on_inference_completed)
        
        # if hasattr(self.model_manager, 'performance_updated'):
        #     # 修改为使用正确的信号签名
        #     self.model_manager.performance_updated.connect(self.on_performance_updated)
    
    def refresh_model_info(self):
        """刷新模型信息"""
        # 获取当前加载的模型信息
        current_model = self.model_manager.get_current_model()
        if current_model:
            self.info_widget.update_info(current_model)
        else:
            self.info_widget.clear_info()
    
    @pyqtSlot(str)
    def on_model_selected(self, model_id):
        """模型选择处理"""
        self.model_selected.emit(model_id)
        
        # 获取模型详细信息
        model_info = self.model_manager.get_model_info(model_id)
        if model_info:
            self.info_widget.update_info(model_info)
            
            # 加载模型参数配置
            if 'parameters' in model_info:
                self.parameters_widget.set_parameters(model_info['parameters'])
    
    @pyqtSlot(str)
    def on_model_double_clicked(self, model_id):
        """模型双击处理"""
        # 加载模型
        try:
            success = self.model_manager.load_model_by_id(model_id)
            if success:
                QMessageBox.information(self, "成功", "模型加载成功")
                self.refresh_model_info()
            else:
                QMessageBox.warning(self, "警告", "模型加载失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载错误: {str(e)}")
    
    @pyqtSlot(str)
    def on_model_loaded(self, model_name):
        """模型加载完成处理"""
        self.refresh_model_info()
        self.model_list_widget.refresh_models()
        
        # 启用测试按钮
        self.test_single_button.setEnabled(True)
        self.test_batch_button.setEnabled(True)
    
    @pyqtSlot(dict)
    def on_inference_completed(self, result):
        """推理完成处理"""
        # 更新性能数据
        if 'performance' in result:
            self.performance_widget.update_performance(result['performance'])
    
    @pyqtSlot(dict)
    def on_performance_updated(self, event_type, performance_data):
        """性能更新处理 - 新版本"""
        # event_type 可能是事件类型，如 'inference', 'training' 等
        # performance_data 是性能数据对象
        
        if isinstance(performance_data, dict):
            self.performance_widget.update_performance(performance_data)
        else:
            # 如果performance_data不是字典，尝试转换
            try:
                # 假设performance_data有to_dict()方法或可以转换为字典
                if hasattr(performance_data, 'to_dict'):
                    perf_dict = performance_data.to_dict()
                else:
                    # 尝试直接访问属性
                    perf_dict = {
                        'inference_time': getattr(performance_data, 'inference_time', 0),
                        'memory_usage': getattr(performance_data, 'memory_usage', 0),
                        'fps': getattr(performance_data, 'fps', 0),
                        'accuracy': getattr(performance_data, 'accuracy', 0),
                        'gpu_usage': getattr(performance_data, 'gpu_usage', 0),
                        'batch_size': getattr(performance_data, 'batch_size', 1)
                    }
                self.performance_widget.update_performance(perf_dict)
            except Exception as e:
                self.logger.warning(f"无法处理性能数据: {e}")
    
    @pyqtSlot(dict)
    def on_parameters_changed(self, parameters):
        """参数变化处理"""
        # 更新模型管理器参数
        self.model_manager.set_inference_parameters(parameters)
        
        # 发出信号
        self.inference_parameters_changed.emit(parameters)
        
        self.logger.debug(f"推理参数已更新: {parameters}")
    
    def browse_test_image(self):
        """浏览测试图像"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择测试图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*.*)"
        )
        
        if file_path:
            self.test_image_input.setText(file_path)
    
    def test_single_image(self):
        """单张图像测试"""
        image_path = self.test_image_input.text()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "警告", "请选择有效的测试图像")
            return
        
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        try:
            # 执行推理
            result = self.model_manager.inference_single_image(image_path)
            
            # 显示结果
            if result:
                self.display_test_result(result)
            else:
                QMessageBox.warning(self, "警告", "测试失败，无结果返回")
        
        except Exception as e:
            self.logger.error(f"单张图像测试错误: {e}")
            QMessageBox.critical(self, "错误", f"测试失败: {str(e)}")
    
    def test_batch_images(self):
        """批量图像测试"""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self, "选择批量测试图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*.*)"
        )
        
        if not file_paths:
            return
        
        if not self.model_manager.is_model_loaded():
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        try:
            # 执行批量推理
            results = self.model_manager.inference_batch_images(file_paths)
            
            # 显示汇总结果
            if results:
                self.display_batch_test_result(results, file_paths)
            else:
                QMessageBox.warning(self, "警告", "批量测试失败，无结果返回")
        
        except Exception as e:
            self.logger.error(f"批量图像测试错误: {e}")
            QMessageBox.critical(self, "错误", f"批量测试失败: {str(e)}")
    
    def display_test_result(self, result):
        """显示测试结果"""
        # 清空之前的结果
        self.test_result_text.clear()
        
        # 构建结果文本
        result_text = "=== 单张图像测试结果 ===\n\n"
        
        # 基本结果
        if 'detections' in result:
            detections = result['detections']
            result_text += f"检测到对象: {len(detections)} 个\n"
            
            for i, det in enumerate(detections, 1):
                result_text += f"\n对象 {i}:\n"
                result_text += f"  类别: {det.get('class', 'N/A')}\n"
                result_text += f"  置信度: {det.get('confidence', 0):.3f}\n"
                result_text += f"  位置: [{det.get('x', 0):.1f}, {det.get('y', 0):.1f}, "
                result_text += f"{det.get('width', 0):.1f}, {det.get('height', 0):.1f}]\n"
        
        # 性能信息
        if 'performance' in result:
            perf = result['performance']
            result_text += f"\n性能信息:\n"
            result_text += f"  推理时间: {perf.get('inference_time', 0):.1f} ms\n"
            result_text += f"  内存使用: {perf.get('memory_usage', 0):.1f} MB\n"
            result_text += f"  推理速度: {perf.get('fps', 0):.1f} FPS\n"
        
        # 显示结果
        self.test_result_text.setText(result_text)
    
    def display_batch_test_result(self, results, file_paths):
        """显示批量测试结果"""
        # 清空之前的结果
        self.test_result_text.clear()
        
        # 构建结果文本
        result_text = f"=== 批量测试结果 ({len(file_paths)} 张图像) ===\n\n"
        
        total_detections = 0
        total_inference_time = 0
        
        for i, (file_path, result) in enumerate(zip(file_paths, results), 1):
            filename = os.path.basename(file_path)
            detections = result.get('detections', [])
            performance = result.get('performance', {})
            
            result_text += f"图像 {i}: {filename}\n"
            result_text += f"  检测到对象: {len(detections)} 个\n"
            
            if 'inference_time' in performance:
                total_inference_time += performance['inference_time']
            
            total_detections += len(detections)
        
        # 汇总信息
        result_text += f"\n=== 汇总信息 ===\n"
        result_text += f"总图像数: {len(file_paths)}\n"
        result_text += f"总检测数: {total_detections}\n"
        result_text += f"平均检测数: {total_detections/len(file_paths):.1f} 个/图像\n"
        
        if len(file_paths) > 0:
            avg_inference_time = total_inference_time / len(file_paths)
            result_text += f"平均推理时间: {avg_inference_time:.1f} ms/图像\n"
            result_text += f"处理速度: {1000/avg_inference_time:.1f} FPS\n"
        
        # 显示结果
        self.test_result_text.setText(result_text)
    
    def reset_all(self):
        """重置所有组件"""
        self.info_widget.clear_info()
        self.performance_widget.reset_performance()
        self.test_result_text.clear()
        self.test_image_input.clear()
        self.test_single_button.setEnabled(False)
        self.test_batch_button.setEnabled(False)