"""
工业视觉检测系统 - 分析组件
负责检测结果的显示、统计和可视化分析
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QFrame, QFileDialog, QMessageBox,
    QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QSplitter, QProgressBar,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QToolButton, QSizePolicy, QDateEdit,
    QTimeEdit, QDateTimeEdit, QScrollArea, QProgressDialog,
    QDialog, QDialogButtonBox, QAbstractItemView
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QTimer, QSize, QDateTime,
    QDate, QTime, QModelIndex, QItemSelectionModel,
    QThread, QMutex, QWaitCondition
)
from PyQt6.QtGui import (
    QFont, QColor, QIcon, QPixmap, QAction, QBrush,
    QPen, QPainter, QPainterPath
)
from PyQt6.QtCharts import (
    QChart, QChartView, QLineSeries, QBarSeries,
    QBarSet, QPieSeries, QScatterSeries,
    QValueAxis, QCategoryAxis, QDateTimeAxis,
    QBarCategoryAxis
)

class DetectionResultWidget(QGroupBox):
    """检测结果显示组件"""
    
    # 信号：结果项点击
    result_selected = pyqtSignal(dict)
    
    def __init__(self):
        """初始化检测结果组件"""
        super().__init__("检测结果")
        
        self.results = []
        self.current_index = -1
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["全部", "正常", "缺陷", "警告", "错误"])
        self.filter_combo.currentTextChanged.connect(self.filter_results)
        toolbar_layout.addWidget(QLabel("筛选:"))
        toolbar_layout.addWidget(self.filter_combo)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["时间", "置信度", "类型"])
        self.sort_combo.currentTextChanged.connect(self.sort_results)
        toolbar_layout.addWidget(QLabel("排序:"))
        toolbar_layout.addWidget(self.sort_combo)
        
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.clear_results)
        toolbar_layout.addWidget(self.clear_button)
        
        self.export_button = QPushButton("导出")
        self.export_button.clicked.connect(self.export_results)
        toolbar_layout.addWidget(self.export_button)
        
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels([
            "时间", "类型", "置信度", "位置", "尺寸", "状态", "备注"
        ])
        
        # 设置表格属性
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setSortingEnabled(True)
        
        # 设置列宽
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.result_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        
        self.result_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.result_table.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        layout.addWidget(self.result_table)
        
        # 统计信息
        self.stats_label = QLabel("检测统计: 0 个结果")
        self.stats_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
    
    def add_result(self, result):
        """添加检测结果"""
        # 解析结果
        timestamp = result.get('timestamp', datetime.now())
        result_type = result.get('type', '未知')
        confidence = result.get('confidence', 0.0)
        bbox = result.get('bbox', [0, 0, 0, 0])
        status = result.get('status', '正常')
        note = result.get('note', '')
        
        # 添加到结果列表
        result_item = {
            'timestamp': timestamp,
            'type': result_type,
            'confidence': confidence,
            'bbox': bbox,
            'status': status,
            'note': note,
            'raw_data': result
        }
        
        self.results.append(result_item)
        
        # 更新表格
        self.update_table()
        
        # 自动滚动到最新结果
        if len(self.results) > 0:
            self.result_table.scrollToBottom()
    
    def add_batch_results(self, results):
        """批量添加检测结果"""
        for result in results:
            self.add_result(result)
    
    def update_table(self):
        """更新表格显示"""
        self.result_table.setRowCount(len(self.results))
        
        for i, result in enumerate(self.results):
            # 时间
            timestamp = result['timestamp']
            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime("%H:%M:%S")
                date_str = timestamp.strftime("%Y-%m-%d")
                time_item = QTableWidgetItem(f"{date_str}\n{time_str}")
            else:
                time_item = QTableWidgetItem(str(timestamp))
            self.result_table.setItem(i, 0, time_item)
            
            # 类型
            type_item = QTableWidgetItem(result['type'])
            # 根据类型设置颜色
            if result['type'] == '缺陷':
                type_item.setForeground(QBrush(QColor(220, 0, 0)))
            elif result['type'] == '警告':
                type_item.setForeground(QBrush(QColor(255, 165, 0)))
            elif result['type'] == '正常':
                type_item.setForeground(QBrush(QColor(0, 128, 0)))
            self.result_table.setItem(i, 1, type_item)
            
            # 置信度
            conf_str = f"{result['confidence']:.3f}"
            conf_item = QTableWidgetItem(conf_str)
            # 根据置信度设置颜色
            if result['confidence'] > 0.8:
                conf_item.setForeground(QBrush(QColor(0, 128, 0)))
            elif result['confidence'] > 0.5:
                conf_item.setForeground(QBrush(QColor(255, 165, 0)))
            else:
                conf_item.setForeground(QBrush(QColor(220, 0, 0)))
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.result_table.setItem(i, 2, conf_item)
            
            # 位置
            bbox = result['bbox']
            if len(bbox) >= 4:
                pos_str = f"({bbox[0]:.0f}, {bbox[1]:.0f})"
            else:
                pos_str = "N/A"
            pos_item = QTableWidgetItem(pos_str)
            self.result_table.setItem(i, 3, pos_item)
            
            # 尺寸
            if len(bbox) >= 4:
                size_str = f"{bbox[2]:.0f}×{bbox[3]:.0f}"
            else:
                size_str = "N/A"
            size_item = QTableWidgetItem(size_str)
            self.result_table.setItem(i, 4, size_item)
            
            # 状态
            status_item = QTableWidgetItem(result['status'])
            # 根据状态设置颜色
            if result['status'] == '通过':
                status_item.setForeground(QBrush(QColor(0, 128, 0)))
            elif result['status'] == '失败':
                status_item.setForeground(QBrush(QColor(220, 0, 0)))
            elif result['status'] == '警告':
                status_item.setForeground(QBrush(QColor(255, 165, 0)))
            self.result_table.setItem(i, 5, status_item)
            
            # 备注
            note_item = QTableWidgetItem(result['note'])
            self.result_table.setItem(i, 6, note_item)
        
        # 更新统计信息
        self.update_stats()
    
    def filter_results(self, filter_text):
        """筛选结果"""
        # TODO: 实现筛选逻辑
        pass
    
    def sort_results(self, sort_by):
        """排序结果"""
        if sort_by == "时间":
            self.results.sort(key=lambda x: x['timestamp'], reverse=True)
        elif sort_by == "置信度":
            self.results.sort(key=lambda x: x['confidence'], reverse=True)
        elif sort_by == "类型":
            self.results.sort(key=lambda x: x['type'])
        
        self.update_table()
    
    def clear_results(self):
        """清除所有结果"""
        reply = QMessageBox.question(
            self, "确认清除",
            "确定要清除所有检测结果吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.results.clear()
            self.update_table()
    
    def export_results(self):
        """导出结果"""
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "导出结果", "",
            "CSV文件 (*.csv);;JSON文件 (*.json);;Excel文件 (*.xlsx)"
        )
        
        if not file_path:
            return
        
        try:
            # 准备数据
            export_data = []
            for result in self.results:
                export_result = {
                    'timestamp': result['timestamp'].isoformat() if isinstance(result['timestamp'], datetime) else result['timestamp'],
                    'type': result['type'],
                    'confidence': result['confidence'],
                    'bbox': result['bbox'],
                    'status': result['status'],
                    'note': result['note']
                }
                export_data.append(export_result)
            
            # 根据文件类型导出
            if file_path.endswith('.csv'):
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            elif file_path.endswith('.xlsx'):
                df = pd.DataFrame(export_data)
                df.to_excel(file_path, index=False)
            
            QMessageBox.information(self, "成功", f"结果已导出到: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def update_stats(self):
        """更新统计信息"""
        if not self.results:
            self.stats_label.setText("检测统计: 0 个结果")
            return
        
        # 计算各类别数量
        type_counts = defaultdict(int)
        status_counts = defaultdict(int)
        
        for result in self.results:
            type_counts[result['type']] += 1
            status_counts[result['status']] += 1
        
        # 构建统计文本
        stats_text = f"检测统计: {len(self.results)} 个结果"
        
        if type_counts:
            type_stats = "，".join([f"{k}: {v}" for k, v in type_counts.items()])
            stats_text += f" | 类型: {type_stats}"
        
        if status_counts:
            status_stats = "，".join([f"{k}: {v}" for k, v in status_counts.items()])
            stats_text += f" | 状态: {status_stats}"
        
        self.stats_label.setText(stats_text)
    
    def on_selection_changed(self):
        """选择变化处理"""
        selected_items = self.result_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            if 0 <= row < len(self.results):
                self.current_index = row
                self.result_selected.emit(self.results[row]['raw_data'])
    
    def on_item_double_clicked(self, item):
        """双击项目处理"""
        row = item.row()
        if 0 <= row < len(self.results):
            self.result_selected.emit(self.results[row]['raw_data'])
            # TODO: 可以显示详细信息的弹窗
    
    def get_selected_result(self):
        """获取选中的结果"""
        if 0 <= self.current_index < len(self.results):
            return self.results[self.current_index]
        return None

class StatisticsChartWidget(QGroupBox):
    """统计图表组件"""
    
    def __init__(self):
        """初始化统计图表组件"""
        super().__init__("统计图表")
        
        self.charts = {}
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 图表类型选择
        chart_layout = QHBoxLayout()
        chart_layout.addWidget(QLabel("图表类型:"))
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "缺陷分布", "时间趋势", "置信度分布", 
            "类型统计", "状态比例", "自定义"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.update_chart)
        chart_layout.addWidget(self.chart_type_combo)
        
        # 时间范围选择
        chart_layout.addWidget(QLabel("时间范围:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems([
            "最近1小时", "最近24小时", "最近7天", "最近30天", "全部"
        ])
        self.time_range_combo.currentTextChanged.connect(self.update_chart)
        chart_layout.addWidget(self.time_range_combo)
        
        chart_layout.addStretch()
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.update_chart)
        chart_layout.addWidget(self.refresh_button)
        
        layout.addLayout(chart_layout)
        
        # 图表视图
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 创建默认图表
        self.create_default_chart()
        
        layout.addWidget(self.chart_view)
        
        self.setLayout(layout)
    
    def create_default_chart(self):
        """创建默认图表"""
        chart = QChart()
        chart.setTitle("检测统计")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # 添加示例数据
        series = QBarSeries()
        
        bar_set = QBarSet("缺陷")
        bar_set.append([1, 2, 3, 4, 5])
        bar_set.setColor(QColor(255, 99, 132))
        
        bar_set2 = QBarSet("正常")
        bar_set2.append([5, 4, 3, 2, 1])
        bar_set2.setColor(QColor(54, 162, 235))
        
        series.append(bar_set)
        series.append(bar_set2)
        
        chart.addSeries(series)
        
        # 创建坐标轴
        categories = ["类别1", "类别2", "类别3", "类别4", "类别5"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 10)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        self.chart_view.setChart(chart)
    
    def update_chart(self):
        """更新图表"""
        chart_type = self.chart_type_combo.currentText()
        time_range = self.time_range_combo.currentText()
        
        # TODO: 根据数据和选项更新图表
        # 这里先保持默认图表
        pass
    
    def update_with_results(self, results):
        """使用结果数据更新图表"""
        if not results:
            return
        
        # 根据当前选择的图表类型更新
        chart_type = self.chart_type_combo.currentText()
        
        if chart_type == "缺陷分布":
            self.create_defect_distribution_chart(results)
        elif chart_type == "时间趋势":
            self.create_time_trend_chart(results)
        elif chart_type == "置信度分布":
            self.create_confidence_distribution_chart(results)
        elif chart_type == "类型统计":
            self.create_type_statistics_chart(results)
        elif chart_type == "状态比例":
            self.create_status_pie_chart(results)
    
    def create_defect_distribution_chart(self, results):
        """创建缺陷分布图表"""
        chart = QChart()
        chart.setTitle("缺陷分布")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # 分析缺陷类型分布
        defect_counts = defaultdict(int)
        for result in results:
            if result.get('type') == '缺陷':
                defect_name = result.get('note', '未知缺陷')
                defect_counts[defect_name] += 1
        
        if not defect_counts:
            chart.setTitle("无缺陷数据")
            self.chart_view.setChart(chart)
            return
        
        # 创建柱状图
        series = QBarSeries()
        
        bar_set = QBarSet("缺陷数量")
        categories = []
        
        for defect_name, count in defect_counts.items():
            bar_set.append(count)
            categories.append(defect_name)
        
        bar_set.setColor(QColor(255, 99, 132))
        series.append(bar_set)
        
        chart.addSeries(series)
        
        # 创建坐标轴
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, max(defect_counts.values()) * 1.2)
        axis_y.setTitleText("数量")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        self.chart_view.setChart(chart)
    
    def create_time_trend_chart(self, results):
        """创建时间趋势图表"""
        chart = QChart()
        chart.setTitle("检测时间趋势")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # 按时间分组
        time_data = defaultdict(int)
        for result in results:
            timestamp = result.get('timestamp')
            if isinstance(timestamp, datetime):
                # 按小时分组
                time_key = timestamp.strftime("%Y-%m-%d %H:00")
                time_data[time_key] += 1
        
        if not time_data:
            chart.setTitle("无时间数据")
            self.chart_view.setChart(chart)
            return
        
        # 创建折线图
        series = QLineSeries()
        series.setName("检测数量")
        
        # 排序时间
        sorted_times = sorted(time_data.items())
        
        for i, (time_key, count) in enumerate(sorted_times):
            series.append(i, count)
        
        series.setColor(QColor(54, 162, 235))
        chart.addSeries(series)
        
        # 创建坐标轴
        axis_x = QValueAxis()
        axis_x.setRange(0, len(sorted_times) - 1)
        axis_x.setTitleText("时间")
        
        # 设置时间标签
        axis_x.setTickCount(min(10, len(sorted_times)))
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, max(time_data.values()) * 1.2)
        axis_y.setTitleText("数量")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        self.chart_view.setChart(chart)
    
    def create_confidence_distribution_chart(self, results):
        """创建置信度分布图表"""
        chart = QChart()
        chart.setTitle("置信度分布")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # 分析置信度分布
        confidence_ranges = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for result in results:
            confidence = result.get('confidence', 0)
            if confidence < 0.2:
                confidence_ranges["0.0-0.2"] += 1
            elif confidence < 0.4:
                confidence_ranges["0.2-0.4"] += 1
            elif confidence < 0.6:
                confidence_ranges["0.4-0.6"] += 1
            elif confidence < 0.8:
                confidence_ranges["0.6-0.8"] += 1
            else:
                confidence_ranges["0.8-1.0"] += 1
        
        # 创建柱状图
        series = QBarSeries()
        
        bar_set = QBarSet("数量")
        categories = []
        
        for range_name, count in confidence_ranges.items():
            bar_set.append(count)
            categories.append(range_name)
        
        bar_set.setColor(QColor(75, 192, 192))
        series.append(bar_set)
        
        chart.addSeries(series)
        
        # 创建坐标轴
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, max(confidence_ranges.values()) * 1.2)
        axis_y.setTitleText("数量")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        self.chart_view.setChart(chart)
    
    def create_type_statistics_chart(self, results):
        """创建类型统计图表"""
        chart = QChart()
        chart.setTitle("检测类型统计")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # 分析类型分布
        type_counts = defaultdict(int)
        for result in results:
            result_type = result.get('type', '未知')
            type_counts[result_type] += 1
        
        if not type_counts:
            chart.setTitle("无类型数据")
            self.chart_view.setChart(chart)
            return
        
        # 创建柱状图
        series = QBarSeries()
        
        bar_set = QBarSet("数量")
        categories = []
        
        for type_name, count in type_counts.items():
            bar_set.append(count)
            categories.append(type_name)
        
        # 设置颜色
        colors = [QColor(255, 99, 132), QColor(54, 162, 235), 
                 QColor(255, 206, 86), QColor(75, 192, 192),
                 QColor(153, 102, 255), QColor(255, 159, 64)]
        
        bar_set.setColor(colors[0])
        series.append(bar_set)
        
        chart.addSeries(series)
        
        # 创建坐标轴
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, max(type_counts.values()) * 1.2)
        axis_y.setTitleText("数量")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        
        self.chart_view.setChart(chart)
    
    def create_status_pie_chart(self, results):
        """创建状态比例饼图"""
        chart = QChart()
        chart.setTitle("检测状态比例")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        
        # 分析状态分布
        status_counts = defaultdict(int)
        for result in results:
            status = result.get('status', '未知')
            status_counts[status] += 1
        
        if not status_counts:
            chart.setTitle("无状态数据")
            self.chart_view.setChart(chart)
            return
        
        # 创建饼图
        series = QPieSeries()
        
        # 颜色映射
        color_map = {
            '通过': QColor(75, 192, 192),
            '失败': QColor(255, 99, 132),
            '警告': QColor(255, 206, 86),
            '未知': QColor(201, 203, 207)
        }
        
        for status, count in status_counts.items():
            slice_ = series.append(f"{status} ({count})", count)
            
            # 设置颜色
            if status in color_map:
                slice_.setColor(color_map[status])
            
            # 添加标签
            slice_.setLabelVisible(True)
        
        chart.addSeries(series)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.chart_view.setChart(chart)

class QualityMetricsWidget(QGroupBox):
    """质量指标组件"""
    
    def __init__(self):
        """初始化质量指标组件"""
        super().__init__("质量指标")
        
        self.metrics = {
            'total_inspected': 0,
            'defect_count': 0,
            'pass_count': 0,
            'fail_count': 0,
            'avg_confidence': 0.0,
            'avg_inspection_time': 0.0
        }
        
        self.init_ui()
        self.update_display()
    
    def init_ui(self):
        """初始化UI"""
        layout = QGridLayout()
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(10)
        
        # 总检测数
        layout.addWidget(QLabel("总检测数:"), 0, 0)
        self.total_label = QLabel("0")
        self.total_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.total_label, 0, 1)
        
        # 缺陷数量
        layout.addWidget(QLabel("缺陷数量:"), 0, 2)
        self.defect_label = QLabel("0")
        self.defect_label.setStyleSheet("font-weight: bold; color: #dc3545; font-size: 14px;")
        layout.addWidget(self.defect_label, 0, 3)
        
        # 通过率
        layout.addWidget(QLabel("通过率:"), 1, 0)
        self.pass_rate_label = QLabel("0.0%")
        self.pass_rate_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.pass_rate_label, 1, 1)
        
        # 缺陷率
        layout.addWidget(QLabel("缺陷率:"), 1, 2)
        self.defect_rate_label = QLabel("0.0%")
        self.defect_rate_label.setStyleSheet("font-weight: bold; color: #dc3545; font-size: 14px;")
        layout.addWidget(self.defect_rate_label, 1, 3)
        
        # 平均置信度
        layout.addWidget(QLabel("平均置信度:"), 2, 0)
        self.avg_conf_label = QLabel("0.000")
        self.avg_conf_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.avg_conf_label, 2, 1)
        
        # 平均检测时间
        layout.addWidget(QLabel("平均检测时间:"), 2, 2)
        self.avg_time_label = QLabel("0.0 ms")
        self.avg_time_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.avg_time_label, 2, 3)
        
        # OEE指标
        layout.addWidget(QLabel("设备综合效率 (OEE):"), 3, 0)
        self.oee_label = QLabel("0.0%")
        self.oee_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #28a745;")
        layout.addWidget(self.oee_label, 3, 1)
        
        # 直通率
        layout.addWidget(QLabel("一次通过率 (FPY):"), 3, 2)
        self.fpy_label = QLabel("0.0%")
        self.fpy_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(self.fpy_label, 3, 3)
        
        # 详细统计按钮
        self.details_button = QPushButton("详细统计")
        self.details_button.clicked.connect(self.show_details)
        layout.addWidget(self.details_button, 4, 0, 1, 4)
        
        self.setLayout(layout)
    
    def update_metrics(self, results):
        """更新质量指标"""
        if not results:
            self.reset_metrics()
            return
        
        # 重置指标
        self.metrics = {
            'total_inspected': 0,
            'defect_count': 0,
            'pass_count': 0,
            'fail_count': 0,
            'avg_confidence': 0.0,
            'avg_inspection_time': 0.0,
            'confidences': [],
            'inspection_times': []
        }
        
        # 计算指标
        for result in results:
            self.metrics['total_inspected'] += 1
            
            # 统计缺陷
            if result.get('type') == '缺陷':
                self.metrics['defect_count'] += 1
            
            # 统计通过/失败
            status = result.get('status', '')
            if status == '通过':
                self.metrics['pass_count'] += 1
            elif status == '失败':
                self.metrics['fail_count'] += 1
            
            # 收集置信度
            confidence = result.get('confidence', 0)
            if confidence > 0:
                self.metrics['confidences'].append(confidence)
            
            # 收集检测时间
            if 'inspection_time' in result:
                self.metrics['inspection_times'].append(result['inspection_time'])
        
        # 计算平均值
        if self.metrics['confidences']:
            self.metrics['avg_confidence'] = np.mean(self.metrics['confidences'])
        
        if self.metrics['inspection_times']:
            self.metrics['avg_inspection_time'] = np.mean(self.metrics['inspection_times'])
        
        # 更新显示
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        # 基础指标
        self.total_label.setText(str(self.metrics['total_inspected']))
        self.defect_label.setText(str(self.metrics['defect_count']))
        
        # 通过率和缺陷率
        total = self.metrics['total_inspected']
        if total > 0:
            pass_rate = (self.metrics['pass_count'] / total) * 100
            defect_rate = (self.metrics['defect_count'] / total) * 100
        else:
            pass_rate = 0.0
            defect_rate = 0.0
        
        self.pass_rate_label.setText(f"{pass_rate:.1f}%")
        self.defect_rate_label.setText(f"{defect_rate:.1f}%")
        
        # 平均置信度和检测时间
        self.avg_conf_label.setText(f"{self.metrics['avg_confidence']:.3f}")
        self.avg_time_label.setText(f"{self.metrics['avg_inspection_time']:.1f} ms")
        
        # OEE和FPY（简化计算）
        if total > 0:
            availability = 0.95  # 假设可用性95%
            performance = 0.98   # 假设性能效率98%
            quality = (total - self.metrics['defect_count']) / total
            
            oee = availability * performance * quality * 100
            fpy = (self.metrics['pass_count'] / total) * 100
            
            self.oee_label.setText(f"{oee:.1f}%")
            self.fpy_label.setText(f"{fpy:.1f}%")
        else:
            self.oee_label.setText("0.0%")
            self.fpy_label.setText("0.0%")
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics = {
            'total_inspected': 0,
            'defect_count': 0,
            'pass_count': 0,
            'fail_count': 0,
            'avg_confidence': 0.0,
            'avg_inspection_time': 0.0
        }
        self.update_display()
    
    def show_details(self):
        """显示详细统计"""
        details_text = f"""
=== 质量指标详细统计 ===

基础统计:
  总检测数: {self.metrics['total_inspected']}
  缺陷数量: {self.metrics['defect_count']}
  通过数量: {self.metrics['pass_count']}
  失败数量: {self.metrics['fail_count']}

比率指标:
  通过率: {(self.metrics['pass_count']/self.metrics['total_inspected']*100) if self.metrics['total_inspected'] > 0 else 0:.1f}%
  缺陷率: {(self.metrics['defect_count']/self.metrics['total_inspected']*100) if self.metrics['total_inspected'] > 0 else 0:.1f}%
  失败率: {(self.metrics['fail_count']/self.metrics['total_inspected']*100) if self.metrics['total_inspected'] > 0 else 0:.1f}%

性能指标:
  平均置信度: {self.metrics['avg_confidence']:.3f}
  平均检测时间: {self.metrics['avg_inspection_time']:.1f} ms

生产指标:
  设备综合效率 (OEE): {self.oee_label.text()}
  一次通过率 (FPY): {self.fpy_label.text()}
        """
        
        QMessageBox.information(self, "详细统计", details_text.strip())

class AnalysisWidget(QWidget):
    """分析主组件"""
    
    # 信号
    result_selected = pyqtSignal(dict)
    analysis_export_requested = pyqtSignal(str)
    realtime_analysis_toggled = pyqtSignal(bool)
    
    def __init__(self, data_manager):
        """初始化分析组件
        
        Args:
            data_manager: 数据管理器实例
        """
        super().__init__()
        
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # 状态变量
        self.realtime_analysis = False
        self.current_results = []
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 初始化数据
        self.load_historical_data()
        
        self.logger.info("分析组件初始化完成")
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)
        
        # 工具栏
        self.create_toolbar(main_layout)
        
        # 创建主分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 上部：质量指标和图表
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        
        # 质量指标（左侧）
        self.metrics_widget = QualityMetricsWidget()
        top_layout.addWidget(self.metrics_widget, 1)  # 1/3宽度
        
        # 统计图表（右侧）
        self.chart_widget = StatisticsChartWidget()
        top_layout.addWidget(self.chart_widget, 2)  # 2/3宽度
        
        splitter.addWidget(top_widget)
        
        # 下部：检测结果
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.result_widget = DetectionResultWidget()
        bottom_layout.addWidget(self.result_widget)
        
        splitter.addWidget(bottom_widget)
        
        # 设置分割器比例
        splitter.setSizes([300, 400])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def create_toolbar(self, parent_layout):
        """创建工具栏"""
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        toolbar.setMaximumHeight(40)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 分析模式
        layout.addWidget(QLabel("分析模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["历史分析", "实时分析"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)
        
        layout.addSpacing(20)
        
        # 数据范围
        layout.addWidget(QLabel("数据范围:"))
        self.range_combo = QComboBox()
        self.range_combo.addItems([
            "全部数据", "今天", "最近7天", "最近30天", 
            "自定义范围"
        ])
        self.range_combo.currentTextChanged.connect(self.on_range_changed)
        layout.addWidget(self.range_combo)
        
        # 自定义范围控件（默认隐藏）
        self.custom_range_widget = QWidget()
        custom_layout = QHBoxLayout(self.custom_range_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))
        custom_layout.addWidget(self.start_date_edit)
        
        custom_layout.addWidget(QLabel("到"))
        
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        custom_layout.addWidget(self.end_date_edit)
        
        self.apply_range_button = QPushButton("应用")
        self.apply_range_button.clicked.connect(self.apply_custom_range)
        custom_layout.addWidget(self.apply_range_button)
        
        custom_layout.addStretch()
        self.custom_range_widget.setVisible(False)
        layout.addWidget(self.custom_range_widget)
        
        layout.addSpacing(20)
        
        # 分析按钮
        self.analyze_button = QPushButton("分析")
        self.analyze_button.setStyleSheet("background-color: #007bff; color: white;")
        self.analyze_button.clicked.connect(self.perform_analysis)
        layout.addWidget(self.analyze_button)
        
        # 导出按钮
        self.export_button = QPushButton("导出分析")
        self.export_button.clicked.connect(self.export_analysis)
        layout.addWidget(self.export_button)
        
        # 重置按钮
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_analysis)
        layout.addWidget(self.reset_button)
        
        layout.addStretch()
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)
        
        parent_layout.addWidget(toolbar)
    
    def connect_signals(self):
        """连接信号"""
        # 数据管理器信号
        if hasattr(self.data_manager, 'data_updated'):
            self.data_manager.data_updated.connect(self.on_data_updated)
        
        if hasattr(self.data_manager, 'real_time_data_received'):
            self.data_manager.real_time_data_received.connect(self.on_realtime_data_received)
        
        # 结果组件信号
        self.result_widget.result_selected.connect(self.on_result_selected)
        
        # 图表组件信号
        # 图表更新已经在图表组件内部处理
    
    def load_historical_data(self):
        """加载历史数据"""
        try:
            # 从数据管理器获取历史数据
            historical_data = self.data_manager.get_historical_data()
            
            if historical_data:
                self.current_results = historical_data
                self.update_all_components()
                self.status_label.setText(f"已加载 {len(historical_data)} 条历史数据")
            else:
                self.status_label.setText("无历史数据")
                
        except Exception as e:
            self.logger.error(f"加载历史数据错误: {e}")
            self.status_label.setText("加载历史数据失败")
    
    @pyqtSlot(str)
    def on_mode_changed(self, mode):
        """分析模式变化处理"""
        self.realtime_analysis = (mode == "实时分析")
        self.realtime_analysis_toggled.emit(self.realtime_analysis)
        
        if self.realtime_analysis:
            self.status_label.setText("实时分析模式已启用")
            # TODO: 启动实时数据订阅
        else:
            self.status_label.setText("历史分析模式")
            # TODO: 停止实时数据订阅
    
    @pyqtSlot(str)
    def on_range_changed(self, range_text):
        """数据范围变化处理"""
        if range_text == "自定义范围":
            self.custom_range_widget.setVisible(True)
        else:
            self.custom_range_widget.setVisible(False)
            
            # 应用预设范围
            self.apply_preset_range(range_text)
    
    def apply_preset_range(self, range_text):
        """应用预设时间范围"""
        end_date = datetime.now()
        
        if range_text == "今天":
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif range_text == "最近7天":
            start_date = end_date - timedelta(days=7)
        elif range_text == "最近30天":
            start_date = end_date - timedelta(days=30)
        elif range_text == "全部数据":
            start_date = datetime.min
        else:
            return
        
        # 从数据管理器获取指定范围的数据
        try:
            filtered_data = self.data_manager.get_data_by_date_range(start_date, end_date)
            self.current_results = filtered_data
            self.update_all_components()
            
            self.status_label.setText(f"已加载 {len(filtered_data)} 条数据 ({range_text})")
            
        except Exception as e:
            self.logger.error(f"应用范围过滤错误: {e}")
            self.status_label.setText(f"过滤失败: {str(e)}")
    
    def apply_custom_range(self):
        """应用自定义时间范围"""
        start_date = self.start_date_edit.date().toPyDate()
        end_date = self.end_date_edit.date().toPyDate()
        
        # 将日期转换为datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        # 从数据管理器获取数据
        try:
            filtered_data = self.data_manager.get_data_by_date_range(start_datetime, end_datetime)
            self.current_results = filtered_data
            self.update_all_components()
            
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            self.status_label.setText(f"已加载 {len(filtered_data)} 条数据 ({start_str} 到 {end_str})")
            
        except Exception as e:
            self.logger.error(f"应用自定义范围错误: {e}")
            self.status_label.setText(f"自定义范围过滤失败: {str(e)}")
    
    @pyqtSlot(dict)
    def on_data_updated(self, data_info):
        """数据更新处理"""
        # 如果处于实时分析模式，更新当前结果
        if self.realtime_analysis and 'results' in data_info:
            new_results = data_info['results']
            self.current_results.extend(new_results)
            self.update_all_components()
            
            self.status_label.setText(f"实时数据更新: +{len(new_results)} 条")
    
    @pyqtSlot(dict)
    def on_realtime_data_received(self, data):
        """实时数据接收处理"""
        if self.realtime_analysis:
            # 添加实时结果
            self.result_widget.add_result(data)
            
            # 更新当前结果列表
            self.current_results.append(data)
            
            # 更新其他组件
            self.metrics_widget.update_metrics(self.current_results)
            self.chart_widget.update_with_results(self.current_results)
    
    @pyqtSlot(dict)
    def on_result_selected(self, result):
        """结果选择处理"""
        self.result_selected.emit(result)
        
        # 可以在这里显示结果的详细信息
        # 例如：在状态栏显示选中结果的信息
        result_type = result.get('type', '未知')
        confidence = result.get('confidence', 0)
        self.status_label.setText(f"选中: {result_type} (置信度: {confidence:.3f})")
    
    def update_all_components(self):
        """更新所有组件"""
        # 更新结果表格
        self.result_widget.results = self.current_results.copy()
        self.result_widget.update_table()
        
        # 更新质量指标
        self.metrics_widget.update_metrics(self.current_results)
        
        # 更新统计图表
        self.chart_widget.update_with_results(self.current_results)
    
    def perform_analysis(self):
        """执行分析"""
        if not self.current_results:
            QMessageBox.warning(self, "警告", "没有可分析的数据")
            return
        
        try:
            self.status_label.setText("正在分析数据...")
            
            # 更新所有组件（已经包含分析逻辑）
            self.update_all_components()
            
            # 计算额外统计信息
            total_count = len(self.current_results)
            defect_count = sum(1 for r in self.current_results if r.get('type') == '缺陷')
            
            self.status_label.setText(f"分析完成: {total_count} 条数据，{defect_count} 个缺陷")
            
        except Exception as e:
            self.logger.error(f"分析错误: {e}")
            self.status_label.setText(f"分析失败: {str(e)}")
            QMessageBox.critical(self, "分析错误", f"分析过程中出错: {str(e)}")
    
    def export_analysis(self):
        """导出分析结果"""
        if not self.current_results:
            QMessageBox.warning(self, "警告", "没有可导出的分析结果")
            return
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "导出分析结果", "",
            "HTML报告 (*.html);;PDF文件 (*.pdf);;Excel文件 (*.xlsx)"
        )
        
        if not file_path:
            return
        
        try:
            # 准备分析报告数据
            report_data = {
                'analysis_time': datetime.now().isoformat(),
                'total_results': len(self.current_results),
                'defect_count': sum(1 for r in self.current_results if r.get('type') == '缺陷'),
                'pass_count': sum(1 for r in self.current_results if r.get('status') == '通过'),
                'metrics': self.metrics_widget.metrics,
                'results_sample': self.current_results[:100] if len(self.current_results) > 100 else self.current_results
            }
            
            # 根据文件类型导出
            if file_path.endswith('.html'):
                self.export_html_report(file_path, report_data)
            elif file_path.endswith('.pdf'):
                self.export_pdf_report(file_path, report_data)
            elif file_path.endswith('.xlsx'):
                self.export_excel_report(file_path, report_data)
            
            self.analysis_export_requested.emit(file_path)
            self.status_label.setText(f"分析结果已导出到: {file_path}")
            
            QMessageBox.information(self, "成功", f"分析结果已导出到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"导出分析错误: {e}")
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def export_html_report(self, file_path, report_data):
        """导出HTML报告"""
        # 构建HTML内容
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>工业视觉检测分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .metric-value {{ color: #007bff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .defect {{ background-color: #ffebee; }}
        .pass {{ background-color: #e8f5e9; }}
        .timestamp {{ font-size: 12px; color: #999; text-align: right; }}
    </style>
</head>
<body>
    <h1>工业视觉检测分析报告</h1>
    <div class="timestamp">生成时间: {report_data['analysis_time']}</div>
    
    <div class="summary">
        <h2>分析摘要</h2>
        <div class="metric">
            <span class="metric-label">总检测数:</span>
            <span class="metric-value">{report_data['total_results']}</span>
        </div>
        <div class="metric">
            <span class="metric-label">缺陷数量:</span>
            <span class="metric-value">{report_data['defect_count']}</span>
        </div>
        <div class="metric">
            <span class="metric-label">通过数量:</span>
            <span class="metric-value">{report_data['pass_count']}</span>
        </div>
        <div class="metric">
            <span class="metric-label">缺陷率:</span>
            <span class="metric-value">{(report_data['defect_count']/report_data['total_results']*100) if report_data['total_results'] > 0 else 0:.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">通过率:</span>
            <span class="metric-value">{(report_data['pass_count']/report_data['total_results']*100) if report_data['total_results'] > 0 else 0:.1f}%</span>
        </div>
    </div>
    
    <h2>详细结果 (前{len(report_data['results_sample'])}条)</h2>
    <table>
        <tr>
            <th>时间</th>
            <th>类型</th>
            <th>置信度</th>
            <th>状态</th>
            <th>备注</th>
        </tr>
"""
        
        # 添加结果行
        for result in report_data['results_sample']:
            result_class = ''
            if result.get('type') == '缺陷':
                result_class = 'defect'
            elif result.get('status') == '通过':
                result_class = 'pass'
            
            timestamp = result.get('timestamp', '')
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp_str = str(timestamp)
            
            html_content += f"""
        <tr class="{result_class}">
            <td>{timestamp_str}</td>
            <td>{result.get('type', '未知')}</td>
            <td>{result.get('confidence', 0):.3f}</td>
            <td>{result.get('status', '未知')}</td>
            <td>{result.get('note', '')}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <div class="timestamp">
        <p>报告结束</p>
    </div>
</body>
</html>
"""
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def export_pdf_report(self, file_path, report_data):
        """导出PDF报告"""
        # TODO: 实现PDF导出功能
        # 可以使用reportlab或weasyprint库
        raise NotImplementedError("PDF导出功能尚未实现")
    
    def export_excel_report(self, file_path, report_data):
        """导出Excel报告"""
        import pandas as pd
        
        # 创建DataFrame
        data_rows = []
        for result in self.current_results:
            timestamp = result.get('timestamp', '')
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
            
            data_rows.append({
                '时间': timestamp_str,
                '类型': result.get('type', ''),
                '置信度': result.get('confidence', 0),
                '状态': result.get('status', ''),
                '备注': result.get('note', '')
            })
        
        df = pd.DataFrame(data_rows)
        
        # 写入Excel文件
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='检测结果', index=False)
            
            # 添加统计信息
            summary_data = {
                '指标': ['总检测数', '缺陷数量', '通过数量', '缺陷率', '通过率', '生成时间'],
                '数值': [
                    report_data['total_results'],
                    report_data['defect_count'],
                    report_data['pass_count'],
                    f"{(report_data['defect_count']/report_data['total_results']*100) if report_data['total_results'] > 0 else 0:.1f}%",
                    f"{(report_data['pass_count']/report_data['total_results']*100) if report_data['total_results'] > 0 else 0:.1f}%",
                    report_data['analysis_time']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
    
    def reset_analysis(self):
        """重置分析"""
        reply = QMessageBox.question(
            self, "确认重置",
            "确定要重置所有分析结果吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.current_results.clear()
            self.result_widget.clear_results()
            self.metrics_widget.reset_metrics()
            self.chart_widget.create_default_chart()
            self.status_label.setText("分析已重置")
    
    def update_results(self, result):
        """更新结果（从外部调用）"""
        if isinstance(result, list):
            self.result_widget.add_batch_results(result)
        else:
            self.result_widget.add_result(result)
        
        # 添加到当前结果列表
        if isinstance(result, list):
            self.current_results.extend(result)
        else:
            self.current_results.append(result)
        
        # 更新其他组件
        self.metrics_widget.update_metrics(self.current_results)
        self.chart_widget.update_with_results(self.current_results)