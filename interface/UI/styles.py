"""
工业视觉检测系统 - 样式表
定义应用程序的整体外观和主题
"""

def get_style_sheet(theme="dark"):
    """获取样式表
    
    Args:
        theme: 主题类型，支持 "dark"（深色）或 "light"（浅色）
    
    Returns:
        str: 样式表字符串
    """
    if theme == "dark":
        return get_dark_theme()
    elif theme == "light":
        return get_light_theme()
    else:
        return get_dark_theme()

def get_dark_theme():
    """深色主题样式表"""
    return """
    /* ===== 主窗口 ===== */
    QMainWindow {
        background-color: #2b2b2b;
    }
    
    /* ===== 通用设置 ===== */
    QWidget {
        color: #e0e0e0;
        font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
        font-size: 11px;
    }
    
    /* ===== 按钮 ===== */
    QPushButton {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 6px 12px;
        min-height: 20px;
    }
    
    QPushButton:hover {
        background-color: #4a4a4a;
        border-color: #666;
    }
    
    QPushButton:pressed {
        background-color: #2a2a2a;
        border-color: #444;
    }
    
    QPushButton:checked {
        background-color: #0078d7;
        border-color: #005a9e;
    }
    
    QPushButton:disabled {
        background-color: #2b2b2b;
        border-color: #444;
        color: #666;
    }
    
    /* 主要按钮 */
    QPushButton[class="primary"] {
        background-color: #0078d7;
        border-color: #005a9e;
        color: white;
        font-weight: bold;
    }
    
    QPushButton[class="primary"]:hover {
        background-color: #106ebe;
        border-color: #005a9e;
    }
    
    QPushButton[class="primary"]:pressed {
        background-color: #005a9e;
        border-color: #004578;
    }
    
    /* 成功按钮 */
    QPushButton[class="success"] {
        background-color: #28a745;
        border-color: #1e7e34;
        color: white;
    }
    
    QPushButton[class="success"]:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
    
    /* 警告按钮 */
    QPushButton[class="warning"] {
        background-color: #ffc107;
        border-color: #d39e00;
        color: #212529;
    }
    
    QPushButton[class="warning"]:hover {
        background-color: #e0a800;
        border-color: #d39e00;
    }
    
    /* 危险按钮 */
    QPushButton[class="danger"] {
        background-color: #dc3545;
        border-color: #bd2130;
        color: white;
    }
    
    QPushButton[class="danger"]:hover {
        background-color: #c82333;
        border-color: #bd2130;
    }
    
    /* ===== 标签 ===== */
    QLabel {
        color: #e0e0e0;
    }
    
    QLabel[class="title"] {
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
    }
    
    QLabel[class="subtitle"] {
        font-size: 12px;
        font-weight: bold;
        color: #cccccc;
    }
    
    QLabel[class="info"] {
        color: #6c757d;
        font-size: 10px;
    }
    
    QLabel[class="success"] {
        color: #28a745;
    }
    
    QLabel[class="warning"] {
        color: #ffc107;
    }
    
    QLabel[class="error"] {
        color: #dc3545;
    }
    
    /* ===== 文本框 ===== */
    QLineEdit {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-radius: 3px;
        padding: 5px 8px;
        selection-background-color: #0078d7;
        selection-color: white;
    }
    
    QLineEdit:hover {
        border-color: #666;
    }
    
    QLineEdit:focus {
        border-color: #0078d7;
        background-color: #2d2d2d;
    }
    
    QLineEdit:disabled {
        background-color: #2b2b2b;
        border-color: #444;
        color: #666;
    }
    
    QLineEdit[class="search"] {
        padding-left: 30px;
        background-image: url("resources/icons/search.png");
        background-repeat: no-repeat;
        background-position: 8px center;
    }
    
    /* ===== 文本编辑框 ===== */
    QTextEdit {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-radius: 3px;
        padding: 5px;
        selection-background-color: #0078d7;
        selection-color: white;
    }
    
    QTextEdit:focus {
        border-color: #0078d7;
    }
    
    QTextEdit:disabled {
        background-color: #2b2b2b;
        border-color: #444;
        color: #666;
    }
    
    /* ===== 组合框 ===== */
    QComboBox {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-radius: 3px;
        padding: 5px 8px;
        min-height: 20px;
    }
    
    QComboBox:hover {
        border-color: #666;
    }
    
    QComboBox:focus {
        border-color: #0078d7;
    }
    
    QComboBox:disabled {
        background-color: #2b2b2b;
        border-color: #444;
        color: #666;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: url("resources/icons/down-arrow.png");
        width: 12px;
        height: 12px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #3c3c3c;
        border: 1px solid #555;
        selection-background-color: #0078d7;
        selection-color: white;
    }
    
    /* ===== 复选框 ===== */
    QCheckBox {
        spacing: 5px;
    }
    
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    
    QCheckBox::indicator:unchecked {
        border: 1px solid #666;
        background-color: #3c3c3c;
        border-radius: 2px;
    }
    
    QCheckBox::indicator:unchecked:hover {
        border-color: #777;
    }
    
    QCheckBox::indicator:checked {
        border: 1px solid #0078d7;
        background-color: #0078d7;
        border-radius: 2px;
        image: url("resources/icons/checkmark.png");
    }
    
    QCheckBox::indicator:disabled {
        border: 1px solid #444;
        background-color: #2b2b2b;
    }
    
    /* ===== 单选按钮 ===== */
    QRadioButton {
        spacing: 5px;
    }
    
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
    
    QRadioButton::indicator:unchecked {
        border: 1px solid #666;
        background-color: #3c3c3c;
        border-radius: 8px;
    }
    
    QRadioButton::indicator:unchecked:hover {
        border-color: #777;
    }
    
    QRadioButton::indicator:checked {
        border: 1px solid #0078d7;
        background-color: #0078d7;
        border-radius: 8px;
    }
    
    QRadioButton::indicator:disabled {
        border: 1px solid #444;
        background-color: #2b2b2b;
    }
    
    /* ===== 滑块 ===== */
    QSlider::groove:horizontal {
        background-color: #444;
        height: 4px;
        border-radius: 2px;
    }
    
    QSlider::sub-page:horizontal {
        background-color: #0078d7;
        border-radius: 2px;
    }
    
    QSlider::add-page:horizontal {
        background-color: #444;
        border-radius: 2px;
    }
    
    QSlider::handle:horizontal {
        background-color: #ffffff;
        border: 1px solid #666;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }
    
    QSlider::handle:horizontal:hover {
        background-color: #f0f0f0;
        border-color: #777;
    }
    
    QSlider::groove:vertical {
        background-color: #444;
        width: 4px;
        border-radius: 2px;
    }
    
    QSlider::sub-page:vertical {
        background-color: #0078d7;
        border-radius: 2px;
    }
    
    QSlider::add-page:vertical {
        background-color: #444;
        border-radius: 2px;
    }
    
    QSlider::handle:vertical {
        background-color: #ffffff;
        border: 1px solid #666;
        width: 16px;
        height: 16px;
        margin: 0 -6px;
        border-radius: 8px;
    }
    
    QSlider::handle:vertical:hover {
        background-color: #f0f0f0;
        border-color: #777;
    }
    
    /* ===== 进度条 ===== */
    QProgressBar {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-radius: 3px;
        text-align: center;
    }
    
    QProgressBar::chunk {
        background-color: #0078d7;
        border-radius: 2px;
    }
    
    QProgressBar[class="success"]::chunk {
        background-color: #28a745;
    }
    
    QProgressBar[class="warning"]::chunk {
        background-color: #ffc107;
    }
    
    QProgressBar[class="danger"]::chunk {
        background-color: #dc3545;
    }
    
    /* ===== 选项卡 ===== */
    QTabWidget::pane {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-radius: 3px;
        margin-top: -1px;
    }
    
    QTabWidget::tab-bar {
        alignment: left;
    }
    
    QTabBar::tab {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 6px 12px;
        margin-right: 2px;
    }
    
    QTabBar::tab:hover {
        background-color: #4a4a4a;
    }
    
    QTabBar::tab:selected {
        background-color: #2b2b2b;
        border-color: #555;
        border-bottom-color: #2b2b2b;
    }
    
    QTabBar::tab:!selected {
        margin-top: 2px;
    }
    
    /* ===== 表格 ===== */
    QTableWidget {
        background-color: #3c3c3c;
        border: 1px solid #555;
        gridline-color: #555;
        selection-background-color: #0078d7;
        selection-color: white;
        alternate-background-color: #353535;
    }
    
    QTableWidget::item {
        padding: 5px;
        border: none;
    }
    
    QTableWidget::item:selected {
        background-color: #0078d7;
        color: white;
    }
    
    QTableWidget QHeaderView::section {
        background-color: #2b2b2b;
        color: #e0e0e0;
        padding: 6px;
        border: 1px solid #555;
        border-left: none;
        border-top: none;
        font-weight: bold;
    }
    
    QTableWidget QHeaderView::section:first {
        border-left: 1px solid #555;
    }
    
    QTableWidget QHeaderView::section:hover {
        background-color: #3a3a3a;
    }
    
    QTableWidget QHeaderView::section:pressed {
        background-color: #2a2a2a;
    }
    
    /* ===== 列表 ===== */
    QListWidget {
        background-color: #3c3c3c;
        border: 1px solid #555;
        outline: none;
    }
    
    QListWidget::item {
        padding: 5px;
        border: none;
    }
    
    QListWidget::item:hover {
        background-color: #4a4a4a;
    }
    
    QListWidget::item:selected {
        background-color: #0078d7;
        color: white;
    }
    
    QListWidget::item:selected:!active {
        background-color: #005a9e;
    }
    
    /* ===== 树形视图 ===== */
    QTreeWidget {
        background-color: #3c3c3c;
        border: 1px solid #555;
        outline: none;
    }
    
    QTreeWidget::item {
        padding: 4px;
        border: none;
    }
    
    QTreeWidget::item:hover {
        background-color: #4a4a4a;
    }
    
    QTreeWidget::item:selected {
        background-color: #0078d7;
        color: white;
    }
    
    QTreeWidget::item:selected:!active {
        background-color: #005a9e;
    }
    
    QTreeWidget::branch:has-siblings:!adjoins-item {
        border-image: url("resources/icons/branch-line.png") 0;
    }
    
    QTreeWidget::branch:has-siblings:adjoins-item {
        border-image: url("resources/icons/branch-more.png") 0;
    }
    
    QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {
        border-image: url("resources/icons/branch-end.png") 0;
    }
    
    QTreeWidget::branch:has-children:!has-siblings:closed,
    QTreeWidget::branch:closed:has-children:has-siblings {
        border-image: none;
        image: url("resources/icons/branch-closed.png");
    }
    
    QTreeWidget::branch:open:has-children:!has-siblings,
    QTreeWidget::branch:open:has-children:has-siblings {
        border-image: none;
        image: url("resources/icons/branch-open.png");
    }
    
    /* ===== 分组框 ===== */
    QGroupBox {
        background-color: #3c3c3c;
        border: 1px solid #555;
        border-radius: 4px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
        color: #cccccc;
    }
    
    /* ===== 框架 ===== */
    QFrame {
        background-color: #3c3c3c;
        border: 1px solid #555;
    }
    
    QFrame[class="separator"] {
        background-color: #555;
        border: none;
        min-height: 1px;
        max-height: 1px;
    }
    
    QFrame[class="panel"] {
        background-color: #2b2b2b;
        border: 1px solid #444;
        border-radius: 4px;
    }
    
    /* ===== 菜单栏 ===== */
    QMenuBar {
        background-color: #2b2b2b;
        color: #e0e0e0;
        border-bottom: 1px solid #444;
    }
    
    QMenuBar::item {
        background-color: transparent;
        padding: 5px 10px;
    }
    
    QMenuBar::item:selected {
        background-color: #3c3c3c;
    }
    
    QMenuBar::item:pressed {
        background-color: #2a2a2a;
    }
    
    /* ===== 菜单 ===== */
    QMenu {
        background-color: #3c3c3c;
        border: 1px solid #555;
        color: #e0e0e0;
    }
    
    QMenu::item {
        padding: 5px 20px;
    }
    
    QMenu::item:selected {
        background-color: #0078d7;
        color: white;
    }
    
    QMenu::item:disabled {
        color: #666;
    }
    
    QMenu::separator {
        background-color: #555;
        height: 1px;
        margin: 5px 10px;
    }
    
    /* ===== 工具栏 ===== */
    QToolBar {
        background-color: #2b2b2b;
        border: none;
        border-bottom: 1px solid #444;
        spacing: 5px;
        padding: 2px;
    }
    
    QToolBar::separator {
        background-color: #555;
        width: 1px;
        margin: 0 5px;
    }
    
    /* ===== 工具栏按钮 ===== */
    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 3px;
    }
    
    QToolButton:hover {
        background-color: #3c3c3c;
        border-color: #555;
    }
    
    QToolButton:pressed {
        background-color: #2a2a2a;
        border-color: #444;
    }
    
    QToolButton:checked {
        background-color: #0078d7;
        border-color: #005a9e;
    }
    
    /* ===== 状态栏 ===== */
    QStatusBar {
        background-color: #2b2b2b;
        color: #e0e0e0;
        border-top: 1px solid #444;
    }
    
    QStatusBar::item {
        border: none;
    }
    
    /* ===== 滚动条 ===== */
    QScrollBar:vertical {
        background-color: #2b2b2b;
        width: 12px;
        margin: 0;
    }
    
    QScrollBar::handle:vertical {
        background-color: #555;
        border-radius: 6px;
        min-height: 20px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #666;
    }
    
    QScrollBar::handle:vertical:pressed {
        background-color: #444;
    }
    
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        background-color: #2b2b2b;
        height: 0px;
        subcontrol-position: top;
        subcontrol-origin: margin;
    }
    
    QScrollBar:horizontal {
        background-color: #2b2b2b;
        height: 12px;
        margin: 0;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #555;
        border-radius: 6px;
        min-width: 20px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #666;
    }
    
    QScrollBar::handle:horizontal:pressed {
        background-color: #444;
    }
    
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {
        background-color: #2b2b2b;
        width: 0px;
        subcontrol-position: left;
        subcontrol-origin: margin;
    }
    
    /* ===== 分割器 ===== */
    QSplitter::handle {
        background-color: #444;
    }
    
    QSplitter::handle:hover {
        background-color: #555;
    }
    
    QSplitter::handle:horizontal {
        width: 3px;
    }
    
    QSplitter::handle:vertical {
        height: 3px;
    }
    
    /* ===== 消息框 ===== */
    QMessageBox {
        background-color: #2b2b2b;
    }
    
    QMessageBox QLabel {
        color: #e0e0e0;
    }
    
    /* ===== 特殊状态 ===== */
    .highlight {
        background-color: rgba(0, 120, 215, 0.3);
        border: 1px solid rgba(0, 120, 215, 0.5);
        border-radius: 3px;
    }
    
    .error-border {
        border: 1px solid #dc3545 !important;
    }
    
    .warning-border {
        border: 1px solid #ffc107 !important;
    }
    
    .success-border {
        border: 1px solid #28a745 !important;
    }
    
    .info-border {
        border: 1px solid #17a2b8 !important;
    }
    
    /* ===== 动画效果 ===== */
    QProgressBar, QSlider {
        transition: background-color 0.2s ease;
    }
    
    QPushButton, QToolButton {
        transition: all 0.2s ease;
    }
    """

def get_light_theme():
    """浅色主题样式表"""
    return """
    /* ===== 主窗口 ===== */
    QMainWindow {
        background-color: #f5f5f5;
    }
    
    /* ===== 通用设置 ===== */
    QWidget {
        color: #333333;
        font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
        font-size: 11px;
    }
    
    /* ===== 按钮 ===== */
    QPushButton {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 6px 12px;
        min-height: 20px;
    }
    
    QPushButton:hover {
        background-color: #f8f9fa;
        border-color: #b8b8b8;
    }
    
    QPushButton:pressed {
        background-color: #e9ecef;
        border-color: #a0a0a0;
    }
    
    QPushButton:checked {
        background-color: #0078d7;
        border-color: #005a9e;
        color: white;
    }
    
    QPushButton:disabled {
        background-color: #f8f9fa;
        border-color: #dee2e6;
        color: #adb5bd;
    }
    
    /* 主要按钮 */
    QPushButton[class="primary"] {
        background-color: #0078d7;
        border-color: #005a9e;
        color: white;
        font-weight: bold;
    }
    
    QPushButton[class="primary"]:hover {
        background-color: #106ebe;
        border-color: #005a9e;
    }
    
    QPushButton[class="primary"]:pressed {
        background-color: #005a9e;
        border-color: #004578;
    }
    
    /* 成功按钮 */
    QPushButton[class="success"] {
        background-color: #28a745;
        border-color: #1e7e34;
        color: white;
    }
    
    QPushButton[class="success"]:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
    
    /* 警告按钮 */
    QPushButton[class="warning"] {
        background-color: #ffc107;
        border-color: #d39e00;
        color: #212529;
    }
    
    QPushButton[class="warning"]:hover {
        background-color: #e0a800;
        border-color: #d39e00;
    }
    
    /* 危险按钮 */
    QPushButton[class="danger"] {
        background-color: #dc3545;
        border-color: #bd2130;
        color: white;
    }
    
    QPushButton[class="danger"]:hover {
        background-color: #c82333;
        border-color: #bd2130;
    }
    
    /* ===== 标签 ===== */
    QLabel {
        color: #333333;
    }
    
    QLabel[class="title"] {
        font-size: 14px;
        font-weight: bold;
        color: #212529;
    }
    
    QLabel[class="subtitle"] {
        font-size: 12px;
        font-weight: bold;
        color: #495057;
    }
    
    QLabel[class="info"] {
        color: #6c757d;
        font-size: 10px;
    }
    
    QLabel[class="success"] {
        color: #28a745;
    }
    
    QLabel[class="warning"] {
        color: #ffc107;
    }
    
    QLabel[class="error"] {
        color: #dc3545;
    }
    
    /* ===== 文本框 ===== */
    QLineEdit {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 3px;
        padding: 5px 8px;
        selection-background-color: #0078d7;
        selection-color: white;
    }
    
    QLineEdit:hover {
        border-color: #adb5bd;
    }
    
    QLineEdit:focus {
        border-color: #0078d7;
        background-color: #ffffff;
    }
    
    QLineEdit:disabled {
        background-color: #e9ecef;
        border-color: #ced4da;
        color: #6c757d;
    }
    
    QLineEdit[class="search"] {
        padding-left: 30px;
        background-image: url("resources/icons/search-light.png");
        background-repeat: no-repeat;
        background-position: 8px center;
    }
    
    /* ===== 文本编辑框 ===== */
    QTextEdit {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 3px;
        padding: 5px;
        selection-background-color: #0078d7;
        selection-color: white;
    }
    
    QTextEdit:focus {
        border-color: #0078d7;
    }
    
    QTextEdit:disabled {
        background-color: #e9ecef;
        border-color: #ced4da;
        color: #6c757d;
    }
    
    /* ===== 组合框 ===== */
    QComboBox {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 3px;
        padding: 5px 8px;
        min-height: 20px;
    }
    
    QComboBox:hover {
        border-color: #adb5bd;
    }
    
    QComboBox:focus {
        border-color: #0078d7;
    }
    
    QComboBox:disabled {
        background-color: #e9ecef;
        border-color: #ced4da;
        color: #6c757d;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: url("resources/icons/down-arrow-light.png");
        width: 12px;
        height: 12px;
    }
    
    QComboBox QAbstractItemView {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        selection-background-color: #0078d7;
        selection-color: white;
    }
    
    /* ===== 复选框 ===== */
    QCheckBox {
        spacing: 5px;
    }
    
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
    }
    
    QCheckBox::indicator:unchecked {
        border: 1px solid #ced4da;
        background-color: #ffffff;
        border-radius: 2px;
    }
    
    QCheckBox::indicator:unchecked:hover {
        border-color: #adb5bd;
    }
    
    QCheckBox::indicator:checked {
        border: 1px solid #0078d7;
        background-color: #0078d7;
        border-radius: 2px;
        image: url("resources/icons/checkmark-light.png");
    }
    
    QCheckBox::indicator:disabled {
        border: 1px solid #e9ecef;
        background-color: #f8f9fa;
    }
    
    /* ===== 单选按钮 ===== */
    QRadioButton {
        spacing: 5px;
    }
    
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
    
    QRadioButton::indicator:unchecked {
        border: 1px solid #ced4da;
        background-color: #ffffff;
        border-radius: 8px;
    }
    
    QRadioButton::indicator:unchecked:hover {
        border-color: #adb5bd;
    }
    
    QRadioButton::indicator:checked {
        border: 1px solid #0078d7;
        background-color: #0078d7;
        border-radius: 8px;
    }
    
    QRadioButton::indicator:disabled {
        border: 1px solid #e9ecef;
        background-color: #f8f9fa;
    }
    
    /* ===== 滑块 ===== */
    QSlider::groove:horizontal {
        background-color: #e9ecef;
        height: 4px;
        border-radius: 2px;
    }
    
    QSlider::sub-page:horizontal {
        background-color: #0078d7;
        border-radius: 2px;
    }
    
    QSlider::add-page:horizontal {
        background-color: #e9ecef;
        border-radius: 2px;
    }
    
    QSlider::handle:horizontal {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }
    
    QSlider::handle:horizontal:hover {
        background-color: #f8f9fa;
        border-color: #adb5bd;
    }
    
    QSlider::groove:vertical {
        background-color: #e9ecef;
        width: 4px;
        border-radius: 2px;
    }
    
    QSlider::sub-page:vertical {
        background-color: #0078d7;
        border-radius: 2px;
    }
    
    QSlider::add-page:vertical {
        background-color: #e9ecef;
        border-radius: 2px;
    }
    
    QSlider::handle:vertical {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        width: 16px;
        height: 16px;
        margin: 0 -6px;
        border-radius: 8px;
    }
    
    QSlider::handle:vertical:hover {
        background-color: #f8f9fa;
        border-color: #adb5bd;
    }
    
    /* ===== 进度条 ===== */
    QProgressBar {
        background-color: #e9ecef;
        border: 1px solid #ced4da;
        border-radius: 3px;
        text-align: center;
    }
    
    QProgressBar::chunk {
        background-color: #0078d7;
        border-radius: 2px;
    }
    
    QProgressBar[class="success"]::chunk {
        background-color: #28a745;
    }
    
    QProgressBar[class="warning"]::chunk {
        background-color: #ffc107;
    }
    
    QProgressBar[class="danger"]::chunk {
        background-color: #dc3545;
    }
    
    /* ===== 表格 ===== */
    QTableWidget {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        gridline-color: #dee2e6;
        selection-background-color: #0078d7;
        selection-color: white;
        alternate-background-color: #f8f9fa;
    }
    
    QTableWidget::item {
        padding: 5px;
        border: none;
    }
    
    QTableWidget::item:selected {
        background-color: #0078d7;
        color: white;
    }
    
    QTableWidget QHeaderView::section {
        background-color: #f8f9fa;
        color: #495057;
        padding: 6px;
        border: 1px solid #dee2e6;
        border-left: none;
        border-top: none;
        font-weight: bold;
    }
    
    QTableWidget QHeaderView::section:first {
        border-left: 1px solid #dee2e6;
    }
    
    QTableWidget QHeaderView::section:hover {
        background-color: #e9ecef;
    }
    
    QTableWidget QHeaderView::section:pressed {
        background-color: #dee2e6;
    }
    
    /* ===== 特殊状态 ===== */
    .highlight {
        background-color: rgba(0, 120, 215, 0.1);
        border: 1px solid rgba(0, 120, 215, 0.3);
        border-radius: 3px;
    }
    
    .error-border {
        border: 1px solid #dc3545 !important;
    }
    
    .warning-border {
        border: 1px solid #ffc107 !important;
    }
    
    .success-border {
        border: 1px solid #28a745 !important;
    }
    
    .info-border {
        border: 1px solid #17a2b8 !important;
    }
    """

def get_industrial_theme():
    """工业主题样式表（专门为工业视觉系统设计）"""
    return """
    /* 工业视觉系统专用主题 */
    
    /* 主色调：工业蓝 */
    QMainWindow {
        background-color: #1a1a2e;
        border: 1px solid #16213e;
    }
    
    /* 控制面板 */
    QGroupBox {
        background-color: #16213e;
        border: 2px solid #0f3460;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 12px;
        font-weight: bold;
        color: #e6e6e6;
    }
    
    QGroupBox::title {
        color: #4fc3f7;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* 按钮 */
    QPushButton {
        background-color: #0f3460;
        border: 1px solid #1a5f7a;
        border-radius: 4px;
        padding: 8px 16px;
        color: #ffffff;
        font-weight: bold;
    }
    
    QPushButton:hover {
        background-color: #1a5f7a;
        border-color: #57c1ff;
    }
    
    QPushButton:pressed {
        background-color: #0a2640;
        border-color: #0f3460;
    }
    
    QPushButton[class="start"] {
        background-color: #00c853;
        border-color: #00a844;
        color: white;
    }
    
    QPushButton[class="stop"] {
        background-color: #ff3d00;
        border-color: #d32f00;
        color: white;
    }
    
    QPushButton[class="record"] {
        background-color: #f44336;
        border-color: #d32f2f;
        color: white;
    }
    
    /* 摄像头显示区域 */
    QFrame[class="camera-display"] {
        background-color: #000000;
        border: 3px solid #0f3460;
        border-radius: 4px;
    }
    
    /* 状态指示灯 */
    QLabel[class="status-indicator"] {
        background-color: #666666;
        border-radius: 7px;
        min-width: 14px;
        min-height: 14px;
        max-width: 14px;
        max-height: 14px;
    }
    
    QLabel[class="status-indicator-connected"] {
        background-color: #00e676;
    }
    
    QLabel[class="status-indicator-disconnected"] {
        background-color: #ff3d00;
    }
    
    QLabel[class="status-indicator-processing"] {
        background-color: #ffea00;
    }
    
    /* 数值显示 */
    QLabel[class="value-display"] {
        font-family: "Consolas", "Monaco", monospace;
        font-size: 12px;
        background-color: #0a2640;
        border: 1px solid #1a5f7a;
        border-radius: 3px;
        padding: 4px 8px;
        color: #4fc3f7;
    }
    
    /* 仪表盘样式 */
    QProgressBar[class="gauge"] {
        background-color: #0a2640;
        border: 2px solid #1a5f7a;
        border-radius: 8px;
        text-align: center;
        color: white;
    }
    
    QProgressBar[class="gauge"]::chunk {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                         stop:0 #00c853, stop:0.5 #ffea00, stop:1 #ff3d00);
        border-radius: 6px;
    }
    
    /* 工业字体 */
    QLabel, QPushButton, QComboBox, QLineEdit {
        font-family: "Segoe UI", "Arial", sans-serif;
    }
    
    QLabel[class="industrial-title"] {
        font-family: "Arial Black", "Segoe UI Black", sans-serif;
        font-size: 16px;
        color: #4fc3f7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 表格 */
    QTableWidget {
        background-color: #16213e;
        border: 1px solid #0f3460;
        gridline-color: #1a5f7a;
        selection-background-color: #4fc3f7;
        selection-color: #000000;
        alternate-background-color: #0f3460;
    }
    
    QTableWidget::item {
        color: #e6e6e6;
    }
    
    QTableWidget QHeaderView::section {
        background-color: #0a2640;
        color: #4fc3f7;
        border: 1px solid #1a5f7a;
        font-weight: bold;
    }
    
    /* 滑块 */
    QSlider::groove:horizontal {
        background-color: #0a2640;
        height: 6px;
        border-radius: 3px;
    }
    
    QSlider::sub-page:horizontal {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                         stop:0 #00c853, stop:0.5 #ffea00, stop:1 #ff3d00);
        border-radius: 3px;
    }
    
    QSlider::handle:horizontal {
        background-color: #ffffff;
        border: 2px solid #4fc3f7;
        width: 18px;
        height: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }
    
    /* 选项卡 */
    QTabWidget::pane {
        background-color: #16213e;
        border: 2px solid #0f3460;
        border-radius: 4px;
    }
    
    QTabBar::tab {
        background-color: #0a2640;
        border: 1px solid #1a5f7a;
        border-bottom: none;
        padding: 8px 16px;
        color: #aaaaaa;
    }
    
    QTabBar::tab:selected {
        background-color: #16213e;
        border-color: #0f3460;
        color: #4fc3f7;
        font-weight: bold;
    }
    
    QTabBar::tab:hover:!selected {
        background-color: #0f3460;
    }
    
    /* 动画效果 */
    QPushButton, QToolButton {
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    QPushButton:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    QPushButton:pressed {
        transform: translateY(0);
        box-shadow: none;
    }
    """

def get_status_style(status):
    """根据状态获取样式
    
    Args:
        status: 状态字符串
    
    Returns:
        str: 对应的样式类名
    """
    status_styles = {
        'normal': 'normal',
        'success': 'success',
        'warning': 'warning',
        'error': 'error',
        'info': 'info',
        'processing': 'processing',
        'connected': 'success',
        'disconnected': 'error',
        'running': 'success',
        'stopped': 'normal',
        'idle': 'info'
    }
    
    return status_styles.get(status.lower(), 'normal')

def get_button_size(size="medium"):
    """获取按钮尺寸样式
    
    Args:
        size: 尺寸类型
    
    Returns:
        str: 对应的尺寸样式
    """
    sizes = {
        'small': """
            QPushButton {
                padding: 4px 8px;
                font-size: 10px;
                min-height: 18px;
            }
        """,
        'medium': """
            QPushButton {
                padding: 6px 12px;
                font-size: 11px;
                min-height: 20px;
            }
        """,
        'large': """
            QPushButton {
                padding: 8px 16px;
                font-size: 13px;
                min-height: 24px;
            }
        """
    }
    
    return sizes.get(size, sizes['medium'])

def get_color_palette(theme="dark"):
    """获取颜色调色板
    
    Args:
        theme: 主题类型
    
    Returns:
        dict: 颜色字典
    """
    if theme == "dark":
        return {
            'primary': '#0078d7',
            'secondary': '#6c757d',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'background': '#2b2b2b',
            'foreground': '#e0e0e0',
            'border': '#555',
            'highlight': '#3c3c3c'
        }
    else:  # light theme
        return {
            'primary': '#0078d7',
            'secondary': '#6c757d',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'background': '#f5f5f5',
            'foreground': '#333',
            'border': '#ced4da',
            'highlight': '#ffffff'
        }

def get_font_settings():
    """获取字体设置"""
    return {
        'family': 'Microsoft YaHei, Segoe UI, Arial, sans-serif',
        'size_small': '10px',
        'size_normal': '11px',
        'size_large': '13px',
        'size_title': '14px',
        'size_heading': '16px',
        'weight_normal': 'normal',
        'weight_bold': 'bold',
        'monospace': 'Consolas, Monaco, monospace'
    }

if __name__ == "__main__":
    # 测试样式表
    print("深色主题示例:")
    print(get_dark_theme()[:500])
    print("\n...\n")
    
    print("浅色主题示例:")
    print(get_light_theme()[:500])
    print("\n...\n")
    
    print("工业主题示例:")
    print(get_industrial_theme()[:500])