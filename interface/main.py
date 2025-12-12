#!/usr/bin/env python3
"""
工业视觉检测系统 - 主程序入口
基于PyQt6的工业流水线视觉检测系统
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志系统
def setup_logging():
    """配置应用程序日志"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "industrial_vision.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# 导入PyQt6模块
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTranslator, QLocale
from PyQt6.QtGui import QFont, QPalette, QColor

# 导入应用程序模块
from UI.main_window import MainWindow
from config.settings import Settings

def setup_application_style(app):
    """设置应用程序的整体样式"""
    # 设置应用程序字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    # 设置应用程序调色板
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    # 设置高DPI支持
    # app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    # app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

def main():
    """应用程序主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("启动工业视觉检测系统")
    
    try:
        # 创建Qt应用程序实例
        app = QApplication(sys.argv)
        app.setApplicationName("Industrial Vision System")
        app.setApplicationDisplayName("工业视觉检测系统")
        app.setOrganizationName("Industrial Vision")
        app.setOrganizationDomain("industrial-vision.com")
        
        # 设置应用程序样式
        setup_application_style(app)
        
        # 加载翻译文件（如果存在）
        translator = QTranslator()
        locale = QLocale.system().name()
        translation_path = project_root / "resources" / "translations" / f"app_{locale}.qm"
        
        if translation_path.exists():
            if translator.load(str(translation_path)):
                app.installTranslator(translator)
                logger.info(f"加载翻译文件: {translation_path}")
        
        # 初始化设置
        settings = Settings()
        
        # 创建并显示主窗口
        logger.info("初始化主窗口")
        main_window = MainWindow(settings)
        main_window.show()
        
        # 启动应用程序事件循环
        logger.info("应用程序启动完成，进入事件循环")
        exit_code = app.exec()
        
        # 应用程序退出
        logger.info(f"应用程序退出，退出码: {exit_code}")
        return exit_code
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())