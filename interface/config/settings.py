"""
工业视觉检测系统 - 配置管理器
负责应用程序设置的加载、保存和管理
"""

import json
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import yaml

from PyQt6.QtCore import QSettings, QObject, pyqtSignal


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ThemeMode(Enum):
    """主题模式枚举"""
    DARK = "dark"
    LIGHT = "light"
    INDUSTRIAL = "industrial"
    AUTO = "auto"


class CameraType(Enum):
    """摄像头类型枚举"""
    USB = "usb"
    INDUSTRIAL = "industrial"
    IP = "ip"
    VIRTUAL = "virtual"


class ModelFramework(Enum):
    """模型框架枚举"""
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    CUSTOM = "custom"


@dataclass
class CameraSettings:
    """摄像头设置"""
    camera_id: str = ""
    camera_type: CameraType = CameraType.USB
    device_id: int = 0
    resolution: List[int] = field(default_factory=lambda: [640, 480])
    fps: int = 30
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    exposure: float = 0.0
    gain: float = 0.0
    auto_exposure: bool = True
    auto_white_balance: bool = True
    gamma: float = 1.0
    focus: float = 0.0
    rtsp_url: str = ""
    ip_address: str = ""
    port: int = 554
    username: str = ""
    password: str = ""
    save_path: str = "captures"
    auto_save: bool = False
    save_interval: int = 60  # 秒


@dataclass
class ModelSettings:
    """模型设置"""
    model_id: str = ""
    model_path: str = ""
    model_framework: ModelFramework = ModelFramework.ONNX
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    batch_size: int = 1
    normalize_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    normalize_std: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    enable_nms: bool = True
    enable_tta: bool = False
    use_gpu: bool = True
    use_half: bool = False
    gpu_id: int = 0
    num_threads: int = 4
    warmup_iterations: int = 10


@dataclass
class AnalysisSettings:
    """分析设置"""
    realtime_analysis: bool = True
    auto_record_defects: bool = True
    defect_threshold: float = 0.7
    max_defects_per_frame: int = 10
    save_defect_images: bool = True
    defect_save_path: str = "defects"
    data_save_path: str = "data"
    export_format: str = "csv"  # csv, json, excel
    auto_export: bool = False
    export_interval: int = 3600  # 秒
    keep_history_days: int = 30
    max_history_records: int = 100000


@dataclass
class UISettings:
    """界面设置"""
    theme: ThemeMode = ThemeMode.DARK
    language: str = "zh_CN"
    font_family: str = "Microsoft YaHei"
    font_size: int = 11
    window_width: int = 1400
    window_height: int = 900
    window_maximized: bool = False
    show_toolbar: bool = True
    show_statusbar: bool = True
    show_grid: bool = False
    show_crosshair: bool = False
    show_fps: bool = True
    show_timestamp: bool = True
    display_mode: str = "original"  # original, grayscale, edges, binary
    zoom_factor: float = 1.0
    splitter_sizes: List[int] = field(default_factory=lambda: [900, 500])


@dataclass
class SystemSettings:
    """系统设置"""
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_path: str = "logs/industrial_vision.log"
    log_rotation: bool = True
    log_max_size: int = 100  # MB
    log_keep_days: int = 7
    check_for_updates: bool = True
    auto_save_settings: bool = True
    save_settings_interval: int = 300  # 秒
    backup_enabled: bool = True
    backup_path: str = "backups"
    backup_interval: int = 86400  # 秒
    backup_keep_count: int = 7
    use_hardware_acceleration: bool = True
    max_memory_usage: int = 4096  # MB


@dataclass
class IndustrialSettings:
    """工业设置"""
    production_line_id: str = "line_01"
    station_id: str = "station_01"
    product_type: str = "default"
    quality_standards: Dict[str, Any] = field(default_factory=dict)
    sampling_rate: float = 1.0  # 抽样率
    alarm_enabled: bool = True
    alarm_threshold: float = 0.05  # 缺陷率报警阈值
    auto_stop_on_high_defect: bool = False
    stop_threshold: float = 0.1  # 自动停止阈值
    calibration_enabled: bool = True
    calibration_interval: int = 3600  # 秒
    pixel_per_mm: float = 10.0
    reference_length: float = 10.0  # mm
    reference_pixels: int = 100


class Settings(QObject):
    """配置管理器类"""
    
    # 信号定义
    settings_changed = pyqtSignal(str)  # setting_name
    settings_loaded = pyqtSignal()
    settings_saved = pyqtSignal()
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self):
        """初始化配置管理器"""
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化默认设置
        self._init_default_settings()
        
        # Qt设置对象
        self.qt_settings = QSettings("IndustrialVision", "IndustrialVisionSystem")
        
        # 配置文件路径
        self.config_dir = self._get_config_dir()
        self.config_file = self.config_dir / "config.yaml"
        self.backup_dir = self.config_dir / "backups"
        
        # 确保目录存在
        self._ensure_directories()
        
        # 加载设置
        self.load_settings()
        
        self.logger.info("配置管理器初始化完成")
    
    def _get_config_dir(self) -> Path:
        """获取配置目录"""
        # 优先使用用户目录
        if sys.platform == "win32":
            config_dir = Path(os.environ.get("APPDATA", "")) / "IndustrialVision"
        elif sys.platform == "darwin":
            config_dir = Path.home() / "Library" / "Application Support" / "IndustrialVision"
        else:
            config_dir = Path.home() / ".config" / "IndustrialVision"
        
        # 如果用户目录不可用，使用程序目录
        if not config_dir.exists() and not config_dir.parent.exists():
            config_dir = Path.cwd() / "config"
        
        return config_dir
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"创建配置目录失败: {e}")
    
    def _init_default_settings(self):
        """初始化默认设置"""
        self.camera = CameraSettings()
        self.model = ModelSettings()
        self.analysis = AnalysisSettings()
        self.ui = UISettings()
        self.system = SystemSettings()
        self.industrial = IndustrialSettings()
        
        # 初始化质量标准
        self.industrial.quality_standards = {
            "defect_rate_max": 0.01,  # 最大缺陷率1%
            "min_confidence": 0.8,    # 最小置信度
            "max_inspection_time": 100,  # 最大检测时间(ms)
            "dimension_tolerance": 0.1,  # 尺寸公差(mm)
            "position_tolerance": 0.5,   # 位置公差(mm)
            "surface_quality": "good",   # 表面质量要求
        }
    
    def load_settings(self, config_file: Optional[Path] = None) -> bool:
        """加载设置"""
        try:
            if config_file is None:
                config_file = self.config_file
            
            self.logger.info(f"加载配置文件: {config_file}")
            
            # 如果配置文件不存在，创建默认配置
            if not config_file.exists():
                self.logger.warning("配置文件不存在，使用默认设置")
                self.save_settings()  # 保存默认设置
                return True
            
            # 加载YAML配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                self.logger.warning("配置文件为空，使用默认设置")
                return True
            
            # 更新设置
            self._update_from_dict(config_data)
            
            # 加载Qt设置（界面状态等）
            self._load_qt_settings()
            
            self.settings_loaded.emit()
            self.logger.info("配置加载成功")
            return True
            
        except yaml.YAMLError as e:
            error_msg = f"配置文件格式错误: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
            
        except Exception as e:
            error_msg = f"加载配置失败: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def save_settings(self, config_file: Optional[Path] = None) -> bool:
        """保存设置"""
        try:
            if config_file is None:
                config_file = self.config_file
            
            self.logger.info(f"保存配置文件: {config_file}")
            
            # 创建备份
            self._create_backup()
            
            # 转换为字典
            config_data = self._to_dict()
            
            # 保存为YAML格式
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            # 保存Qt设置
            self._save_qt_settings()
            
            self.settings_saved.emit()
            self.logger.info("配置保存成功")
            return True
            
        except Exception as e:
            error_msg = f"保存配置失败: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def _create_backup(self):
        """创建配置备份"""
        try:
            if not self.config_file.exists():
                return
            
            # 生成备份文件名
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"config_backup_{timestamp}.yaml"
            
            # 复制配置文件
            import shutil
            shutil.copy2(self.config_file, backup_file)
            
            # 清理旧备份
            self._cleanup_old_backups()
            
            self.logger.debug(f"配置备份已创建: {backup_file}")
            
        except Exception as e:
            self.logger.warning(f"创建配置备份失败: {e}")
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        try:
            backup_files = sorted(self.backup_dir.glob("config_backup_*.yaml"))
            
            # 保留最近N个备份
            keep_count = self.system.backup_keep_count
            if len(backup_files) > keep_count:
                for old_file in backup_files[:-keep_count]:
                    old_file.unlink()
                    self.logger.debug(f"删除旧备份: {old_file}")
                    
        except Exception as e:
            self.logger.warning(f"清理旧备份失败: {e}")
    
    def _load_qt_settings(self):
        """加载Qt设置（界面状态等）"""
        try:
            # 窗口几何状态
            geometry = self.qt_settings.value("window_geometry")
            if geometry:
                # 可以在主窗口中恢复
            
            # 窗口状态
                state = self.qt_settings.value("window_state")
            if state:
                # 可以在主窗口中恢复
                # 工具栏和状态栏状态
                show_toolbar = self.qt_settings.value("show_toolbar", True, type=bool)
                show_statusbar = self.qt_settings.value("show_statusbar", True, type=bool)
                # 界面设置
                theme_str = self.qt_settings.value("theme", "dark")
                self.ui.theme = ThemeMode(theme_str)
                
                language = self.qt_settings.value("language", "zh_CN")
                self.ui.language = language
                
                font_size = self.qt_settings.value("font_size", 11, type=int)
                self.ui.font_size = font_size
                
                self.logger.debug("Qt设置加载完成")
            
        except Exception as e:
            self.logger.warning(f"加载Qt设置失败: {e}")
    
    def _save_qt_settings(self):
        """保存Qt设置"""
        try:
            # 窗口状态由主窗口保存
            # 这里只保存需要持久化的设置
            
            self.qt_settings.setValue("theme", self.ui.theme.value)
            self.qt_settings.setValue("language", self.ui.language)
            self.qt_settings.setValue("font_size", self.ui.font_size)
            self.qt_settings.sync()
            
            self.logger.debug("Qt设置保存完成")
            
        except Exception as e:
            self.logger.warning(f"保存Qt设置失败: {e}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新设置"""
        try:
            # 更新摄像头设置
            if "camera" in config_dict:
                camera_dict = config_dict["camera"]
                if "camera_type" in camera_dict:
                    camera_dict["camera_type"] = CameraType(camera_dict["camera_type"])
                self._update_dataclass(self.camera, camera_dict)
            
            # 更新模型设置
            if "model" in config_dict:
                model_dict = config_dict["model"]
                if "model_framework" in model_dict:
                    model_dict["model_framework"] = ModelFramework(model_dict["model_framework"])
                self._update_dataclass(self.model, model_dict)
            
            # 更新分析设置
            if "analysis" in config_dict:
                self._update_dataclass(self.analysis, config_dict["analysis"])
            
            # 更新界面设置
            if "ui" in config_dict:
                ui_dict = config_dict["ui"]
                if "theme" in ui_dict:
                    ui_dict["theme"] = ThemeMode(ui_dict["theme"])
                self._update_dataclass(self.ui, ui_dict)
            
            # 更新系统设置
            if "system" in config_dict:
                system_dict = config_dict["system"]
                if "log_level" in system_dict:
                    system_dict["log_level"] = LogLevel(system_dict["log_level"])
                self._update_dataclass(self.system, system_dict)
            
            # 更新工业设置
            if "industrial" in config_dict:
                self._update_dataclass(self.industrial, config_dict["industrial"])
            
        except Exception as e:
            self.logger.error(f"从字典更新设置失败: {e}")
            raise
    
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]):
        """更新数据类的属性"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                # 如果是列表，确保类型正确
                current_value = getattr(dataclass_instance, key)
                if isinstance(current_value, list) and isinstance(value, list):
                    # 保持列表类型，但更新内容
                    setattr(dataclass_instance, key, value)
                else:
                    setattr(dataclass_instance, key, value)
    
    def _to_dict(self) -> Dict[str, Any]:
        """将设置转换为字典"""
        return {
            "camera": self._dataclass_to_dict(self.camera),
            "model": self._dataclass_to_dict(self.model),
            "analysis": self._dataclass_to_dict(self.analysis),
            "ui": self._dataclass_to_dict(self.ui),
            "system": self._dataclass_to_dict(self.system),
            "industrial": self._dataclass_to_dict(self.industrial),
            "version": "1.0.0",
            "last_modified": self._get_timestamp()
        }
    
    def _dataclass_to_dict(self, dataclass_instance) -> Dict[str, Any]:
        """将数据类实例转换为字典"""
        result = {}
        for key, value in asdict(dataclass_instance).items():
            # 处理枚举类型
            if isinstance(value, Enum):
                result[key] = value.value
            # 处理其他可序列化类型
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                result[key] = value
        return result
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_setting(self, category: str, key: str) -> Any:
        """获取特定设置值
        
        Args:
            category: 设置类别 (camera, model, analysis, ui, system, industrial)
            key: 设置键名
        
        Returns:
            设置值，如果不存在则返回None
        """
        try:
            if hasattr(self, category):
                category_obj = getattr(self, category)
                if hasattr(category_obj, key):
                    return getattr(category_obj, key)
            
            self.logger.warning(f"设置不存在: {category}.{key}")
            return None
            
        except Exception as e:
            self.logger.error(f"获取设置失败: {e}")
            return None
    
    def set_setting(self, category: str, key: str, value: Any) -> bool:
        """设置特定值
        
        Args:
            category: 设置类别
            key: 设置键名
            value: 设置值
        
        Returns:
            是否成功
        """
        try:
            if hasattr(self, category):
                category_obj = getattr(self, category)
                
                # 检查属性是否存在
                if not hasattr(category_obj, key):
                    self.logger.warning(f"设置不存在: {category}.{key}")
                    return False
                
                # 获取当前值以检查类型
                current_value = getattr(category_obj, key)
                
                # 类型检查
                if not self._check_type_compatibility(current_value, value):
                    self.logger.warning(f"类型不匹配: {category}.{key} ({type(current_value)} != {type(value)})")
                    return False
                
                # 设置新值
                setattr(category_obj, key, value)
                
                # 发出设置变化信号
                setting_path = f"{category}.{key}"
                self.settings_changed.emit(setting_path)
                
                self.logger.debug(f"设置已更新: {setting_path} = {value}")
                return True
            
            self.logger.warning(f"设置类别不存在: {category}")
            return False
            
        except Exception as e:
            self.logger.error(f"设置值失败: {e}")
            return False
    
    def _check_type_compatibility(self, current_value: Any, new_value: Any) -> bool:
        """检查类型兼容性"""
        # None值通常可以接受
        if new_value is None:
            return True
        
        # 枚举类型检查
        if isinstance(current_value, Enum):
            return isinstance(new_value, str)
        
        # 基本类型检查
        if isinstance(current_value, (int, float)):
            return isinstance(new_value, (int, float))
        
        if isinstance(current_value, bool):
            return isinstance(new_value, bool)
        
        if isinstance(current_value, str):
            return isinstance(new_value, str)
        
        if isinstance(current_value, list):
            return isinstance(new_value, list)
        
        if isinstance(current_value, dict):
            return isinstance(new_value, dict)
        
        return True
    
    def reset_to_defaults(self, category: Optional[str] = None) -> bool:
        """重置为默认设置
        
        Args:
            category: 可选，指定重置的类别，如果为None则重置所有
        
        Returns:
            是否成功
        """
        try:
            if category is None:
                # 重置所有设置
                self._init_default_settings()
                self.logger.info("所有设置已重置为默认值")
            else:
                # 重置特定类别
                if category == "camera":
                    self.camera = CameraSettings()
                elif category == "model":
                    self.model = ModelSettings()
                elif category == "analysis":
                    self.analysis = AnalysisSettings()
                elif category == "ui":
                    self.ui = UISettings()
                elif category == "system":
                    self.system = SystemSettings()
                elif category == "industrial":
                    self.industrial = IndustrialSettings()
                else:
                    self.logger.warning(f"未知的设置类别: {category}")
                    return False
                
                self.logger.info(f"{category} 设置已重置为默认值")
            
            # 发出设置变化信号
            self.settings_changed.emit("all" if category is None else category)
            
            # 保存更改
            self.save_settings()
            
            return True
            
        except Exception as e:
            self.logger.error(f"重置设置失败: {e}")
            return False
    
    def export_settings(self, export_path: Path) -> bool:
        """导出设置到文件
        
        Args:
            export_path: 导出文件路径
        
        Returns:
            是否成功
        """
        try:
            self.logger.info(f"导出设置到: {export_path}")
            
            # 确保目录存在
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 获取当前设置字典
            config_dict = self._to_dict()
            
            # 根据扩展名选择格式
            if export_path.suffix.lower() in ['.yaml', '.yml']:
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            elif export_path.suffix.lower() == '.json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                self.logger.warning(f"不支持的导出格式: {export_path.suffix}")
                return False
            
            self.logger.info("设置导出成功")
            return True
            
        except Exception as e:
            self.logger.error(f"导出设置失败: {e}")
            return False
    
    def import_settings(self, import_path: Path) -> bool:
        """从文件导入设置
        
        Args:
            import_path: 导入文件路径
        
        Returns:
            是否成功
        """
        try:
            if not import_path.exists():
                self.logger.error(f"导入文件不存在: {import_path}")
                return False
            
            self.logger.info(f"从文件导入设置: {import_path}")
            
            # 根据扩展名选择格式
            if import_path.suffix.lower() in ['.yaml', '.yml']:
                with open(import_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            elif import_path.suffix.lower() == '.json':
                with open(import_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            else:
                self.logger.warning(f"不支持的导入格式: {import_path.suffix}")
                return False
            
            # 更新设置
            self._update_from_dict(config_dict)
            
            # 保存到当前配置
            self.save_settings()
            
            self.logger.info("设置导入成功")
            return True
            
        except Exception as e:
            self.logger.error(f"导入设置失败: {e}")
            return False
    
    def validate_settings(self) -> Dict[str, List[str]]:
        """验证设置的完整性
        
        Returns:
            包含验证结果的字典
        """
        validation_result = {
            "warnings": [],
            "errors": []
        }
        
        try:
            # 验证摄像头设置
            if self.camera.camera_type == CameraType.IP and not self.camera.rtsp_url:
                validation_result["errors"].append("IP摄像头需要RTSP URL")
            
            if self.camera.camera_type == CameraType.IP and self.camera.rtsp_url:
                if not self.camera.rtsp_url.startswith(('rtsp://', 'http://', 'https://')):
                    validation_result["warnings"].append("RTSP URL格式可能不正确")
            
            # 验证模型设置
            if self.model.model_path and not Path(self.model.model_path).exists():
                validation_result["errors"].append(f"模型文件不存在: {self.model.model_path}")
            
            if self.model.confidence_threshold < 0 or self.model.confidence_threshold > 1:
                validation_result["errors"].append("置信度阈值必须在0-1之间")
            
            if self.model.iou_threshold < 0 or self.model.iou_threshold > 1:
                validation_result["errors"].append("IOU阈值必须在0-1之间")
            
            if self.model.input_width <= 0 or self.model.input_height <= 0:
                validation_result["errors"].append("输入尺寸必须大于0")
            
            # 验证分析设置
            if self.analysis.defect_threshold < 0 or self.analysis.defect_threshold > 1:
                validation_result["errors"].append("缺陷阈值必须在0-1之间")
            
            if self.analysis.max_defects_per_frame <= 0:
                validation_result["warnings"].append("每帧最大缺陷数应大于0")
            
            # 验证系统设置
            if self.system.log_max_size <= 0:
                validation_result["warnings"].append("日志最大大小应大于0")
            
            if self.system.max_memory_usage < 512:
                validation_result["warnings"].append("最大内存使用量建议不小于512MB")
            
            # 验证工业设置
            if self.industrial.pixel_per_mm <= 0:
                validation_result["errors"].append("像素/毫米值必须大于0")
            
            if self.industrial.alarm_threshold < 0 or self.industrial.alarm_threshold > 1:
                validation_result["errors"].append("报警阈值必须在0-1之间")
            
            if self.industrial.stop_threshold < 0 or self.industrial.stop_threshold > 1:
                validation_result["errors"].append("停止阈值必须在0-1之间")
            
            self.logger.debug(f"设置验证完成: {len(validation_result['warnings'])} 警告, {len(validation_result['errors'])} 错误")
            
        except Exception as e:
            self.logger.error(f"设置验证失败: {e}")
            validation_result["errors"].append(f"验证过程出错: {str(e)}")
        
        return validation_result
    
    def get_settings_summary(self) -> Dict[str, Any]:
        """获取设置摘要"""
        return {
            "camera": {
                "type": self.camera.camera_type.value,
                "resolution": self.camera.resolution,
                "fps": self.camera.fps
            },
            "model": {
                "framework": self.model.model_framework.value,
                "input_size": f"{self.model.input_width}x{self.model.input_height}",
                "confidence_threshold": self.model.confidence_threshold
            },
            "analysis": {
                "realtime": self.analysis.realtime_analysis,
                "defect_threshold": self.analysis.defect_threshold
            },
            "ui": {
                "theme": self.ui.theme.value,
                "language": self.ui.language
            },
            "system": {
                "log_level": self.system.log_level.value,
                "hardware_acceleration": self.system.use_hardware_acceleration
            },
            "industrial": {
                "production_line": self.industrial.production_line_id,
                "station": self.industrial.station_id,
                "pixel_per_mm": self.industrial.pixel_per_mm
            }
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            # 保存当前设置
            if self.system.auto_save_settings:
                self.save_settings()
            
            self.logger.info("配置管理器清理完成")
            
        except Exception as e:
            self.logger.error(f"配置管理器清理失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()


def create_default_config_file(config_path: Path) -> bool:
    """创建默认配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        是否成功
    """
    try:
        settings = Settings()
        
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存默认设置
        return settings.save_settings(config_path)
        
    except Exception as e:
        logging.error(f"创建默认配置文件失败: {e}")
        return False


if __name__ == "__main__":
    # 测试配置管理器
    logging.basicConfig(level=logging.INFO)
    
    print("测试配置管理器...")
    
    # 创建配置管理器
    settings = Settings()
    
    # 获取设置摘要
    summary = settings.get_settings_summary()
    print("设置摘要:", json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 验证设置
    validation = settings.validate_settings()
    print("验证结果:")
    print("  警告:", validation["warnings"])
    print("  错误:", validation["errors"])
    
    # 测试设置修改
    settings.set_setting("ui", "theme", ThemeMode.LIGHT.value)
    print("主题已修改为:", settings.ui.theme.value)
    
    # 导出设置
    test_export_path = Path("test_settings.yaml")
    if settings.export_settings(test_export_path):
        print(f"设置已导出到: {test_export_path}")
        test_export_path.unlink()  # 清理测试文件
    
    print("配置管理器测试完成")