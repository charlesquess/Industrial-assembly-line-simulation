"""
工业视觉检测系统 - 摄像头管理器
负责摄像头设备的连接、控制和视频流管理
支持多种摄像头接口（USB摄像头、工业相机等）
"""

import logging
import threading
import time
import queue
import ctypes
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import cv2

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QMutex, QMutexLocker

@dataclass
class CameraInfo:
    """摄像头信息类"""
    id: str
    name: str
    device_id: int
    model: str
    vendor: str
    serial_number: str = ""
    is_connected: bool = False
    capabilities: Dict[str, Any] = field(default_factory=dict)
    current_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraSettings:
    """摄像头设置类"""
    resolution: Tuple[int, int] = (640, 480)
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


@dataclass
class FrameInfo:
    """帧信息类"""
    frame: np.ndarray
    timestamp: datetime
    frame_id: int
    camera_id: str
    frame_size: Tuple[int, int]
    frame_rate: float
    exposure_time: Optional[float] = None
    gain: Optional[float] = None


class CameraType(Enum):
    """摄像头类型枚举"""
    USB_CAMERA = "usb"
    INDUSTRIAL_CAMERA = "industrial"
    IP_CAMERA = "ip"
    VIRTUAL_CAMERA = "virtual"


class CameraError(Exception):
    """摄像头错误异常类"""
    pass


class BaseCamera:
    """摄像头基类"""
    
    def __init__(self, camera_id: str, camera_type: CameraType):
        self.camera_id = camera_id
        self.camera_type = camera_type
        self.is_connected = False
        self.is_capturing = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.frame_count = 0
        self.start_time = None
        self.logger = logging.getLogger(f"{__name__}.{camera_type.value}")
        
    def connect(self) -> bool:
        """连接摄像头"""
        raise NotImplementedError
    
    def disconnect(self) -> bool:
        """断开摄像头连接"""
        raise NotImplementedError
    
    def start_capture(self) -> bool:
        """开始捕获视频"""
        raise NotImplementedError
    
    def stop_capture(self) -> bool:
        """停止捕获视频"""
        raise NotImplementedError
    
    def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧图像"""
        raise NotImplementedError
    
    def get_info(self) -> CameraInfo:
        """获取摄像头信息"""
        raise NotImplementedError
    
    def get_settings(self) -> CameraSettings:
        """获取摄像头设置"""
        raise NotImplementedError
    
    def set_settings(self, settings: CameraSettings) -> bool:
        """设置摄像头参数"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """检查摄像头是否可用"""
        raise NotImplementedError


class USBCamera(BaseCamera):
    """USB摄像头类"""
    
    def __init__(self, camera_id: str, device_id: int = 0):
        super().__init__(camera_id, CameraType.USB_CAMERA)
        self.device_id = device_id
        self.cap = None
        self.current_settings = CameraSettings()
        self.camera_info = None
        self._capture_lock = threading.Lock()
        
    def connect(self) -> bool:
        """连接USB摄像头"""
        try:
            with self._capture_lock:
                if self.cap is not None:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
                
                if not self.cap.isOpened():
                    self.logger.error(f"无法打开摄像头 {self.device_id}")
                    return False
                
                # 获取摄像头信息
                self.camera_info = self._get_camera_info()
                self.is_connected = True
                
                # 设置默认参数
                self._apply_default_settings()
                
                self.logger.info(f"USB摄像头 {self.device_id} 连接成功")
                return True
                
        except Exception as e:
            self.logger.error(f"连接摄像头失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开USB摄像头连接"""
        try:
            self.stop_capture()
            
            with self._capture_lock:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                
                self.is_connected = False
                self.logger.info(f"USB摄像头 {self.device_id} 已断开")
                return True
                
        except Exception as e:
            self.logger.error(f"断开摄像头失败: {e}")
            return False
    
    def start_capture(self) -> bool:
        """开始捕获视频"""
        if not self.is_connected or self.cap is None:
            return False
        
        if self.is_capturing:
            return True
        
        try:
            self.is_capturing = True
            self.frame_count = 0
            self.start_time = time.time()
            
            # 启动捕获线程
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self.capture_thread.start()
            
            self.logger.info("开始视频捕获")
            return True
            
        except Exception as e:
            self.logger.error(f"启动视频捕获失败: {e}")
            self.is_capturing = False
            return False
    
    def stop_capture(self) -> bool:
        """停止捕获视频"""
        if not self.is_capturing:
            return True
        
        try:
            self.is_capturing = False
            
            # 等待捕获线程结束
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            # 清空帧队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("停止视频捕获")
            return True
            
        except Exception as e:
            self.logger.error(f"停止视频捕获失败: {e}")
            return False
    
    def get_frame(self) -> Optional[FrameInfo]:
        """获取一帧图像"""
        try:
            # 非阻塞获取帧
            frame_data = self.frame_queue.get_nowait()
            return frame_data
        except queue.Empty:
            return None
    
    def get_info(self) -> CameraInfo:
        """获取摄像头信息"""
        if self.camera_info is None:
            return CameraInfo(
                id=self.camera_id,
                name=f"USB Camera {self.device_id}",
                device_id=self.device_id,
                model="USB Camera",
                vendor="Unknown",
                is_connected=self.is_connected
            )
        return self.camera_info
    
    def get_settings(self) -> CameraSettings:
        """获取当前设置"""
        if not self.is_connected or self.cap is None:
            return self.current_settings
        
        try:
            # 获取当前参数
            self.current_settings.brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            self.current_settings.contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)
            self.current_settings.saturation = self.cap.get(cv2.CAP_PROP_SATURATION)
            self.current_settings.exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            self.current_settings.gain = self.cap.get(cv2.CAP_PROP_GAIN)
            self.current_settings.gamma = self.cap.get(cv2.CAP_PROP_GAMMA)
            
        except Exception as e:
            self.logger.warning(f"获取摄像头设置失败: {e}")
        
        return self.current_settings
    
    def set_settings(self, settings: CameraSettings) -> bool:
        """设置摄像头参数"""
        if not self.is_connected or self.cap is None:
            return False
        
        try:
            success = True
            
            # 设置分辨率
            if settings.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])
            
            # 设置帧率
            if settings.fps:
                self.cap.set(cv2.CAP_PROP_FPS, settings.fps)
            
            # 设置亮度
            if settings.brightness is not None:
                if not self.cap.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness):
                    self.logger.warning("无法设置亮度")
                    success = False
            
            # 设置对比度
            if settings.contrast is not None:
                if not self.cap.set(cv2.CAP_PROP_CONTRAST, settings.contrast):
                    self.logger.warning("无法设置对比度")
                    success = False
            
            # 设置饱和度
            if settings.saturation is not None:
                if not self.cap.set(cv2.CAP_PROP_SATURATION, settings.saturation):
                    self.logger.warning("无法设置饱和度")
                    success = False
            
            # 设置曝光
            if settings.exposure is not None:
                if not self.cap.set(cv2.CAP_PROP_EXPOSURE, settings.exposure):
                    self.logger.warning("无法设置曝光")
                    success = False
            
            # 设置增益
            if settings.gain is not None:
                if not self.cap.set(cv2.CAP_PROP_GAIN, settings.gain):
                    self.logger.warning("无法设置增益")
                    success = False
            
            # 设置伽马
            if settings.gamma is not None:
                if not self.cap.set(cv2.CAP_PROP_GAMMA, settings.gamma):
                    self.logger.warning("无法设置伽马")
                    success = False
            
            # 设置自动曝光
            if settings.auto_exposure is not None:
                # 注意：自动曝光的控制方式因摄像头而异
                # 这里是一个示例实现
                try:
                    if settings.auto_exposure:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动模式
                    else:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动模式
                except:
                    self.logger.warning("无法设置自动曝光")
            
            # 保存设置
            self.current_settings = settings
            
            if success:
                self.logger.info("摄像头设置已更新")
            else:
                self.logger.warning("部分摄像头设置更新失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"设置摄像头参数失败: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查摄像头是否可用"""
        # 尝试打开摄像头
        test_cap = None
        try:
            test_cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
            if test_cap.isOpened():
                test_cap.release()
                return True
            return False
        except:
            return False
        finally:
            if test_cap is not None:
                test_cap.release()
    
    def _capture_loop(self):
        """捕获循环"""
        while self.is_capturing:
            try:
                with self._capture_lock:
                    if self.cap is None:
                        break
                    
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        self.logger.warning("读取帧失败")
                        time.sleep(0.01)
                        continue
                
                # 计算帧率
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # 获取当前设置
                current_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                current_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 创建帧信息
                frame_info = FrameInfo(
                    frame=frame.copy(),
                    timestamp=datetime.now(),
                    frame_id=self.frame_count,
                    camera_id=self.camera_id,
                    frame_size=(current_width, current_height),
                    frame_rate=fps
                )
                
                # 将帧放入队列（如果队列已满，则丢弃最旧的帧）
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame_info)
                
                # 控制帧率
                target_fps = self.current_settings.fps
                if target_fps > 0:
                    time.sleep(1.0 / target_fps)
                    
            except Exception as e:
                self.logger.error(f"捕获循环错误: {e}")
                time.sleep(0.1)
    
    def _get_camera_info(self) -> CameraInfo:
        """获取摄像头信息"""
        if self.cap is None:
            return CameraInfo(
                id=self.camera_id,
                name=f"USB Camera {self.device_id}",
                device_id=self.device_id,
                model="Unknown",
                vendor="Unknown",
                is_connected=False
            )
        
        try:
            # 获取摄像头能力信息
            capabilities = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
                'format': self.cap.get(cv2.CAP_PROP_FORMAT),
                'mode': self.cap.get(cv2.CAP_PROP_MODE)
            }
            
            return CameraInfo(
                id=self.camera_id,
                name=f"USB Camera {self.device_id}",
                device_id=self.device_id,
                model="USB Camera",
                vendor="Unknown",
                is_connected=True,
                capabilities=capabilities
            )
            
        except Exception as e:
            self.logger.warning(f"获取摄像头信息失败: {e}")
            return CameraInfo(
                id=self.camera_id,
                name=f"USB Camera {self.device_id}",
                device_id=self.device_id,
                model="Unknown",
                vendor="Unknown",
                is_connected=True
            )
    
    def _apply_default_settings(self):
        """应用默认设置"""
        if self.cap is None:
            return
        
        try:
            # 设置默认分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_settings.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_settings.resolution[1])
            
            # 设置默认帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.current_settings.fps)
            
            # 应用其他设置
            self.set_settings(self.current_settings)
            
        except Exception as e:
            self.logger.warning(f"应用默认设置失败: {e}")


class IndustrialCamera(BaseCamera):
    """工业相机类（示例实现）"""
    
    def __init__(self, camera_id: str):
        super().__init__(camera_id, CameraType.INDUSTRIAL_CAMERA)
        self.logger.warning("工业相机支持需要安装相应的SDK，这里仅提供示例实现")
    
    def connect(self) -> bool:
        self.is_connected = True
        return True
    
    def disconnect(self) -> bool:
        self.is_connected = False
        return True
    
    def start_capture(self) -> bool:
        self.is_capturing = True
        return True
    
    def stop_capture(self) -> bool:
        self.is_capturing = False
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        # 返回测试图像
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_info(self) -> CameraInfo:
        return CameraInfo(
            id=self.camera_id,
            name="Industrial Camera",
            device_id=0,
            model="Example Industrial Camera",
            vendor="Example Vendor",
            is_connected=self.is_connected
        )
    
    def get_settings(self) -> CameraSettings:
        return CameraSettings()
    
    def set_settings(self, settings: CameraSettings) -> bool:
        return True
    
    def is_available(self) -> bool:
        return False


class IPCamera(BaseCamera):
    """IP摄像头类"""
    
    def __init__(self, camera_id: str, rtsp_url: str):
        super().__init__(camera_id, CameraType.IP_CAMERA)
        self.rtsp_url = rtsp_url
    
    def connect(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.is_connected = self.cap.isOpened()
            return self.is_connected
        except Exception as e:
            self.logger.error(f"连接IP摄像头失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        return True
    
    def start_capture(self) -> bool:
        # IP摄像头通常不需要特殊的启动过程
        self.is_capturing = True
        return True
    
    def stop_capture(self) -> bool:
        self.is_capturing = False
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_capturing or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def get_info(self) -> CameraInfo:
        return CameraInfo(
            id=self.camera_id,
            name="IP Camera",
            device_id=0,
            model="IP Camera",
            vendor="Unknown",
            serial_number="",
            is_connected=self.is_connected
        )
    
    def get_settings(self) -> CameraSettings:
        return CameraSettings()
    
    def set_settings(self, settings: CameraSettings) -> bool:
        # IP摄像头通常不支持设置参数
        return False
    
    def is_available(self) -> bool:
        # 测试连接
        test_cap = cv2.VideoCapture(self.rtsp_url)
        if test_cap.isOpened():
            test_cap.release()
            return True
        return False


class CameraManager(QObject):
    """摄像头管理器"""
    
    # 信号定义
    frame_received = pyqtSignal(object)  # 发送帧数据 (FrameInfo 或 dict)
    camera_connected = pyqtSignal(str, object)  # camera_id, CameraInfo
    camera_disconnected = pyqtSignal(str)  # camera_id
    camera_error = pyqtSignal(str, str)  # camera_id, error_message
    camera_list_updated = pyqtSignal(list)  # List[CameraInfo]
    camera_settings_changed = pyqtSignal(dict)  # 摄像头设置变化
    capture_started = pyqtSignal()  # 捕获开始信号
    
    def __init__(self):
        """初始化摄像头管理器"""
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # 摄像头字典
        self.cameras: Dict[str, BaseCamera] = {}
        self.current_camera: Optional[BaseCamera] = None
        
        # 帧处理线程
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self._process_frames)
        self.frame_timer.start(16)  # 约60Hz
        
        # 摄像头发现
        self.discovery_timer = QTimer()
        self.discovery_timer.timeout.connect(self._discover_cameras)
        self.discovery_timer.start(5000)  # 每5秒扫描一次
        
        # 互斥锁
        self.camera_lock = QMutex()
        
        self.logger.info("摄像头管理器初始化完成")
    
    def discover_cameras(self) -> List[CameraInfo]:
        """发现可用摄像头"""
        camera_list = []
        
        try:
            # 检测USB摄像头
            for i in range(0, 10):  # 检查前10个设备
                camera_id = f"usb_{i}"
                
                # 检查是否已存在
                if camera_id in self.cameras:
                    camera = self.cameras[camera_id]
                    if camera.is_available():
                        camera_list.append(camera.get_info())
                    continue
                
                # 创建并测试摄像头
                camera = USBCamera(camera_id, i)
                if camera.is_available():
                    self.cameras[camera_id] = camera
                    camera_list.append(camera.get_info())
            
            # 检测IP摄像头（示例）
            # 这里可以添加配置文件中的IP摄像头
            
            # 检测工业相机（需要SDK支持）
            # 这里可以添加工业相机的检测逻辑
            
        except Exception as e:
            self.logger.error(f"摄像头发现错误: {e}")
        
        # 发出信号
        self.camera_list_updated.emit(camera_list)
        
        return camera_list
    
    def connect_camera(self, camera_id: str, camera_type: str = "usb", **kwargs) -> bool:
        """连接摄像头"""
        try:
            with QMutexLocker(self.camera_lock):
                # 确保 camera_id 是字符串类型
                camera_id = str(camera_id)
                
                # 检查摄像头是否已存在
                if camera_id in self.cameras:
                    camera = self.cameras[camera_id]
                else:
                    # 创建摄像头实例
                    if camera_type == "usb":
                        device_id = kwargs.get("device_id", 0)
                        camera = USBCamera(camera_id, device_id)
                    elif camera_type == "industrial":
                        camera = IndustrialCamera(camera_id)
                    elif camera_type == "ip":
                        rtsp_url = kwargs.get("rtsp_url", "")
                        camera = IPCamera(camera_id, rtsp_url)
                    else:
                        raise CameraError(f"不支持的摄像头类型: {camera_type}")
                    
                    self.cameras[camera_id] = camera
                
                # 连接摄像头
                if camera.connect():
                    self.current_camera = camera
                    
                    # 获取摄像头信息
                    camera_info = camera.get_info()  # 获取的是 CameraInfo 对象
                    
                    # 发出连接信号
                    self.camera_connected.emit(camera_id, camera_info)  # 确保 camera_id 是字符串类型
                    
                    self.logger.info(f"摄像头 {camera_id} 连接成功")
                    return True
                else:
                    self.logger.error(f"摄像头 {camera_id} 连接失败")
                    return False
                    
        except Exception as e:
            error_msg = f"连接摄像头失败: {str(e)}"
            self.logger.error(error_msg)
            self.camera_error.emit(camera_id, error_msg)
            return False
                    
        except Exception as e:
            error_msg = f"连接摄像头失败: {str(e)}"
            self.logger.error(error_msg)
            self.camera_error.emit(camera_id, error_msg)
            return False
    
    def disconnect_camera(self, camera_id: Optional[str] = None) -> bool:
        """断开摄像头连接"""
        try:
            with QMutexLocker(self.camera_lock):
                if camera_id is None and self.current_camera:
                    camera_id = self.current_camera.camera_id
                
                if camera_id in self.cameras:
                    camera = self.cameras[camera_id]
                    
                    # 停止捕获
                    camera.stop_capture()
                    
                    # 断开连接
                    success = camera.disconnect()
                    
                    # 发出断开信号
                    self.camera_disconnected.emit(camera_id)
                    
                    # 如果断开的是当前摄像头，清除引用
                    if self.current_camera and self.current_camera.camera_id == camera_id:
                        self.current_camera = None
                    
                    self.logger.info(f"摄像头 {camera_id} 已断开")
                    return success
                else:
                    self.logger.warning(f"找不到摄像头: {camera_id}")
                    return False
                    
        except Exception as e:
            error_msg = f"断开摄像头失败: {str(e)}"
            self.logger.error(error_msg)
            self.camera_error.emit(camera_id or "", error_msg)
            return False
    
    def start_capture(self) -> bool:
        """开始捕获视频"""
        try:
            with QMutexLocker(self.camera_lock):
                if not self.current_camera or not self.current_camera.is_connected:
                    self.logger.warning("没有已连接的摄像头")
                    return False
                
                success = self.current_camera.start_capture()
                if success:
                    self.logger.info("视频捕获已开始")
                else:
                    self.logger.error("视频捕获启动失败")
                
                return success
                
        except Exception as e:
            self.logger.error(f"开始捕获失败: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """停止捕获视频"""
        try:
            with QMutexLocker(self.camera_lock):
                if not self.current_camera:
                    return True
                
                success = self.current_camera.stop_capture()
                if success:
                    self.logger.info("视频捕获已停止")
                else:
                    self.logger.error("视频捕获停止失败")
                
                return success
                
        except Exception as e:
            self.logger.error(f"停止捕获失败: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获单帧图像"""
        try:
            with QMutexLocker(self.camera_lock):
                if not self.current_camera or not self.current_camera.is_connected:
                    return None
                
                # 对于USB摄像头，使用队列获取最新帧
                if isinstance(self.current_camera, USBCamera):
                    frame_info = self.current_camera.get_frame()
                    if frame_info:
                        return frame_info.frame
                
                # 对于其他摄像头类型，直接读取
                return self.current_camera.get_frame()
                
        except Exception as e:
            self.logger.error(f"捕获单帧失败: {e}")
            return None
    
    def get_camera_info(self, camera_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取摄像头信息（返回字典格式）"""
        try:
            with QMutexLocker(self.camera_lock):
                if camera_id is None and self.current_camera:
                    camera_id = self.current_camera.camera_id
                
                if camera_id in self.cameras:
                    camera = self.cameras[camera_id]
                    info = camera.get_info()
                    
                    # 转换为字典格式
                    info_dict = {
                        'id': info.id,
                        'name': info.name,
                        'device_id': info.device_id,
                        'model': info.model,
                        'vendor': info.vendor,
                        'serial_number': info.serial_number,
                        'is_connected': info.is_connected,
                        'capabilities': info.capabilities,
                        'current_settings': info.current_settings
                    }
                    
                    # 添加额外的信息
                    if hasattr(info, 'capabilities') and isinstance(info.capabilities, dict):
                        info_dict.update(info.capabilities)
                    
                    return info_dict
                
                return None
                
        except Exception as e:
            self.logger.error(f"获取摄像头信息失败: {e}")
            return None
    
    def get_camera_settings(self) -> Optional[CameraSettings]:
        """获取当前摄像头设置"""
        try:
            with QMutexLocker(self.camera_lock):
                if not self.current_camera:
                    return None
                
                return self.current_camera.get_settings()
                
        except Exception as e:
            self.logger.error(f"获取摄像头设置失败: {e}")
            return None
    
    def set_camera_settings(self, settings: CameraSettings) -> bool:
        """设置摄像头参数"""
        try:
            with QMutexLocker(self.camera_lock):
                if not self.current_camera:
                    return False
                
                success = self.current_camera.set_settings(settings)
                
                if success:
                    self.logger.info("摄像头设置已更新")
                else:
                    self.logger.warning("摄像头设置更新失败")
                
                return success
                
        except Exception as e:
            self.logger.error(f"设置摄像头参数失败: {e}")
            return False
    
    def is_camera_connected(self) -> bool:
        """检查摄像头是否已连接"""
        with QMutexLocker(self.camera_lock):
            return self.current_camera is not None and self.current_camera.is_connected
    
    def is_capturing(self) -> bool:
        """检查是否正在捕获视频"""
        with QMutexLocker(self.camera_lock):
            return self.current_camera is not None and self.current_camera.is_capturing
    
    def get_available_cameras(self) -> List[Dict[str, Any]]:
        """获取可用摄像头列表"""
        camera_list = []
        
        try:
            # 检测USB摄像头
            for i in range(0, 10):
                camera_id = f"usb_{i}"
                camera = USBCamera(camera_id, i)
                
                if camera.is_available():
                    info = camera.get_info()
                    camera_list.append({
                        'id': camera_id,
                        'name': info.name,
                        'type': 'usb',
                        'device_id': i,
                        'model': info.model,
                        'vendor': info.vendor
                    })
            
            self.logger.info(f"发现 {len(camera_list)} 个可用摄像头")
            
        except Exception as e:
            self.logger.error(f"获取可用摄像头列表失败: {e}")
        
        return camera_list
    
    def select_camera(self, camera_id: str) -> bool:
        """选择摄像头"""
        try:
            with QMutexLocker(self.camera_lock):
                if camera_id not in self.cameras:
                    self.logger.warning(f"摄像头 {camera_id} 不存在")
                    return False
                
                # 如果已选择其他摄像头，先断开
                if self.current_camera and self.current_camera.camera_id != camera_id:
                    self.disconnect_camera()
                
                self.current_camera = self.cameras[camera_id]
                self.logger.info(f"已选择摄像头: {camera_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"选择摄像头失败: {e}")
            return False
    
    def save_frame(self, filepath: str, frame: Optional[np.ndarray] = None) -> bool:
        """保存当前帧到文件"""
        try:
            if frame is None:
                frame = self.capture_frame()
            
            if frame is None:
                self.logger.warning("没有可保存的帧")
                return False
            
            success = cv2.imwrite(filepath, frame)
            
            if success:
                self.logger.info(f"帧已保存到: {filepath}")
            else:
                self.logger.error("保存帧失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"保存帧失败: {e}")
            return False
    
    def get_frame_statistics(self) -> Dict[str, Any]:
        """获取帧统计信息"""
        try:
            with QMutexLocker(self.camera_lock):
                if not self.current_camera:
                    return {}
                
                frame = self.capture_frame()
                if frame is None:
                    return {}
                
                # 计算统计信息
                height, width = frame.shape[:2]
                channels = 1 if len(frame.shape) == 2 else frame.shape[2]
                
                # 计算直方图
                if channels == 1:
                    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
                    hist_mean = np.mean(hist)
                    hist_std = np.std(hist)
                else:
                    hist_mean = 0
                    hist_std = 0
                    for i in range(channels):
                        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
                        hist_mean += np.mean(hist)
                        hist_std += np.std(hist)
                    hist_mean /= channels
                    hist_std /= channels
                
                # 计算图像质量指标
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if channels > 1 else frame
                blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                stats = {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'dtype': str(frame.dtype),
                    'size_bytes': frame.nbytes,
                    'histogram_mean': float(hist_mean),
                    'histogram_std': float(hist_std),
                    'blur_value': float(blur_value),
                    'frame_mean': float(np.mean(frame)),
                    'frame_std': float(np.std(frame)),
                    'frame_min': float(np.min(frame)),
                    'frame_max': float(np.max(frame))
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"获取帧统计信息失败: {e}")
            return {}
    
    def _process_frames(self):
        """处理帧数据"""
        try:
            if not self.current_camera:
                self.logger.debug("当前没有活动的摄像头")
                return
            
            if not self.current_camera.is_capturing:
                self.logger.debug("摄像头未在捕获状态")
                return
            
            # 获取帧
            if isinstance(self.current_camera, USBCamera):
                frame_info = self.current_camera.get_frame()
                if frame_info:
                    self.logger.debug(f"从USB摄像头获取到帧: {frame_info.frame.shape if frame_info.frame is not None else 'None'}")
                    
                    # 构建帧信息字典
                    frame_data = {
                        'frame': frame_info.frame,
                        'timestamp': frame_info.timestamp,
                        'frame_id': frame_info.frame_id,
                        'camera_id': frame_info.camera_id,
                        'width': frame_info.frame_size[0],
                        'height': frame_info.frame_size[1],
                        'fps': frame_info.frame_rate
                    }
                    
                    # 发射信号
                    self.frame_received.emit(frame_data)
                else:
                    self.logger.debug("从USB摄像头获取帧失败或队列为空")
            else:
                # 处理其他类型的摄像头
                frame = self.current_camera.get_frame()
                if frame is not None:
                    self.logger.debug(f"从摄像头获取到帧: {frame.shape}")
                    
                    frame_data = {
                        'frame': frame,
                        'timestamp': datetime.now(),
                        'frame_id': -1,
                        'camera_id': self.current_camera.camera_id,
                        'width': frame.shape[1],
                        'height': frame.shape[0],
                        'fps': 0
                    }
                    
                    self.frame_received.emit(frame_data)
                else:
                    self.logger.debug("从摄像头获取帧失败")
                    
        except Exception as e:
            self.logger.error(f"处理帧数据错误: {e}", exc_info=True)
    
    def _discover_cameras(self):
        """定期发现摄像头"""
        try:
            self.discover_cameras()
        except Exception as e:
            self.logger.error(f"摄像头发现错误: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("正在清理摄像头管理器...")
        
        # 停止定时器
        if self.frame_timer.isActive():
            self.frame_timer.stop()
        
        if self.discovery_timer.isActive():
            self.discovery_timer.stop()
        
        # 断开所有摄像头
        for camera_id in list(self.cameras.keys()):
            self.disconnect_camera(camera_id)
        
        self.logger.info("摄像头管理器清理完成")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()