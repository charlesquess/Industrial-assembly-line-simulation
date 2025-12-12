"""
工业视觉检测系统 - 模型管理器
负责深度学习模型的加载、推理和管理
支持多种模型格式和推理后端
"""

import logging
import json
import os
import sys
import time
import threading
import queue
import traceback
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
import hashlib
import copy

import numpy as np
import cv2

# 尝试导入深度学习框架
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("警告: ONNX Runtime 未安装，ONNX模型支持将被禁用")

try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch 未安装，PyTorch模型支持将被禁用")

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("警告: TensorFlow 未安装，TensorFlow模型支持将被禁用")

try:
    import openvino.runtime as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("警告: OpenVINO 未安装，OpenVINO模型支持将被禁用")

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QMutex, QMutexLocker

@dataclass
class ModelInfo:
    """模型信息类"""
    model_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    format: str = ""  # onnx, torch, tensorflow, openvino
    framework: str = ""  # pytorch, tensorflow, onnx, etc.
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    num_classes: int = 0
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    batch_size: int = 1
    precision: str = "fp32"  # fp32, fp16, int8
    created_date: str = ""
    modified_date: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    file_size: int = 0
    is_loaded: bool = False
    requires_preprocessing: bool = True
    requires_postprocessing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """从字典创建"""
        return cls(**data)


@dataclass
class ModelConfig:
    """模型配置类"""
    # 推理参数
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    batch_size: int = 1
    input_width: int = 640
    input_height: int = 640
    
    # 预处理参数
    normalize_mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    normalize_std: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    resize_method: str = "letterbox"  # letterbox, stretch, crop
    
    # 后处理参数
    enable_nms: bool = True
    enable_tta: bool = False  # 测试时增强
    max_detections: int = 100
    
    # 硬件参数
    use_gpu: bool = True
    use_half: bool = False
    gpu_device_id: int = 0
    num_threads: int = 4
    
    # 性能参数
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建"""
        return cls(**data)


@dataclass
class InferenceResult:
    """推理结果类"""
    # 基础信息
    success: bool = False
    error_message: str = ""
    inference_time: float = 0.0
    preprocess_time: float = 0.0
    postprocess_time: float = 0.0
    
    # 检测结果
    detections: List[Dict[str, Any]] = field(default_factory=list)
    num_detections: int = 0
    
    # 分类结果
    classifications: List[Dict[str, Any]] = field(default_factory=list)
    
    # 分割结果
    segmentation_masks: Optional[np.ndarray] = None
    
    # 性能统计
    memory_usage: float = 0.0  # MB
    fps: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 原始输出
    raw_output: Optional[List[np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        if self.segmentation_masks is not None:
            result['segmentation_masks'] = self.segmentation_masks.tolist()
        if self.raw_output is not None:
            result['raw_output'] = [arr.tolist() for arr in self.raw_output]
        return result


@dataclass
class ModelPerformance:
    """模型性能统计类"""
    model_id: str
    inference_times: List[float] = field(default_factory=list)
    memory_usages: List[float] = field(default_factory=list)
    fps_values: List[float] = field(default_factory=list)
    accuracy_values: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    
    def update(self, inference_time: float, memory_usage: float, success: bool = True):
        """更新性能统计"""
        self.inference_times.append(inference_time)
        self.memory_usages.append(memory_usage)
        
        if inference_time > 0:
            self.fps_values.append(1000.0 / inference_time)
        
        self.total_inferences += 1
        if success:
            self.successful_inferences += 1
        else:
            self.failed_inferences += 1
        
        # 保持历史数据在合理大小
        max_history = 1000
        if len(self.inference_times) > max_history:
            self.inference_times = self.inference_times[-max_history:]
            self.memory_usages = self.memory_usages[-max_history:]
            self.fps_values = self.fps_values[-max_history:]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'min_inference_time': 0.0,
                'max_inference_time': 0.0,
                'avg_fps': 0.0,
                'avg_memory_usage': 0.0,
                'success_rate': 0.0,
                'total_inferences': self.total_inferences
            }
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'avg_fps': np.mean(self.fps_values),
            'avg_memory_usage': np.mean(self.memory_usages),
            'success_rate': self.successful_inferences / max(1, self.total_inferences),
            'total_inferences': self.total_inferences
        }


class ModelFormat(Enum):
    """模型格式枚举"""
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"
    UNKNOWN = "unknown"


class ModelTask(Enum):
    """模型任务类型枚举"""
    OBJECT_DETECTION = "object_detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    POSE_ESTIMATION = "pose_estimation"
    ANOMALY_DETECTION = "anomaly_detection"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class ModelError(Exception):
    """模型错误异常类"""
    pass


class BaseModel:
    """模型基类"""
    
    def __init__(self, model_info: ModelInfo, config: ModelConfig):
        """初始化模型
        
        Args:
            model_info: 模型信息
            config: 模型配置
        """
        self.model_info = model_info
        self.config = config
        self.is_loaded = False
        self.logger = logging.getLogger(f"{__name__}.{model_info.model_id}")
        
        # 推理会话
        self.session = None
        self.device = None
        
        # 性能监控
        self.performance = ModelPerformance(model_info.model_id)
        
    def load(self) -> bool:
        """加载模型"""
        raise NotImplementedError
    
    def unload(self) -> bool:
        """卸载模型"""
        raise NotImplementedError
    
    def inference(self, images: List[np.ndarray]) -> InferenceResult:
        """执行推理"""
        raise NotImplementedError
    
    def preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """预处理图像"""
        raise NotImplementedError
    
    def postprocess(self, raw_output: List[np.ndarray], original_shapes: List[Tuple[int, int]]) -> InferenceResult:
        """后处理原始输出"""
        raise NotImplementedError
    
    def warmup(self, iterations: int = 10) -> bool:
        """预热模型"""
        try:
            test_image = np.random.randn(self.config.input_height, 
                                        self.config.input_width, 3).astype(np.float32)
            
            for i in range(iterations):
                result = self.inference([test_image])
                if not result.success:
                    return False
            
            self.logger.info(f"模型预热完成 ({iterations} 次迭代)")
            return True
            
        except Exception as e:
            self.logger.error(f"模型预热失败: {e}")
            return False
    
    def benchmark(self, iterations: int = 100) -> Dict[str, Any]:
        """性能基准测试"""
        try:
            test_image = np.random.randn(self.config.input_height,
                                        self.config.input_width, 3).astype(np.float32)
            
            inference_times = []
            
            for i in range(iterations):
                start_time = time.perf_counter()
                result = self.inference([test_image])
                end_time = time.perf_counter()
                
                if not result.success:
                    return {"error": "推理失败"}
                
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            stats = {
                'iterations': iterations,
                'avg_time_ms': np.mean(inference_times),
                'min_time_ms': np.min(inference_times),
                'max_time_ms': np.max(inference_times),
                'std_time_ms': np.std(inference_times),
                'fps': 1000.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0.0
            }
            
            self.logger.info(f"性能基准测试完成: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"性能基准测试失败: {e}")
            return {"error": str(e)}
    
    def get_memory_usage(self) -> float:
        """获取内存使用情况（MB）"""
        if not HAS_TORCH:
            return 0.0
        
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            pass
        
        return 0.0


class ONNXModel(BaseModel):
    """ONNX模型类"""
    
    def __init__(self, model_info: ModelInfo, config: ModelConfig):
        super().__init__(model_info, config)
        
        # ONNX运行时选项
        self.providers = []
        self.provider_options = []
        
    def load(self) -> bool:
        """加载ONNX模型"""
        if not HAS_ONNX:
            self.logger.error("ONNX Runtime 未安装")
            return False
        
        try:
            # 检查模型文件
            if not os.path.exists(self.model_info.file_path):
                self.logger.error(f"模型文件不存在: {self.model_info.file_path}")
                return False
            
            # 验证ONNX模型
            try:
                onnx_model = onnx.load(self.model_info.file_path)
                onnx.checker.check_model(onnx_model)
                self.logger.info("ONNX模型验证通过")
            except Exception as e:
                self.logger.warning(f"ONNX模型验证失败: {e}")
            
            # 配置推理提供者
            if self.config.use_gpu and ort.get_device() == 'GPU':
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.provider_options = [{
                    'device_id': self.config.gpu_device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }]
            else:
                self.providers = ['CPUExecutionProvider']
                self.provider_options = [{
                    'intra_op_num_threads': self.config.num_threads,
                    'inter_op_num_threads': self.config.num_threads,
                }]
            
            # 创建推理会话
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads
            sess_options.inter_op_num_threads = self.config.num_threads
            
            self.session = ort.InferenceSession(
                self.model_info.file_path,
                sess_options=sess_options,
                providers=self.providers,
                provider_options=self.provider_options
            )
            
            # 获取输入输出信息
            self._get_io_info()
            
            self.is_loaded = True
            self.model_info.is_loaded = True
            
            self.logger.info(f"ONNX模型加载成功: {self.model_info.name}")
            
            # 预热模型
            self.warmup(self.config.warmup_iterations)
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载ONNX模型失败: {e}")
            traceback.print_exc()
            return False
    
    def unload(self) -> bool:
        """卸载ONNX模型"""
        try:
            self.session = None
            self.is_loaded = False
            self.model_info.is_loaded = False
            
            self.logger.info("ONNX模型已卸载")
            return True
            
        except Exception as e:
            self.logger.error(f"卸载ONNX模型失败: {e}")
            return False
    
    def inference(self, images: List[np.ndarray]) -> InferenceResult:
        """执行ONNX推理"""
        if not self.is_loaded or self.session is None:
            return InferenceResult(
                success=False,
                error_message="模型未加载"
            )
        
        try:
            # 记录开始时间
            total_start = time.perf_counter()
            preprocess_start = total_start
            
            # 预处理
            processed_images = self.preprocess(images)
            original_shapes = [(img.shape[1], img.shape[0]) for img in images]
            
            preprocess_end = time.perf_counter()
            inference_start = preprocess_end
            
            # 准备输入
            input_feed = {}
            for i, input_name in enumerate(self.model_info.input_names):
                # 堆叠批处理图像
                if len(processed_images) > 1:
                    batch_input = np.stack(processed_images, axis=0)
                else:
                    batch_input = processed_images[0][np.newaxis, ...]
                
                input_feed[input_name] = batch_input.astype(np.float32)
            
            # 执行推理
            raw_output = self.session.run(
                self.model_info.output_names,
                input_feed
            )
            
            inference_end = time.perf_counter()
            postprocess_start = inference_end
            
            # 后处理
            result = self.postprocess(raw_output, original_shapes)
            
            postprocess_end = time.perf_counter()
            
            # 计算时间
            result.preprocess_time = (preprocess_end - preprocess_start) * 1000
            result.inference_time = (inference_end - inference_start) * 1000
            result.postprocess_time = (postprocess_end - postprocess_start) * 1000
            result.success = True
            
            # 计算内存使用
            result.memory_usage = self.get_memory_usage()
            
            # 计算FPS
            if result.inference_time > 0:
                result.fps = 1000.0 / result.inference_time
            
            # 更新性能统计
            self.performance.update(
                inference_time=result.inference_time,
                memory_usage=result.memory_usage,
                success=True
            )
            
            return result
            
        except Exception as e:
            error_msg = f"推理失败: {str(e)}"
            self.logger.error(error_msg)
            
            # 更新性能统计
            self.performance.update(
                inference_time=0.0,
                memory_usage=0.0,
                success=False
            )
            
            return InferenceResult(
                success=False,
                error_message=error_msg
            )
    
    def preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """预处理图像"""
        processed_images = []
        
        for image in images:
            # 转换为RGB
            if len(image.shape) == 2:  # 灰度图
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:  # 单通道
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # 调整大小
            original_height, original_width = image.shape[:2]
            
            if self.config.resize_method == "letterbox":
                # 保持宽高比的调整大小
                scale = min(self.config.input_width / original_width,
                          self.config.input_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                resized = cv2.resize(image, (new_width, new_height), 
                                   interpolation=cv2.INTER_LINEAR)
                
                # 填充到目标尺寸
                top = (self.config.input_height - new_height) // 2
                bottom = self.config.input_height - new_height - top
                left = (self.config.input_width - new_width) // 2
                right = self.config.input_width - new_width - left
                
                padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
            else:  # 拉伸
                padded = cv2.resize(image, (self.config.input_width, 
                                          self.config.input_height),
                                  interpolation=cv2.INTER_LINEAR)
            
            # 归一化
            if self.config.normalize_mean and self.config.normalize_std:
                padded = padded.astype(np.float32) / 255.0
                padded = (padded - self.config.normalize_mean) / self.config.normalize_std
            
            # 调整通道顺序 (HWC -> CHW)
            if padded.shape[2] == 3:
                padded = padded.transpose(2, 0, 1)
            
            processed_images.append(padded)
        
        return processed_images
    
    def postprocess(self, raw_output: List[np.ndarray], original_shapes: List[Tuple[int, int]]) -> InferenceResult:
        """后处理原始输出"""
        result = InferenceResult()
        result.raw_output = raw_output
        
        # 根据模型任务类型进行不同的后处理
        task_type = self.model_info.metadata.get('task_type', 'object_detection')
        
        if task_type == 'object_detection':
            result = self._postprocess_detection(raw_output, original_shapes)
        elif task_type == 'classification':
            result = self._postprocess_classification(raw_output)
        elif task_type == 'segmentation':
            result = self._postprocess_segmentation(raw_output, original_shapes)
        else:
            self.logger.warning(f"未知的任务类型: {task_type}")
            result.detections = []
        
        result.num_detections = len(result.detections)
        return result
    
    def _postprocess_detection(self, raw_output: List[np.ndarray], original_shapes: List[Tuple[int, int]]) -> InferenceResult:
        """后处理目标检测结果"""
        result = InferenceResult()
        
        if not raw_output:
            return result
        
        # 假设第一个输出是检测结果
        detections = raw_output[0]
        
        for i, detection in enumerate(detections):
            # 解析检测结果 (YOLO格式: [batch, num_detections, 85])
            # 85 = [x, y, w, h, conf, class_conf_0, ..., class_conf_n]
            if len(detection) >= 6:  # 至少包含[x, y, w, h, conf, class_id]
                x, y, w, h, conf = detection[:5]
                class_id = int(np.argmax(detection[5:])) if len(detection) > 5 else 0
                class_conf = detection[5 + class_id] if len(detection) > 5 else conf
                
                # 过滤低置信度检测
                total_conf = conf * class_conf
                if total_conf < self.config.confidence_threshold:
                    continue
                
                # 转换为原始图像坐标
                original_width, original_height = original_shapes[0]
                scale_x = original_width / self.config.input_width
                scale_y = original_height / self.config.input_height
                
                # 计算边界框
                x1 = int((x - w / 2) * scale_x)
                y1 = int((y - h / 2) * scale_y)
                x2 = int((x + w / 2) * scale_x)
                y2 = int((y + h / 2) * scale_y)
                
                # 确保边界框在图像范围内
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                x2 = max(0, min(x2, original_width - 1))
                y2 = max(0, min(y2, original_height - 1))
                
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0:
                    continue
                
                # 获取类别名称
                class_name = ""
                if 0 <= class_id < len(self.model_info.classes):
                    class_name = self.model_info.classes[class_id]
                
                # 添加到检测结果
                result.detections.append({
                    'bbox': [x1, y1, width, height],
                    'confidence': float(total_conf),
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': [int((x1 + x2) // 2), int((y1 + y2) // 2)],
                    'area': width * height
                })
        
        # 应用非极大值抑制
        if self.config.enable_nms and result.detections:
            result.detections = self._apply_nms(result.detections)
        
        return result
    
    def _postprocess_classification(self, raw_output: List[np.ndarray]) -> InferenceResult:
        """后处理分类结果"""
        result = InferenceResult()
        
        if not raw_output:
            return result
        
        # 假设第一个输出是分类概率
        probs = raw_output[0][0]  # 取第一个批次的第一个输出
        
        # 获取Top-K预测
        k = min(5, len(probs))
        top_k_indices = np.argsort(probs)[-k:][::-1]
        
        for idx in top_k_indices:
            confidence = float(probs[idx])
            
            # 过滤低置信度
            if confidence < self.config.confidence_threshold:
                continue
            
            # 获取类别名称
            class_name = ""
            if 0 <= idx < len(self.model_info.classes):
                class_name = self.model_info.classes[idx]
            
            result.classifications.append({
                'class_id': int(idx),
                'class_name': class_name,
                'confidence': confidence
            })
        
        return result
    
    def _postprocess_segmentation(self, raw_output: List[np.ndarray], original_shapes: List[Tuple[int, int]]) -> InferenceResult:
        """后处理分割结果"""
        result = InferenceResult()
        
        if not raw_output:
            return result
        
        # 假设第一个输出是分割掩码
        mask = raw_output[0][0]  # 取第一个批次的第一个输出
        
        # 调整掩码大小到原始图像尺寸
        original_width, original_height = original_shapes[0]
        resized_mask = cv2.resize(mask, (original_width, original_height),
                                interpolation=cv2.INTER_NEAREST)
        
        result.segmentation_masks = resized_mask
        
        return result
    
    def _apply_nms(self, detections: List[Dict[str, Any]], iou_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """应用非极大值抑制"""
        if not detections:
            return []
        
        if iou_threshold is None:
            iou_threshold = self.config.iou_threshold
        
        # 按置信度排序
        sorted_detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        
        keep = []
        
        while sorted_detections:
            # 取置信度最高的检测
            best = sorted_detections.pop(0)
            keep.append(best)
            
            # 计算与剩余检测的IoU
            to_remove = []
            for i, detection in enumerate(sorted_detections):
                iou = self._calculate_iou(best['bbox'], detection['bbox'])
                if iou > iou_threshold:
                    to_remove.append(i)
            
            # 移除重叠的检测
            for i in reversed(to_remove):
                sorted_detections.pop(i)
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_io_info(self):
        """获取输入输出信息"""
        if self.session is None:
            return
        
        # 获取输入信息
        self.model_info.input_names = [input.name for input in self.session.get_inputs()]
        self.model_info.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取输入形状
        input_info = self.session.get_inputs()[0]
        self.model_info.input_shape = list(input_info.shape)
        
        # 获取输出形状
        output_info = self.session.get_outputs()[0]
        self.model_info.output_shape = list(output_info.shape)
        
        # 从metadata中获取类别信息
        if 'classes' in self.model_info.metadata:
            self.model_info.classes = self.model_info.metadata['classes']
            self.model_info.num_classes = len(self.model_info.classes)
        
        self.logger.info(f"输入名称: {self.model_info.input_names}, 形状: {self.model_info.input_shape}")
        self.logger.info(f"输出名称: {self.model_info.output_names}, 形状: {self.model_info.output_shape}")


class PyTorchModel(BaseModel):
    """PyTorch模型类"""
    
    def __init__(self, model_info: ModelInfo, config: ModelConfig):
        super().__init__(model_info, config)
        
        if not HAS_TORCH:
            raise ModelError("PyTorch 未安装")
        
        self.model = None
        self.device = None
        
    def load(self) -> bool:
        """加载PyTorch模型"""
        try:
            # 检查模型文件
            if not os.path.exists(self.model_info.file_path):
                self.logger.error(f"模型文件不存在: {self.model_info.file_path}")
                return False
            
            # 设置设备
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.config.gpu_device_id}')
                torch.cuda.set_device(self.config.gpu_device_id)
            else:
                self.device = torch.device('cpu')
                self.config.use_gpu = False
            
            self.logger.info(f"使用设备: {self.device}")
            
            # 加载模型
            checkpoint = torch.load(self.model_info.file_path, map_location=self.device)
            
            if 'model' in checkpoint:
                # 包含模型状态字典的检查点
                model_state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # PyTorch Lightning检查点
                model_state_dict = checkpoint['state_dict']
            else:
                # 直接是模型状态字典
                model_state_dict = checkpoint
            
            # 创建模型（这里需要根据具体的模型架构来创建）
            # 这是一个示例，实际使用时需要根据模型类型创建
            self.model = self._create_model(model_state_dict)
            
            if self.model is None:
                self.logger.error("无法创建模型实例")
                return False
            
            # 加载权重
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            # 设置半精度
            if self.config.use_half:
                self.model.half()
                torch.backends.cudnn.benchmark = True
            
            self.is_loaded = True
            self.model_info.is_loaded = True
            
            self.logger.info(f"PyTorch模型加载成功: {self.model_info.name}")
            
            # 预热模型
            self.warmup(self.config.warmup_iterations)
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载PyTorch模型失败: {e}")
            traceback.print_exc()
            return False
    
    def _create_model(self, state_dict):
        """根据状态字典创建模型实例"""
        # 这是一个示例实现，实际使用时需要根据具体的模型架构来创建
        # 可以从metadata中获取模型类型信息
        
        model_type = self.model_info.metadata.get('model_type', '')
        
        try:
            if model_type == 'yolov5':
                # 导入YOLOv5模型
                from models.yolo import Model
                cfg = self.model_info.metadata.get('cfg', 'yolov5s.yaml')
                model = Model(cfg)
                return model
            elif model_type == 'resnet':
                # 使用ResNet
                num_classes = self.model_info.num_classes or 1000
                model = torchvision.models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                return model
            else:
                # 默认尝试创建一个简单的CNN
                self.logger.warning(f"未知的模型类型: {model_type}, 使用默认模型")
                return self._create_default_model(state_dict)
                
        except Exception as e:
            self.logger.error(f"创建模型失败: {e}")
            return None
    
    def _create_default_model(self, state_dict):
        """创建默认模型（示例）"""
        class SimpleCNN(torch.nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
                self.fc1 = torch.nn.Linear(64 * 6 * 6, 128)
                self.fc2 = torch.nn.Linear(128, num_classes)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.max_pool2d(x, 2)
                x = torch.relu(self.conv2(x))
                x = torch.max_pool2d(x, 2)
                x = torch.flatten(x, 1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # 从状态字典推断输入通道数
        for key in state_dict.keys():
            if 'conv1.weight' in key:
                in_channels = state_dict[key].shape[1]
                break
        
        num_classes = self.model_info.num_classes or 10
        model = SimpleCNN(num_classes)
        return model
    
    def inference(self, images: List[np.ndarray]) -> InferenceResult:
        """执行PyTorch推理"""
        if not self.is_loaded or self.model is None:
            return InferenceResult(
                success=False,
                error_message="模型未加载"
            )
        
        try:
            # 记录开始时间
            total_start = time.perf_counter()
            preprocess_start = total_start
            
            # 预处理
            processed_images = self.preprocess(images)
            original_shapes = [(img.shape[1], img.shape[0]) for img in images]
            
            preprocess_end = time.perf_counter()
            inference_start = preprocess_end
            
            # 转换为张量
            with torch.no_grad():
                # 堆叠批处理图像
                if len(processed_images) > 1:
                    batch_tensor = torch.stack([torch.from_numpy(img) for img in processed_images])
                else:
                    batch_tensor = torch.from_numpy(processed_images[0]).unsqueeze(0)
                
                # 移动到设备
                batch_tensor = batch_tensor.to(self.device)
                
                if self.config.use_half:
                    batch_tensor = batch_tensor.half()
                
                # 执行推理
                output = self.model(batch_tensor)
                
                # 转换为numpy数组
                if isinstance(output, torch.Tensor):
                    raw_output = [output.cpu().numpy()]
                elif isinstance(output, (list, tuple)):
                    raw_output = [t.cpu().numpy() for t in output]
                else:
                    raw_output = []
            
            inference_end = time.perf_counter()
            postprocess_start = inference_end
            
            # 后处理
            result = self.postprocess(raw_output, original_shapes)
            
            postprocess_end = time.perf_counter()
            
            # 计算时间
            result.preprocess_time = (preprocess_end - preprocess_start) * 1000
            result.inference_time = (inference_end - inference_start) * 1000
            result.postprocess_time = (postprocess_end - postprocess_start) * 1000
            result.success = True
            
            # 计算内存使用
            result.memory_usage = self.get_memory_usage()
            
            # 计算FPS
            if result.inference_time > 0:
                result.fps = 1000.0 / result.inference_time
            
            # 更新性能统计
            self.performance.update(
                inference_time=result.inference_time,
                memory_usage=result.memory_usage,
                success=True
            )
            
            return result
            
        except Exception as e:
            error_msg = f"推理失败: {str(e)}"
            self.logger.error(error_msg)
            
            # 更新性能统计
            self.performance.update(
                inference_time=0.0,
                memory_usage=0.0,
                success=False
            )
            
            return InferenceResult(
                success=False,
                error_message=error_msg
            )
    
    def preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """预处理图像（与ONNX相同）"""
        # 这里可以复用ONNX的预处理逻辑
        onnx_model = ONNXModel(self.model_info, self.config)
        return onnx_model.preprocess(images)
    
    def postprocess(self, raw_output: List[np.ndarray], original_shapes: List[Tuple[int, int]]) -> InferenceResult:
        """后处理原始输出（与ONNX相同）"""
        # 这里可以复用ONNX的后处理逻辑
        onnx_model = ONNXModel(self.model_info, self.config)
        return onnx_model.postprocess(raw_output, original_shapes)


class ModelManager(QObject):
    """模型管理器"""
    
    # 信号定义
    model_loaded = pyqtSignal(str, object)  # model_id, ModelInfo
    model_unloaded = pyqtSignal(str)  # model_id
    inference_completed = pyqtSignal(object)  # InferenceResult
    performance_updated = pyqtSignal(str, object)  # model_id, ModelPerformance
    model_error = pyqtSignal(str, str)  # model_id, error_message
    model_list_updated = pyqtSignal(list)  # List[ModelInfo]
    
    def __init__(self, models_dir: str = "models"):
        """初始化模型管理器
        
        Args:
            models_dir: 模型文件目录
        """
        super().__init__()
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 模型字典
        self.models: Dict[str, BaseModel] = {}
        self.model_infos: Dict[str, ModelInfo] = {}
        self.current_model: Optional[BaseModel] = None
        
        # 配置
        self.default_config = ModelConfig()
        
        # 模型数据库文件
        self.db_file = self.models_dir / "models.json"
        
        # 互斥锁
        self.model_lock = QMutex()
        
        # 推理队列
        self.inference_queue = queue.Queue()
        self.inference_thread = None
        self.is_inference_running = False
        
        # 性能监控定时器
        self.performance_timer = QTimer()
        self.performance_timer.timeout.connect(self._update_performance)
        self.performance_timer.start(1000)  # 每秒更新一次
        
        # 加载模型数据库
        self._load_model_database()
        
        self.logger.info(f"模型管理器初始化完成，模型目录: {self.models_dir}")
    
    def discover_models(self) -> List[ModelInfo]:
        """发现可用模型"""
        model_files = []
        
        # 搜索模型文件
        for ext in ['.onnx', '.pt', '.pth', '.pb', '.xml', '.tflite', '.engine']:
            model_files.extend(self.models_dir.glob(f"**/*{ext}"))
        
        discovered = []
        
        for model_file in model_files:
            try:
                # 检查是否已在数据库中
                model_id = self._generate_model_id(model_file)
                
                if model_id in self.model_infos:
                    model_info = self.model_infos[model_id]
                else:
                    # 创建新的模型信息
                    model_info = self._create_model_info(model_file)
                    self.model_infos[model_id] = model_info
                
                discovered.append(model_info)
                
            except Exception as e:
                self.logger.error(f"处理模型文件 {model_file} 失败: {e}")
        
        # 保存更新后的数据库
        self._save_model_database()
        
        # 发出信号
        self.model_list_updated.emit(discovered)
        
        return discovered
    
    def load_model(self, model_path: str, model_id: Optional[str] = None) -> bool:
        """加载模型"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 生成模型ID
            if model_id is None:
                model_id = self._generate_model_id(model_path)
            
            with QMutexLocker(self.model_lock):
                # 检查是否已加载
                if model_id in self.models:
                    self.logger.info(f"模型 {model_id} 已加载")
                    self.current_model = self.models[model_id]
                    return True
                
                # 获取或创建模型信息
                if model_id in self.model_infos:
                    model_info = self.model_infos[model_id]
                else:
                    model_info = self._create_model_info(model_path)
                    self.model_infos[model_id] = model_info
                
                # 确定模型格式
                model_format = self._detect_model_format(model_path)
                
                # 创建模型配置
                config = ModelConfig()
                
                # 根据模型信息更新配置
                if model_info.metadata:
                    if 'confidence_threshold' in model_info.metadata:
                        config.confidence_threshold = model_info.metadata['confidence_threshold']
                    if 'input_size' in model_info.metadata:
                        if isinstance(model_info.metadata['input_size'], list) and len(model_info.metadata['input_size']) >= 2:
                            config.input_width = model_info.metadata['input_size'][0]
                            config.input_height = model_info.metadata['input_size'][1]
                
                # 创建模型实例
                model = None
                
                if model_format == ModelFormat.ONNX:
                    if HAS_ONNX:
                        model = ONNXModel(model_info, config)
                    else:
                        raise ModelError("ONNX Runtime 未安装")
                elif model_format == ModelFormat.PYTORCH:
                    if HAS_TORCH:
                        model = PyTorchModel(model_info, config)
                    else:
                        raise ModelError("PyTorch 未安装")
                else:
                    raise ModelError(f"不支持的模型格式: {model_format}")
                
                # 加载模型
                if model and model.load():
                    self.models[model_id] = model
                    self.current_model = model
                    
                    # 更新模型信息
                    model_info.is_loaded = True
                    model_info.modified_date = datetime.now().isoformat()
                    
                    # 保存数据库
                    self._save_model_database()
                    
                    # 发出信号
                    self.model_loaded.emit(model_id, model_info)
                    
                    self.logger.info(f"模型加载成功: {model_info.name}")
                    return True
                else:
                    self.logger.error(f"模型加载失败: {model_info.name}")
                    return False
                    
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            self.logger.error(error_msg)
            self.model_error.emit(model_id or "", error_msg)
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """卸载模型"""
        try:
            with QMutexLocker(self.model_lock):
                if model_id not in self.models:
                    self.logger.warning(f"模型未加载: {model_id}")
                    return False
                
                model = self.models[model_id]
                
                # 停止推理
                if self.current_model == model:
                    self.stop_inference()
                    self.current_model = None
                
                # 卸载模型
                success = model.unload()
                
                # 从字典中移除
                del self.models[model_id]
                
                # 更新模型信息
                if model_id in self.model_infos:
                    self.model_infos[model_id].is_loaded = False
                
                # 发出信号
                self.model_unloaded.emit(model_id)
                
                self.logger.info(f"模型已卸载: {model_id}")
                return success
                
        except Exception as e:
            error_msg = f"卸载模型失败: {str(e)}"
            self.logger.error(error_msg)
            self.model_error.emit(model_id, error_msg)
            return False
    
    def inference_single_image(self, image_path: str) -> Optional[InferenceResult]:
        """单张图像推理"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法读取图像: {image_path}")
                return None
            
            # 执行推理
            return self.inference([image])
            
        except Exception as e:
            self.logger.error(f"单张图像推理失败: {e}")
            return None
    
    def inference_batch_images(self, image_paths: List[str]) -> List[InferenceResult]:
        """批量图像推理"""
        results = []
        
        for image_path in image_paths:
            result = self.inference_single_image(image_path)
            results.append(result)
        
        return results
    
    def inference(self, images: List[np.ndarray]) -> Optional[InferenceResult]:
        """执行推理"""
        try:
            with QMutexLocker(self.model_lock):
                if not self.current_model:
                    self.logger.warning("没有加载的模型")
                    return None
                
                # 执行推理
                result = self.current_model.inference(images)
                
                # 发出信号
                self.inference_completed.emit(result)
                
                return result
                
        except Exception as e:
            error_msg = f"推理失败: {str(e)}"
            self.logger.error(error_msg)
            return InferenceResult(success=False, error_message=error_msg)
    
    def start_inference_stream(self, frame_callback) -> bool:
        """启动推理流"""
        try:
            self.is_inference_running = True
            
            # 启动推理线程
            self.inference_thread = threading.Thread(
                target=self._inference_loop,
                args=(frame_callback,),
                daemon=True
            )
            self.inference_thread.start()
            
            self.logger.info("推理流已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动推理流失败: {e}")
            return False
    
    def stop_inference(self) -> bool:
        """停止推理"""
        try:
            self.is_inference_running = False
            
            # 等待推理线程结束
            if self.inference_thread and self.inference_thread.is_alive():
                self.inference_thread.join(timeout=2.0)
            
            # 清空推理队列
            while not self.inference_queue.empty():
                try:
                    self.inference_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("推理已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止推理失败: {e}")
            return False
    
    def add_image_to_queue(self, image: np.ndarray):
        """添加图像到推理队列"""
        if self.is_inference_running:
            try:
                self.inference_queue.put(image, block=False)
            except queue.Full:
                # 队列已满，丢弃最旧的图像
                try:
                    self.inference_queue.get_nowait()
                    self.inference_queue.put(image, block=False)
                except:
                    pass
    
    def set_inference_parameters(self, parameters: Dict[str, Any]) -> bool:
        """设置推理参数"""
        try:
            with QMutexLocker(self.model_lock):
                if not self.current_model:
                    return False
                
                # 更新配置
                for key, value in parameters.items():
                    if hasattr(self.current_model.config, key):
                        setattr(self.current_model.config, key, value)
                
                self.logger.info("推理参数已更新")
                return True
                
        except Exception as e:
            self.logger.error(f"设置推理参数失败: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.model_infos.get(model_id)
    
    def get_current_model(self) -> Optional[ModelInfo]:
        """获取当前模型信息"""
        if not self.current_model:
            return None
        
        return self.current_model.model_info
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        models = []
        
        for model_id, model_info in self.model_infos.items():
            models.append({
                'id': model_id,
                'name': model_info.name,
                'description': model_info.description,
                'format': model_info.format,
                'loaded': model_info.is_loaded,
                'classes': model_info.classes,
                'file_size': model_info.file_size,
                'created_date': model_info.created_date
            })
        
        return models
    
    def add_model(self, model_path: str, copy_to_dir: bool = True) -> bool:
        """添加模型到管理器"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                self.logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 生成模型ID
            model_id = self._generate_model_id(model_path)
            
            # 检查是否已存在
            if model_id in self.model_infos:
                self.logger.info(f"模型已存在: {model_id}")
                return True
            
            # 复制到模型目录
            if copy_to_dir:
                dest_path = self.models_dir / model_path.name
                shutil.copy2(model_path, dest_path)
                model_path = dest_path
            
            # 创建模型信息
            model_info = self._create_model_info(model_path)
            self.model_infos[model_id] = model_info
            
            # 保存数据库
            self._save_model_database()
            
            # 更新模型列表
            self.discover_models()
            
            self.logger.info(f"模型添加成功: {model_info.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加模型失败: {e}")
            return False
    
    def remove_model(self, model_id: str, delete_file: bool = False) -> bool:
        """移除模型"""
        try:
            with QMutexLocker(self.model_lock):
                if model_id not in self.model_infos:
                    self.logger.warning(f"模型不存在: {model_id}")
                    return False
                
                # 如果模型已加载，先卸载
                if model_id in self.models:
                    self.unload_model(model_id)
                
                # 删除文件
                if delete_file:
                    model_info = self.model_infos[model_id]
                    if model_info.file_path and os.path.exists(model_info.file_path):
                        os.remove(model_info.file_path)
                
                # 从字典中移除
                del self.model_infos[model_id]
                
                # 保存数据库
                self._save_model_database()
                
                self.logger.info(f"模型移除成功: {model_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"移除模型失败: {e}")
            return False
    
    def benchmark_model(self, model_id: str, iterations: int = 100) -> Dict[str, Any]:
        """性能基准测试"""
        try:
            with QMutexLocker(self.model_lock):
                if model_id not in self.models:
                    self.logger.warning(f"模型未加载: {model_id}")
                    return {"error": "模型未加载"}
                
                model = self.models[model_id]
                return model.benchmark(iterations)
                
        except Exception as e:
            self.logger.error(f"性能测试失败: {e}")
            return {"error": str(e)}
    
    def get_performance_stats(self, model_id: str) -> Dict[str, Any]:
        """获取性能统计"""
        try:
            with QMutexLocker(self.model_lock):
                if model_id not in self.models:
                    return {}
                
                model = self.models[model_id]
                return model.performance.get_summary()
                
        except Exception as e:
            self.logger.error(f"获取性能统计失败: {e}")
            return {}
    
    def is_model_loaded(self, model_id: Optional[str] = None) -> bool:
        """检查模型是否已加载"""
        with QMutexLocker(self.model_lock):
            if model_id:
                return model_id in self.models and self.models[model_id].is_loaded
            else:
                return self.current_model is not None and self.current_model.is_loaded
    
    def is_inference_running(self) -> bool:
        """检查推理是否正在运行"""
        return self.is_inference_running
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("正在清理模型管理器...")
        
        # 停止推理
        self.stop_inference()
        
        # 停止定时器
        if self.performance_timer.isActive():
            self.performance_timer.stop()
        
        # 卸载所有模型
        for model_id in list(self.models.keys()):
            self.unload_model(model_id)
        
        self.logger.info("模型管理器清理完成")
    
    def _inference_loop(self, frame_callback):
        """推理循环"""
        while self.is_inference_running:
            try:
                # 从队列获取图像
                try:
                    image = self.inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 执行推理
                result = self.inference([image])
                
                # 调用回调函数
                if result and result.success and frame_callback:
                    frame_callback(image, result)
                    
            except Exception as e:
                self.logger.error(f"推理循环错误: {e}")
    
    def _update_performance(self):
        """更新性能统计"""
        try:
            with QMutexLocker(self.model_lock):
                for model_id, model in self.models.items():
                    summary = model.performance.get_summary()
                    self.performance_updated.emit(model_id, summary)
                    
        except Exception as e:
            self.logger.error(f"更新性能统计失败: {e}")
    
    def _generate_model_id(self, model_path: Path) -> str:
        """生成模型ID（基于文件哈希）"""
        # 计算文件哈希
        hasher = hashlib.md5()
        with open(model_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        
        file_hash = hasher.hexdigest()[:8]
        filename = model_path.stem
        
        return f"{filename}_{file_hash}"
    
    def _detect_model_format(self, model_path: Path) -> ModelFormat:
        """检测模型格式"""
        suffix = model_path.suffix.lower()
        
        if suffix in ['.onnx']:
            return ModelFormat.ONNX
        elif suffix in ['.pt', '.pth']:
            return ModelFormat.PYTORCH
        elif suffix in ['.pb', '.tflite']:
            return ModelFormat.TENSORFLOW
        elif suffix in ['.xml']:
            return ModelFormat.OPENVINO
        elif suffix in ['.engine']:
            return ModelFormat.TENSORRT
        else:
            return ModelFormat.UNKNOWN
    
    def _create_model_info(self, model_path: Path) -> ModelInfo:
        """创建模型信息"""
        # 获取文件信息
        stat = model_path.stat()
        
        # 尝试读取模型元数据
        metadata_file = model_path.with_suffix('.json')
        metadata = {}
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"读取模型元数据失败: {e}")
        
        # 生成模型ID
        model_id = self._generate_model_id(model_path)
        
        # 从metadata中提取信息
        name = metadata.get('name', model_path.stem)
        description = metadata.get('description', '')
        version = metadata.get('version', '1.0.0')
        classes = metadata.get('classes', [])
        task_type = metadata.get('task_type', 'object_detection')
        
        # 检测模型格式
        model_format = self._detect_model_format(model_path)
        
        return ModelInfo(
            model_id=model_id,
            name=name,
            description=description,
            version=version,
            format=model_format.value,
            framework=metadata.get('framework', ''),
            classes=classes,
            num_classes=len(classes),
            confidence_threshold=metadata.get('confidence_threshold', 0.5),
            created_date=datetime.fromtimestamp(stat.st_ctime).isoformat(),
            modified_date=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            author=metadata.get('author', ''),
            tags=metadata.get('tags', []),
            metadata=metadata,
            file_path=str(model_path.absolute()),
            file_size=stat.st_size,
            is_loaded=False,
            requires_preprocessing=True,
            requires_postprocessing=True
        )
    
    def _load_model_database(self):
        """加载模型数据库"""
        try:
            if self.db_file.exists():
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for model_id, model_data in data.items():
                    self.model_infos[model_id] = ModelInfo.from_dict(model_data)
                
                self.logger.info(f"模型数据库已加载，共 {len(self.model_infos)} 个模型")
            else:
                self.logger.info("模型数据库不存在，将创建新数据库")
                
        except Exception as e:
            self.logger.error(f"加载模型数据库失败: {e}")
    
    def _save_model_database(self):
        """保存模型数据库"""
        try:
            # 转换为可序列化的字典
            data = {}
            for model_id, model_info in self.model_infos.items():
                data[model_id] = model_info.to_dict()
            
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug("模型数据库已保存")
            
        except Exception as e:
            self.logger.error(f"保存模型数据库失败: {e}")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()