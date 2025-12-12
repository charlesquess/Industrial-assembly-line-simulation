"""
工业视觉检测系统 - 数据管理器
负责检测数据的存储、管理、分析和导出
支持数据库存储和文件系统存储
"""

import logging
import json
import csv
import sqlite3
import pickle
import hashlib
import shutil
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union, Iterator
import statistics
from collections import defaultdict, Counter
import zipfile
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QMutex, QMutexLocker, QThread

@dataclass
class InspectionData:
    """检测数据类"""
    id: str  # 唯一标识符
    timestamp: datetime  # 检测时间
    camera_id: str  # 摄像头ID
    model_id: str  # 模型ID
    
    # 检测结果
    detections: List[Dict[str, Any]] = field(default_factory=list)  # 检测对象列表
    classification_results: List[Dict[str, Any]] = field(default_factory=list)  # 分类结果
    segmentation_mask_path: Optional[str] = None  # 分割掩码路径
    
    # 统计信息
    total_objects: int = 0  # 总检测对象数
    defect_count: int = 0  # 缺陷数量
    pass_count: int = 0  # 通过数量
    confidence_score: float = 0.0  # 平均置信度
    
    # 质量评估
    inspection_result: str = "未知"  # 检测结果：通过、失败、警告
    quality_score: float = 0.0  # 质量评分 (0-100)
    defect_severity: str = "无"  # 缺陷严重程度：低、中、高
    
    # 图像信息
    original_image_path: Optional[str] = None  # 原始图像路径
    annotated_image_path: Optional[str] = None  # 标注图像路径
    thumbnail_path: Optional[str] = None  # 缩略图路径
    
    # 尺寸信息
    image_width: int = 0
    image_height: int = 0
    
    # 性能数据
    inference_time: float = 0.0  # 推理时间(ms)
    preprocessing_time: float = 0.0  # 预处理时间(ms)
    postprocessing_time: float = 0.0  # 后处理时间(ms)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    tags: List[str] = field(default_factory=list)  # 标签
    notes: str = ""  # 备注
    
    # 系统信息
    operator_id: str = ""  # 操作员ID
    station_id: str = ""  # 工站ID
    batch_id: str = ""  # 批次ID
    product_id: str = ""  # 产品ID
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InspectionData':
        """从字典创建"""
        # 处理时间戳
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        return cls(**data)


@dataclass
class StatisticalSummary:
    """统计摘要类"""
    # 时间范围
    start_time: datetime
    end_time: datetime
    total_duration_hours: float
    
    # 检测统计
    total_inspections: int = 0
    total_defects: int = 0
    total_passes: int = 0
    total_objects: int = 0
    
    # 比率指标
    defect_rate: float = 0.0  # 缺陷率
    pass_rate: float = 0.0  # 通过率
    inspection_rate: float = 0.0  # 检测速率(个/小时)
    
    # 性能指标
    avg_inference_time: float = 0.0
    avg_confidence: float = 0.0
    avg_quality_score: float = 0.0
    
    # 类别统计
    defect_distribution: Dict[str, int] = field(default_factory=dict)  # 缺陷类型分布
    severity_distribution: Dict[str, int] = field(default_factory=dict)  # 严重程度分布
    
    # 趋势数据
    hourly_trend: Dict[str, int] = field(default_factory=dict)  # 小时趋势
    daily_trend: Dict[str, int] = field(default_factory=dict)  # 天趋势
    
    # 操作员统计
    operator_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 操作员统计
    
    # 批次统计
    batch_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # 批次统计
    
    # 质量指标
    quality_metrics: Dict[str, float] = field(default_factory=dict)  # 质量指标
    process_capability: Dict[str, float] = field(default_factory=dict)  # 过程能力
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


class StorageType(Enum):
    """存储类型枚举"""
    SQLITE = "sqlite"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    PICKLE = "pickle"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


class DataExportFormat(Enum):
    """数据导出格式枚举"""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    SQL = "sql"
    HTML = "html"
    PDF = "pdf"
    IMAGE = "image"


class DataCompression(Enum):
    """数据压缩格式枚举"""
    NONE = "none"
    ZIP = "zip"
    GZIP = "gzip"
    TAR = "tar"
    TAR_GZ = "tar.gz"


class DatabaseConnection:
    """数据库连接类"""
    
    def __init__(self, db_path: str = "inspection_data.db"):
        """初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.connection = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        
        # 确保数据库目录存在
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        
    def connect(self) -> bool:
        """连接到数据库"""
        try:
            with self.lock:
                self.connection = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30.0
                )
                self.connection.row_factory = sqlite3.Row  # 返回字典格式
                self.cursor = self.connection.cursor()
                
                # 启用WAL模式提高并发性能
                self.cursor.execute("PRAGMA journal_mode=WAL")
                self.cursor.execute("PRAGMA synchronous=NORMAL")
                self.cursor.execute("PRAGMA cache_size=-2000")  # 2MB缓存
                
                self.logger.info(f"数据库连接成功: {self.db_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开数据库连接"""
        try:
            with self.lock:
                if self.cursor:
                    self.cursor.close()
                if self.connection:
                    self.connection.close()
                
                self.cursor = None
                self.connection = None
                self.logger.info("数据库连接已关闭")
                return True
                
        except Exception as e:
            self.logger.error(f"断开数据库连接失败: {e}")
            return False
    
    def create_tables(self) -> bool:
        """创建数据库表"""
        try:
            with self.lock:
                # 检测结果表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection_results (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    camera_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    total_objects INTEGER DEFAULT 0,
                    defect_count INTEGER DEFAULT 0,
                    pass_count INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 0.0,
                    inspection_result TEXT DEFAULT '未知',
                    quality_score REAL DEFAULT 0.0,
                    defect_severity TEXT DEFAULT '无',
                    original_image_path TEXT,
                    annotated_image_path TEXT,
                    thumbnail_path TEXT,
                    image_width INTEGER DEFAULT 0,
                    image_height INTEGER DEFAULT 0,
                    inference_time REAL DEFAULT 0.0,
                    preprocessing_time REAL DEFAULT 0.0,
                    postprocessing_time REAL DEFAULT 0.0,
                    detections_json TEXT,
                    classification_json TEXT,
                    metadata_json TEXT,
                    tags_json TEXT,
                    notes TEXT,
                    operator_id TEXT,
                    station_id TEXT,
                    batch_id TEXT,
                    product_id TEXT,
                    segmentation_mask_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # 创建索引
                self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON inspection_results(timestamp)')
                self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_camera_id ON inspection_results(camera_id)')
                self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_id ON inspection_results(model_id)')
                self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_result ON inspection_results(inspection_result)')
                self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_id ON inspection_results(batch_id)')
                self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_operator_id ON inspection_results(operator_id)')
                
                # 创建统计表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statistic_type TEXT NOT NULL,
                    statistic_key TEXT NOT NULL,
                    statistic_value TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(statistic_type, statistic_key, start_time, end_time)
                )
                ''')
                
                # 创建标签表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (inspection_id) REFERENCES inspection_results (id),
                    UNIQUE(inspection_id, tag)
                )
                ''')
                
                # 创建缺陷分类表
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS defect_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category_name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    severity_level TEXT DEFAULT '中等',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # 插入默认缺陷分类
                default_defects = [
                    ('划痕', '表面划痕缺陷', '中等'),
                    ('凹坑', '表面凹坑缺陷', '中等'),
                    ('裂纹', '结构裂纹缺陷', '高'),
                    ('污渍', '表面污渍缺陷', '低'),
                    ('变形', '形状变形缺陷', '高'),
                    ('缺失', '部件缺失缺陷', '高'),
                    ('尺寸偏差', '尺寸不符合标准', '中等'),
                    ('颜色异常', '颜色不符合标准', '低')
                ]
                
                for defect in default_defects:
                    self.cursor.execute('''
                    INSERT OR IGNORE INTO defect_categories (category_name, description, severity_level)
                    VALUES (?, ?, ?)
                    ''', defect)
                
                self.connection.commit()
                self.logger.info("数据库表创建成功")
                return True
                
        except Exception as e:
            self.logger.error(f"创建数据库表失败: {e}")
            return False
    
    def save_inspection(self, inspection_data: InspectionData) -> bool:
        """保存检测结果"""
        try:
            with self.lock:
                # 准备数据
                detections_json = json.dumps(inspection_data.detections, ensure_ascii=False)
                classification_json = json.dumps(inspection_data.classification_results, ensure_ascii=False)
                metadata_json = json.dumps(inspection_data.metadata, ensure_ascii=False)
                tags_json = json.dumps(inspection_data.tags, ensure_ascii=False)
                
                # 插入或更新数据
                self.cursor.execute('''
                INSERT OR REPLACE INTO inspection_results (
                    id, timestamp, camera_id, model_id, total_objects, defect_count,
                    pass_count, confidence_score, inspection_result, quality_score,
                    defect_severity, original_image_path, annotated_image_path,
                    thumbnail_path, image_width, image_height, inference_time,
                    preprocessing_time, postprocessing_time, detections_json,
                    classification_json, metadata_json, tags_json, notes,
                    operator_id, station_id, batch_id, product_id, segmentation_mask_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    inspection_data.id,
                    inspection_data.timestamp.isoformat(),
                    inspection_data.camera_id,
                    inspection_data.model_id,
                    inspection_data.total_objects,
                    inspection_data.defect_count,
                    inspection_data.pass_count,
                    inspection_data.confidence_score,
                    inspection_data.inspection_result,
                    inspection_data.quality_score,
                    inspection_data.defect_severity,
                    inspection_data.original_image_path,
                    inspection_data.annotated_image_path,
                    inspection_data.thumbnail_path,
                    inspection_data.image_width,
                    inspection_data.image_height,
                    inspection_data.inference_time,
                    inspection_data.preprocessing_time,
                    inspection_data.postprocessing_time,
                    detections_json,
                    classification_json,
                    metadata_json,
                    tags_json,
                    inspection_data.notes,
                    inspection_data.operator_id,
                    inspection_data.station_id,
                    inspection_data.batch_id,
                    inspection_data.product_id,
                    inspection_data.segmentation_mask_path
                ))
                
                # 保存标签
                if inspection_data.tags:
                    for tag in inspection_data.tags:
                        self.cursor.execute('''
                        INSERT OR IGNORE INTO inspection_tags (inspection_id, tag)
                        VALUES (?, ?)
                        ''', (inspection_data.id, tag))
                
                self.connection.commit()
                self.logger.debug(f"检测结果已保存: {inspection_data.id}")
                return True
                
        except Exception as e:
            self.logger.error(f"保存检测结果失败: {e}")
            return False
    
    def get_inspection(self, inspection_id: str) -> Optional[InspectionData]:
        """获取检测结果"""
        try:
            with self.lock:
                self.cursor.execute('''
                SELECT * FROM inspection_results WHERE id = ?
                ''', (inspection_id,))
                
                row = self.cursor.fetchone()
                if not row:
                    return None
                
                return self._row_to_inspection(row)
                
        except Exception as e:
            self.logger.error(f"获取检测结果失败: {e}")
            return None
    
    def get_inspections_by_date_range(self, start_date: datetime, end_date: datetime, 
                                     limit: int = 1000, offset: int = 0) -> List[InspectionData]:
        """根据日期范围获取检测结果"""
        try:
            with self.lock:
                self.cursor.execute('''
                SELECT * FROM inspection_results 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                ''', (
                    start_date.isoformat(),
                    end_date.isoformat(),
                    limit,
                    offset
                ))
                
                rows = self.cursor.fetchall()
                return [self._row_to_inspection(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"获取日期范围检测结果失败: {e}")
            return []
    
    def get_recent_inspections(self, count: int = 100) -> List[InspectionData]:
        """获取最近的检测结果"""
        try:
            with self.lock:
                self.cursor.execute('''
                SELECT * FROM inspection_results 
                ORDER BY timestamp DESC
                LIMIT ?
                ''', (count,))
                
                rows = self.cursor.fetchall()
                return [self._row_to_inspection(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"获取最近检测结果失败: {e}")
            return []
    
    def update_statistics(self, statistic_type: str, statistic_key: str, 
                         statistic_value: str, start_time: datetime, 
                         end_time: datetime) -> bool:
        """更新统计信息"""
        try:
            with self.lock:
                self.cursor.execute('''
                INSERT OR REPLACE INTO inspection_statistics 
                (statistic_type, statistic_key, statistic_value, start_time, end_time)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    statistic_type,
                    statistic_key,
                    statistic_value,
                    start_time.isoformat(),
                    end_time.isoformat()
                ))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"更新统计信息失败: {e}")
            return False
    
    def get_statistics(self, statistic_type: str, start_time: datetime = None, 
                      end_time: datetime = None) -> List[Dict[str, Any]]:
        """获取统计信息"""
        try:
            with self.lock:
                query = '''
                SELECT statistic_key, statistic_value, start_time, end_time 
                FROM inspection_statistics 
                WHERE statistic_type = ?
                '''
                params = [statistic_type]
                
                if start_time and end_time:
                    query += ' AND start_time >= ? AND end_time <= ?'
                    params.extend([start_time.isoformat(), end_time.isoformat()])
                
                query += ' ORDER BY start_time DESC'
                
                self.cursor.execute(query, params)
                rows = self.cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'key': row['statistic_key'],
                        'value': row['statistic_value'],
                        'start_time': datetime.fromisoformat(row['start_time']),
                        'end_time': datetime.fromisoformat(row['end_time'])
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return []
    
    def get_summary_statistics(self, start_date: datetime = None, 
                              end_date: datetime = None) -> Dict[str, Any]:
        """获取汇总统计"""
        try:
            with self.lock:
                # 构建查询条件
                conditions = []
                params = []
                
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(start_date.isoformat())
                
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(end_date.isoformat())
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # 执行查询
                query = f'''
                SELECT 
                    COUNT(*) as total_inspections,
                    SUM(total_objects) as total_objects,
                    SUM(defect_count) as total_defects,
                    SUM(pass_count) as total_passes,
                    AVG(confidence_score) as avg_confidence,
                    AVG(quality_score) as avg_quality,
                    AVG(inference_time) as avg_inference_time,
                    COUNT(CASE WHEN inspection_result = '通过' THEN 1 END) as pass_count_result,
                    COUNT(CASE WHEN inspection_result = '失败' THEN 1 END) as fail_count_result,
                    COUNT(CASE WHEN inspection_result = '警告' THEN 1 END) as warn_count_result
                FROM inspection_results
                WHERE {where_clause}
                '''
                
                self.cursor.execute(query, params)
                row = self.cursor.fetchone()
                
                if not row or row['total_inspections'] == 0:
                    return {}
                
                # 计算比率
                total_inspections = row['total_inspections']
                total_objects = row['total_objects'] or 0
                total_defects = row['total_defects'] or 0
                total_passes = row['total_passes'] or 0
                
                defect_rate = (total_defects / total_objects * 100) if total_objects > 0 else 0
                pass_rate = (total_passes / total_inspections * 100) if total_inspections > 0 else 0
                
                return {
                    'total_inspections': total_inspections,
                    'total_objects': total_objects,
                    'total_defects': total_defects,
                    'total_passes': total_passes,
                    'defect_rate': defect_rate,
                    'pass_rate': pass_rate,
                    'avg_confidence': row['avg_confidence'] or 0,
                    'avg_quality': row['avg_quality'] or 0,
                    'avg_inference_time': row['avg_inference_time'] or 0,
                    'pass_count_result': row['pass_count_result'] or 0,
                    'fail_count_result': row['fail_count_result'] or 0,
                    'warn_count_result': row['warn_count_result'] or 0
                }
                
        except Exception as e:
            self.logger.error(f"获取汇总统计失败: {e}")
            return {}
    
    def delete_inspection(self, inspection_id: str) -> bool:
        """删除检测结果"""
        try:
            with self.lock:
                # 删除相关标签
                self.cursor.execute('DELETE FROM inspection_tags WHERE inspection_id = ?', (inspection_id,))
                
                # 删除检测结果
                self.cursor.execute('DELETE FROM inspection_results WHERE id = ?', (inspection_id,))
                
                self.connection.commit()
                self.logger.info(f"检测结果已删除: {inspection_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"删除检测结果失败: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """清理旧数据
        
        Args:
            days_to_keep: 保留的天数
            
        Returns:
            删除的记录数
        """
        try:
            with self.lock:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # 获取要删除的记录ID
                self.cursor.execute('''
                SELECT id, original_image_path, annotated_image_path, 
                       thumbnail_path, segmentation_mask_path 
                FROM inspection_results 
                WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                old_records = self.cursor.fetchall()
                deleted_count = 0
                
                for record in old_records:
                    # 删除相关文件
                    file_paths = [
                        record['original_image_path'],
                        record['annotated_image_path'],
                        record['thumbnail_path'],
                        record['segmentation_mask_path']
                    ]
                    
                    for file_path in file_paths:
                        if file_path:
                            try:
                                Path(file_path).unlink(missing_ok=True)
                            except Exception as e:
                                self.logger.warning(f"删除文件失败 {file_path}: {e}")
                    
                    # 删除标签
                    self.cursor.execute('DELETE FROM inspection_tags WHERE inspection_id = ?', (record['id'],))
                    
                    # 删除检测结果
                    self.cursor.execute('DELETE FROM inspection_results WHERE id = ?', (record['id'],))
                    
                    deleted_count += 1
                
                self.connection.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"清理了 {deleted_count} 条旧记录")
                
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"清理旧数据失败: {e}")
            return 0
    
    def _row_to_inspection(self, row) -> InspectionData:
        """将数据库行转换为InspectionData对象"""
        try:
            # 解析JSON字段
            detections = json.loads(row['detections_json']) if row['detections_json'] else []
            classification_results = json.loads(row['classification_json']) if row['classification_json'] else []
            metadata = json.loads(row['metadata_json']) if row['metadata_json'] else {}
            tags = json.loads(row['tags_json']) if row['tags_json'] else []
            
            # 解析时间戳
            timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            
            return InspectionData(
                id=row['id'],
                timestamp=timestamp,
                camera_id=row['camera_id'],
                model_id=row['model_id'],
                detections=detections,
                classification_results=classification_results,
                segmentation_mask_path=row['segmentation_mask_path'],
                total_objects=row['total_objects'],
                defect_count=row['defect_count'],
                pass_count=row['pass_count'],
                confidence_score=row['confidence_score'],
                inspection_result=row['inspection_result'],
                quality_score=row['quality_score'],
                defect_severity=row['defect_severity'],
                original_image_path=row['original_image_path'],
                annotated_image_path=row['annotated_image_path'],
                thumbnail_path=row['thumbnail_path'],
                image_width=row['image_width'],
                image_height=row['image_height'],
                inference_time=row['inference_time'],
                preprocessing_time=row['preprocessing_time'],
                postprocessing_time=row['postprocessing_time'],
                metadata=metadata,
                tags=tags,
                notes=row['notes'],
                operator_id=row['operator_id'],
                station_id=row['station_id'],
                batch_id=row['batch_id'],
                product_id=row['product_id']
            )
            
        except Exception as e:
            self.logger.error(f"转换行数据失败: {e}")
            raise
    
    def backup_database(self, backup_path: str) -> bool:
        """备份数据库"""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(exist_ok=True, parents=True)
            
            # 创建备份
            shutil.copy2(self.db_path, backup_path)
            
            self.logger.info(f"数据库已备份到: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"备份数据库失败: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """恢复数据库"""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                self.logger.error(f"备份文件不存在: {backup_path}")
                return False
            
            # 关闭当前连接
            self.disconnect()
            
            # 恢复备份
            shutil.copy2(backup_path, self.db_path)
            
            # 重新连接
            self.connect()
            
            self.logger.info(f"数据库已从备份恢复: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复数据库失败: {e}")
            return False


class ImageManager:
    """图像管理器"""
    
    def __init__(self, images_dir: str = "images"):
        """初始化图像管理器
        
        Args:
            images_dir: 图像存储目录
        """
        self.images_dir = Path(images_dir)
        self.logger = logging.getLogger(__name__)
        
        # 创建目录结构
        self.original_dir = self.images_dir / "original"
        self.annotated_dir = self.images_dir / "annotated"
        self.thumbnail_dir = self.images_dir / "thumbnails"
        self.masks_dir = self.images_dir / "masks"
        
        for directory in [self.original_dir, self.annotated_dir, 
                         self.thumbnail_dir, self.masks_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def save_original_image(self, image: np.ndarray, inspection_id: str) -> Optional[str]:
        """保存原始图像"""
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"original_{inspection_id}_{timestamp}.jpg"
            filepath = self.original_dir / filename
            
            # 保存图像
            success = cv2.imwrite(str(filepath), image)
            
            if success:
                self.logger.debug(f"原始图像已保存: {filepath}")
                return str(filepath)
            else:
                self.logger.error(f"保存原始图像失败: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"保存原始图像异常: {e}")
            return None
    
    def save_annotated_image(self, image: np.ndarray, inspection_id: str, 
                            detections: List[Dict[str, Any]]) -> Optional[str]:
        """保存标注图像"""
        try:
            # 创建标注图像副本
            annotated_image = image.copy()
            
            # 绘制检测结果
            for detection in detections:
                bbox = detection.get('bbox', [])
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    confidence = detection.get('confidence', 0)
                    class_name = detection.get('class_name', '未知')
                    
                    # 绘制边界框
                    color = (0, 255, 0) if detection.get('status') == '通过' else (0, 0, 255)
                    thickness = 2
                    
                    cv2.rectangle(annotated_image, (int(x), int(y)), 
                                (int(x + w), int(y + h)), color, thickness)
                    
                    # 绘制标签
                    label = f"{class_name}: {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, font_thickness
                    )
                    
                    # 标签背景
                    cv2.rectangle(annotated_image, 
                                (int(x), int(y) - text_height - 5),
                                (int(x) + text_width, int(y)),
                                color, -1)
                    
                    # 标签文本
                    cv2.putText(annotated_image, label,
                              (int(x), int(y) - 5),
                              font, font_scale, (255, 255, 255), font_thickness)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"annotated_{inspection_id}_{timestamp}.jpg"
            filepath = self.annotated_dir / filename
            
            # 保存图像
            success = cv2.imwrite(str(filepath), annotated_image)
            
            if success:
                self.logger.debug(f"标注图像已保存: {filepath}")
                return str(filepath)
            else:
                self.logger.error(f"保存标注图像失败: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"保存标注图像异常: {e}")
            return None
    
    def create_thumbnail(self, image: np.ndarray, inspection_id: str, 
                        max_size: Tuple[int, int] = (200, 200)) -> Optional[str]:
        """创建缩略图"""
        try:
            # 调整大小
            height, width = image.shape[:2]
            target_width, target_height = max_size
            
            # 计算缩放比例
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 调整图像大小
            thumbnail = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"thumbnail_{inspection_id}_{timestamp}.jpg"
            filepath = self.thumbnail_dir / filename
            
            # 保存缩略图
            success = cv2.imwrite(str(filepath), thumbnail)
            
            if success:
                self.logger.debug(f"缩略图已保存: {filepath}")
                return str(filepath)
            else:
                self.logger.error(f"保存缩略图失败: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"创建缩略图异常: {e}")
            return None
    
    def save_segmentation_mask(self, mask: np.ndarray, inspection_id: str) -> Optional[str]:
        """保存分割掩码"""
        try:
            # 确保掩码是uint8类型
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mask_{inspection_id}_{timestamp}.png"
            filepath = self.masks_dir / filename
            
            # 保存掩码
            success = cv2.imwrite(str(filepath), mask)
            
            if success:
                self.logger.debug(f"分割掩码已保存: {filepath}")
                return str(filepath)
            else:
                self.logger.error(f"保存分割掩码失败: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"保存分割掩码异常: {e}")
            return None
    
    def get_image(self, image_path: str) -> Optional[np.ndarray]:
        """获取图像"""
        try:
            if not Path(image_path).exists():
                self.logger.warning(f"图像文件不存在: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"无法读取图像: {image_path}")
                return None
            
            return image
            
        except Exception as e:
            self.logger.error(f"获取图像异常: {e}")
            return None
    
    def delete_image_files(self, inspection_data: InspectionData) -> bool:
        """删除图像文件"""
        try:
            files_to_delete = [
                inspection_data.original_image_path,
                inspection_data.annotated_image_path,
                inspection_data.thumbnail_path,
                inspection_data.segmentation_mask_path
            ]
            
            success = True
            for file_path in files_to_delete:
                if file_path and Path(file_path).exists():
                    try:
                        Path(file_path).unlink()
                        self.logger.debug(f"图像文件已删除: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"删除图像文件失败 {file_path}: {e}")
                        success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"删除图像文件异常: {e}")
            return False
    
    def cleanup_old_images(self, days_to_keep: int = 30) -> int:
        """清理旧图像"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for directory in [self.original_dir, self.annotated_dir, 
                            self.thumbnail_dir, self.masks_dir]:
                for file_path in directory.glob("*"):
                    if file_path.is_file():
                        # 检查文件修改时间
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff_time:
                            try:
                                file_path.unlink()
                                deleted_count += 1
                            except Exception as e:
                                self.logger.warning(f"删除旧图像失败 {file_path}: {e}")
            
            if deleted_count > 0:
                self.logger.info(f"清理了 {deleted_count} 个旧图像文件")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"清理旧图像异常: {e}")
            return 0


class DataExporter:
    """数据导出器"""
    
    def __init__(self):
        """初始化数据导出器"""
        self.logger = logging.getLogger(__name__)
    
    def export_to_csv(self, data: List[InspectionData], filepath: str) -> bool:
        """导出为CSV格式"""
        try:
            filepath = Path(filepath)
            
            # 准备数据
            export_data = []
            for inspection in data:
                row = {
                    'ID': inspection.id,
                    '时间': inspection.timestamp.isoformat(),
                    '摄像头': inspection.camera_id,
                    '模型': inspection.model_id,
                    '总对象数': inspection.total_objects,
                    '缺陷数': inspection.defect_count,
                    '通过数': inspection.pass_count,
                    '置信度': inspection.confidence_score,
                    '检测结果': inspection.inspection_result,
                    '质量评分': inspection.quality_score,
                    '缺陷严重度': inspection.defect_severity,
                    '推理时间(ms)': inspection.inference_time,
                    '预处理时间(ms)': inspection.preprocessing_time,
                    '后处理时间(ms)': inspection.postprocessing_time,
                    '操作员': inspection.operator_id,
                    '工站': inspection.station_id,
                    '批次': inspection.batch_id,
                    '产品': inspection.product_id,
                    '备注': inspection.notes
                }
                export_data.append(row)
            
            # 写入CSV文件
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                if export_data:
                    writer = csv.DictWriter(f, fieldnames=export_data[0].keys())
                    writer.writeheader()
                    writer.writerows(export_data)
            
            self.logger.info(f"数据已导出为CSV: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出CSV失败: {e}")
            return False
    
    def export_to_excel(self, data: List[InspectionData], filepath: str) -> bool:
        """导出为Excel格式"""
        try:
            # 检查是否安装了pandas和openpyxl
            try:
                import pandas as pd
            except ImportError:
                self.logger.error("导出Excel需要安装pandas和openpyxl")
                return False
            
            # 准备数据
            export_data = []
            for inspection in data:
                row = {
                    'ID': inspection.id,
                    '时间': inspection.timestamp,
                    '摄像头': inspection.camera_id,
                    '模型': inspection.model_id,
                    '总对象数': inspection.total_objects,
                    '缺陷数': inspection.defect_count,
                    '通过数': inspection.pass_count,
                    '置信度': inspection.confidence_score,
                    '检测结果': inspection.inspection_result,
                    '质量评分': inspection.quality_score,
                    '缺陷严重度': inspection.defect_severity,
                    '推理时间(ms)': inspection.inference_time,
                    '预处理时间(ms)': inspection.preprocessing_time,
                    '后处理时间(ms)': inspection.postprocessing_time,
                    '操作员': inspection.operator_id,
                    '工站': inspection.station_id,
                    '批次': inspection.batch_id,
                    '产品': inspection.product_id,
                    '备注': inspection.notes,
                    '标签': ','.join(inspection.tags)
                }
                export_data.append(row)
            
            # 创建DataFrame并导出
            df = pd.DataFrame(export_data)
            
            # 创建Excel写入器
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 写入主数据
                df.to_excel(writer, sheet_name='检测数据', index=False)
                
                # 如果有数据，创建统计表
                if not df.empty:
                    # 计算统计信息
                    summary_data = {
                        '总检测数': [len(data)],
                        '总对象数': [df['总对象数'].sum()],
                        '总缺陷数': [df['缺陷数'].sum()],
                        '总通过数': [df['通过数'].sum()],
                        '平均置信度': [df['置信度'].mean()],
                        '平均质量评分': [df['质量评分'].mean()],
                        '平均推理时间': [df['推理时间(ms)'].mean()],
                        '通过率': [(df['检测结果'] == '通过').sum() / len(data) * 100],
                        '缺陷率': [df['缺陷数'].sum() / df['总对象数'].sum() * 100 if df['总对象数'].sum() > 0 else 0]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
                    
                    # 创建分类统计
                    if '检测结果' in df.columns:
                        result_counts = df['检测结果'].value_counts()
                        result_df = pd.DataFrame({
                            '检测结果': result_counts.index,
                            '数量': result_counts.values,
                            '比例': result_counts.values / len(data) * 100
                        })
                        result_df.to_excel(writer, sheet_name='结果分布', index=False)
            
            self.logger.info(f"数据已导出为Excel: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出Excel失败: {e}")
            return False
    
    def export_to_json(self, data: List[InspectionData], filepath: str) -> bool:
        """导出为JSON格式"""
        try:
            filepath = Path(filepath)
            
            # 转换为字典列表
            export_data = [inspection.to_dict() for inspection in data]
            
            # 写入JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据已导出为JSON: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出JSON失败: {e}")
            return False
    
    def export_report(self, data: List[InspectionData], summary: StatisticalSummary, 
                     filepath: str, format: DataExportFormat = DataExportFormat.HTML) -> bool:
        """导出完整报告"""
        try:
            if format == DataExportFormat.HTML:
                return self._export_html_report(data, summary, filepath)
            elif format == DataExportFormat.PDF:
                return self._export_pdf_report(data, summary, filepath)
            else:
                self.logger.error(f"不支持的报告格式: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"导出报告失败: {e}")
            return False
    
    def _export_html_report(self, data: List[InspectionData], 
                           summary: StatisticalSummary, filepath: str) -> bool:
        """导出HTML报告"""
        try:
            filepath = Path(filepath)
            
            # 生成HTML报告
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>工业视觉检测报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .title {{ font-size: 24px; font-weight: bold; color: #333; }}
        .subtitle {{ font-size: 16px; color: #666; margin-top: 10px; }}
        .section {{ margin-bottom: 30px; }}
        .section-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; 
                         border-bottom: 2px solid #3498db; padding-bottom: 5px; 
                         margin-bottom: 15px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); 
                       gap: 15px; margin-bottom: 20px; }}
        .summary-item {{ background-color: #f8f9fa; border: 1px solid #dee2e6; 
                       border-radius: 5px; padding: 15px; text-align: center; }}
        .summary-value {{ font-size: 24px; font-weight: bold; color: #007bff; 
                         margin: 10px 0; }}
        .summary-label {{ font-size: 14px; color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .timestamp {{ text-align: right; color: #6c757d; font-size: 12px; 
                     margin-top: 30px; }}
        .chart-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">工业视觉检测报告</div>
        <div class="subtitle">生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</div>
        <div class="subtitle">报告期间: {summary.start_time.strftime('%Y-%m-%d %H:%M')} 至 {summary.end_time.strftime('%Y-%m-%d %H:%M')}</div>
    </div>
    
    <div class="section">
        <div class="section-title">检测摘要</div>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-label">总检测数</div>
                <div class="summary-value">{summary.total_inspections}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">总对象数</div>
                <div class="summary-value">{summary.total_objects}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">总缺陷数</div>
                <div class="summary-value">{summary.total_defects}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">通过率</div>
                <div class="summary-value">{summary.pass_rate:.1f}%</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">缺陷率</div>
                <div class="summary-value">{summary.defect_rate:.1f}%</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">平均推理时间</div>
                <div class="summary-value">{summary.avg_inference_time:.1f}ms</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <div class="section-title">缺陷分布</div>
        <table>
            <tr>
                <th>缺陷类型</th>
                <th>数量</th>
                <th>比例</th>
                <th>严重程度</th>
            </tr>
"""
            
            # 添加缺陷分布
            for defect_type, count in summary.defect_distribution.items():
                percentage = (count / summary.total_defects * 100) if summary.total_defects > 0 else 0
                html_content += f"""
            <tr>
                <td>{defect_type}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
                <td>{summary.severity_distribution.get(defect_type, '未知')}</td>
            </tr>
"""
            
            html_content += """
        </table>
    </div>
    
    <div class="section">
        <div class="section-title">最近检测记录</div>
        <table>
            <tr>
                <th>时间</th>
                <th>摄像头</th>
                <th>检测结果</th>
                <th>缺陷数</th>
                <th>质量评分</th>
                <th>操作员</th>
            </tr>
"""
            
            # 添加最近检测记录（最多10条）
            recent_data = data[:10] if len(data) > 10 else data
            for inspection in recent_data:
                result_class = ''
                if inspection.inspection_result == '通过':
                    result_class = 'pass'
                elif inspection.inspection_result == '失败':
                    result_class = 'fail'
                elif inspection.inspection_result == '警告':
                    result_class = 'warning'
                
                html_content += f"""
            <tr>
                <td>{inspection.timestamp.strftime('%H:%M:%S')}</td>
                <td>{inspection.camera_id}</td>
                <td class="{result_class}">{inspection.inspection_result}</td>
                <td>{inspection.defect_count}</td>
                <td>{inspection.quality_score:.1f}</td>
                <td>{inspection.operator_id or 'N/A'}</td>
            </tr>
"""
            
            html_content += """
        </table>
    </div>
    
    <div class="section">
        <div class="section-title">质量指标</div>
        <table>
            <tr>
                <th>指标</th>
                <th>数值</th>
                <th>说明</th>
            </tr>
"""
            
            # 添加质量指标
            for metric, value in summary.quality_metrics.items():
                html_content += f"""
            <tr>
                <td>{metric}</td>
                <td>{value:.3f}</td>
                <td>质量评估指标</td>
            </tr>
"""
            
            html_content += """
        </table>
    </div>
    
    <div class="timestamp">
        报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    
</body>
</html>
"""
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML报告已导出: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出HTML报告失败: {e}")
            return False
    
    def _export_pdf_report(self, data: List[InspectionData], 
                          summary: StatisticalSummary, filepath: str) -> bool:
        """导出PDF报告"""
        try:
            # 需要安装reportlab
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
            except ImportError:
                self.logger.error("导出PDF需要安装reportlab")
                return False
            
            # 创建PDF文档
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            elements = []
            
            # 样式
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=20,
                alignment=1  # 居中
            )
            
            # 标题
            elements.append(Paragraph("工业视觉检测报告", title_style))
            elements.append(Paragraph(f"报告期间: {summary.start_time.strftime('%Y-%m-%d %H:%M')} 至 {summary.end_time.strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
            elements.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # 摘要表格
            summary_data = [
                ['指标', '数值', '说明'],
                ['总检测数', str(summary.total_inspections), '检测总数'],
                ['总对象数', str(summary.total_objects), '检测对象总数'],
                ['总缺陷数', str(summary.total_defects), '缺陷总数'],
                ['通过率', f"{summary.pass_rate:.1f}%", '通过检测的比例'],
                ['缺陷率', f"{summary.defect_rate:.1f}%", '缺陷比例'],
                ['平均推理时间', f"{summary.avg_inference_time:.1f}ms", '平均推理时间']
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # 缺陷分布
            elements.append(Paragraph("缺陷分布", styles['Heading2']))
            
            defect_data = [['缺陷类型', '数量', '比例', '严重程度']]
            for defect_type, count in summary.defect_distribution.items():
                percentage = (count / summary.total_defects * 100) if summary.total_defects > 0 else 0
                severity = summary.severity_distribution.get(defect_type, '未知')
                defect_data.append([defect_type, str(count), f"{percentage:.1f}%", severity])
            
            defect_table = Table(defect_data)
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(defect_table)
            elements.append(Spacer(1, 20))
            
            # 生成PDF
            doc.build(elements)
            
            self.logger.info(f"PDF报告已导出: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出PDF报告失败: {e}")
            return False
    
    def create_data_package(self, data: List[InspectionData], filepath: str, 
                           compression: DataCompression = DataCompression.ZIP) -> bool:
        """创建数据包"""
        try:
            filepath = Path(filepath)
            
            if compression == DataCompression.ZIP:
                # 创建ZIP压缩包
                with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # 创建临时目录
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        
                        # 导出数据文件
                        data_file = temp_path / "data.json"
                        self.export_to_json(data, str(data_file))
                        
                        # 导出统计信息
                        summary_file = temp_path / "summary.txt"
                        with open(summary_file, 'w', encoding='utf-8') as f:
                            f.write(f"数据包生成时间: {datetime.now()}\n")
                            f.write(f"数据记录数: {len(data)}\n")
                        
                        # 添加到压缩包
                        zipf.write(data_file, "data.json")
                        zipf.write(summary_file, "summary.txt")
                
                self.logger.info(f"数据包已创建: {filepath}")
                return True
            else:
                self.logger.error(f"不支持的压缩格式: {compression}")
                return False
                
        except Exception as e:
            self.logger.error(f"创建数据包失败: {e}")
            return False


class DataManager(QObject):
    """数据管理器"""
    
    # 信号定义
    data_updated = pyqtSignal(dict)  # 数据更新通知
    data_saved = pyqtSignal(str)  # 数据保存通知
    data_exported = pyqtSignal(str, str)  # 数据导出通知 (文件路径, 格式)
    data_cleaned = pyqtSignal(int)  # 数据清理通知 (清理数量)
    error_occurred = pyqtSignal(str)  # 错误通知
    
    def __init__(self, data_dir: str = "data", 
                 database_path: str = "data/inspection_data.db"):
        """初始化数据管理器
        
        Args:
            data_dir: 数据存储目录
            database_path: 数据库文件路径
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.database_path = database_path
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.db = DatabaseConnection(database_path)
        self.image_manager = ImageManager(str(self.data_dir / "images"))
        self.exporter = DataExporter()
        
        # 数据缓存
        self.recent_inspections: List[InspectionData] = []
        self.cache_max_size = 1000
        
        # 统计缓存
        self.statistics_cache: Dict[str, Any] = {}
        self.cache_expiry_time = 300  # 5分钟
        
        # 互斥锁
        self.data_lock = QMutex()
        
        # 清理定时器
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._periodic_cleanup)
        self.cleanup_timer.start(3600000)  # 每小时清理一次 (1小时)
        
        # 备份定时器
        self.backup_timer = QTimer()
        self.backup_timer.timeout.connect(self._periodic_backup)
        self.backup_timer.start(86400000)  # 每天备份一次 (24小时)
        
        # 初始化数据库
        self._initialize_database()
        
        self.logger.info(f"数据管理器初始化完成，数据目录: {self.data_dir}")
    
    def _initialize_database(self) -> bool:
        """初始化数据库"""
        try:
            # 确保数据目录存在
            self.data_dir.mkdir(exist_ok=True, parents=True)
            
            # 连接数据库
            if not self.db.connect():
                self.logger.error("数据库连接失败")
                return False
            
            # 创建表
            if not self.db.create_tables():
                self.logger.error("数据库表创建失败")
                return False
            
            # 加载最近数据
            self.recent_inspections = self.db.get_recent_inspections(100)
            
            self.logger.info("数据库初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            return False
    
    def save_inspection_result(self, inspection_data: InspectionData, 
                              save_images: bool = True) -> bool:
        """保存检测结果"""
        try:
            with QMutexLocker(self.data_lock):
                # 生成唯一ID（如果尚未生成）
                if not inspection_data.id:
                    inspection_data.id = self._generate_inspection_id(inspection_data)
                
                # 保存图像（如果需要）
                if save_images and inspection_data.original_image_path:
                    # 如果提供了图像数据，保存图像
                    image = self.image_manager.get_image(inspection_data.original_image_path)
                    if image is None:
                        self.logger.warning("无法读取原始图像，跳过图像保存")
                    else:
                        # 保存缩略图
                        thumbnail_path = self.image_manager.create_thumbnail(
                            image, inspection_data.id
                        )
                        if thumbnail_path:
                            inspection_data.thumbnail_path = thumbnail_path
                        
                        # 保存标注图像（如果有检测结果）
                        if inspection_data.detections:
                            annotated_path = self.image_manager.save_annotated_image(
                                image, inspection_data.id, inspection_data.detections
                            )
                            if annotated_path:
                                inspection_data.annotated_image_path = annotated_path
                
                # 保存到数据库
                success = self.db.save_inspection(inspection_data)
                
                if success:
                    # 更新缓存
                    self.recent_inspections.insert(0, inspection_data)
                    if len(self.recent_inspections) > self.cache_max_size:
                        self.recent_inspections = self.recent_inspections[:self.cache_max_size]
                    
                    # 清空统计缓存
                    self.statistics_cache.clear()
                    
                    # 发出信号
                    self.data_saved.emit(inspection_data.id)
                    self.data_updated.emit({
                        'type': 'inspection_saved',
                        'inspection_id': inspection_data.id,
                        'timestamp': inspection_data.timestamp
                    })
                    
                    self.logger.info(f"检测结果已保存: {inspection_data.id}")
                else:
                    self.logger.error(f"保存检测结果失败: {inspection_data.id}")
                
                return success
                
        except Exception as e:
            error_msg = f"保存检测结果异常: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def get_inspection(self, inspection_id: str) -> Optional[InspectionData]:
        """获取检测结果"""
        try:
            with QMutexLocker(self.data_lock):
                # 首先检查缓存
                for inspection in self.recent_inspections:
                    if inspection.id == inspection_id:
                        return inspection
                
                # 从数据库获取
                return self.db.get_inspection(inspection_id)
                
        except Exception as e:
            self.logger.error(f"获取检测结果失败: {e}")
            return None
    
    def get_inspections_by_date_range(self, start_date: datetime, 
                                     end_date: datetime, 
                                     limit: int = 1000) -> List[InspectionData]:
        """根据日期范围获取检测结果"""
        try:
            with QMutexLocker(self.data_lock):
                return self.db.get_inspections_by_date_range(
                    start_date, end_date, limit
                )
                
        except Exception as e:
            self.logger.error(f"获取日期范围检测结果失败: {e}")
            return []
    
    def get_recent_inspections(self, count: int = 100) -> List[InspectionData]:
        """获取最近的检测结果"""
        try:
            with QMutexLocker(self.data_lock):
                if len(self.recent_inspections) >= count:
                    return self.recent_inspections[:count]
                
                # 从数据库获取更多数据
                db_inspections = self.db.get_recent_inspections(count)
                return db_inspections
                
        except Exception as e:
            self.logger.error(f"获取最近检测结果失败: {e}")
            return []
    
    def get_statistical_summary(self, start_date: datetime = None, 
                               end_date: datetime = None) -> StatisticalSummary:
        """获取统计摘要"""
        try:
            # 生成缓存键
            cache_key = f"summary_{start_date}_{end_date}"
            
            # 检查缓存
            current_time = datetime.now()
            if (cache_key in self.statistics_cache and 
                (current_time - self.statistics_cache[cache_key]['timestamp']).seconds < self.cache_expiry_time):
                return self.statistics_cache[cache_key]['data']
            
            with QMutexLocker(self.data_lock):
                # 设置默认时间范围
                if end_date is None:
                    end_date = datetime.now()
                if start_date is None:
                    start_date = end_date - timedelta(days=1)  # 默认最近一天
                
                # 获取数据
                inspections = self.db.get_inspections_by_date_range(start_date, end_date)
                
                if not inspections:
                    summary = StatisticalSummary(
                        start_time=start_date,
                        end_time=end_date,
                        total_duration_hours=(end_date - start_date).total_seconds() / 3600
                    )
                else:
                    # 计算统计信息
                    total_inspections = len(inspections)
                    total_objects = sum(i.total_objects for i in inspections)
                    total_defects = sum(i.defect_count for i in inspections)
                    total_passes = sum(i.pass_count for i in inspections)
                    
                    # 计算比率
                    defect_rate = (total_defects / total_objects * 100) if total_objects > 0 else 0
                    pass_rate = (total_passes / total_inspections * 100) if total_inspections > 0 else 0
                    
                    # 计算平均时间
                    avg_inference_time = statistics.mean([i.inference_time for i in inspections if i.inference_time > 0]) if inspections else 0
                    avg_confidence = statistics.mean([i.confidence_score for i in inspections if i.confidence_score > 0]) if inspections else 0
                    avg_quality_score = statistics.mean([i.quality_score for i in inspections if i.quality_score > 0]) if inspections else 0
                    
                    # 计算缺陷分布
                    defect_distribution = {}
                    severity_distribution = {}
                    
                    for inspection in inspections:
                        for detection in inspection.detections:
                            if detection.get('class_name'):
                                defect_type = detection['class_name']
                                defect_distribution[defect_type] = defect_distribution.get(defect_type, 0) + 1
                                severity_distribution[defect_type] = detection.get('severity', '未知')
                    
                    # 计算趋势
                    hourly_trend = defaultdict(int)
                    daily_trend = defaultdict(int)
                    
                    for inspection in inspections:
                        hour_key = inspection.timestamp.strftime("%Y-%m-%d %H:00")
                        day_key = inspection.timestamp.strftime("%Y-%m-%d")
                        hourly_trend[hour_key] += 1
                        daily_trend[day_key] += 1
                    
                    # 计算过程能力指标
                    quality_scores = [i.quality_score for i in inspections if i.quality_score > 0]
                    
                    if quality_scores:
                        mean_quality = statistics.mean(quality_scores)
                        std_quality = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
                        
                        process_capability = {
                            'cp': (100 - 0) / (6 * std_quality) if std_quality > 0 else 0,
                            'cpk': min((100 - mean_quality) / (3 * std_quality), 
                                      (mean_quality - 0) / (3 * std_quality)) if std_quality > 0 else 0
                        }
                    else:
                        process_capability = {'cp': 0, 'cpk': 0}
                    
                    summary = StatisticalSummary(
                        start_time=start_date,
                        end_time=end_date,
                        total_duration_hours=(end_date - start_date).total_seconds() / 3600,
                        total_inspections=total_inspections,
                        total_objects=total_objects,
                        total_defects=total_defects,
                        total_passes=total_passes,
                        defect_rate=defect_rate,
                        pass_rate=pass_rate,
                        inspection_rate=total_inspections / max(1, (end_date - start_date).total_seconds() / 3600),
                        avg_inference_time=avg_inference_time,
                        avg_confidence=avg_confidence,
                        avg_quality_score=avg_quality_score,
                        defect_distribution=dict(defect_distribution),
                        severity_distribution=dict(severity_distribution),
                        hourly_trend=dict(hourly_trend),
                        daily_trend=dict(daily_trend),
                        quality_metrics={
                            '平均值': avg_quality_score,
                            '标准差': std_quality if 'std_quality' in locals() else 0,
                            '最小值': min(quality_scores) if quality_scores else 0,
                            '最大值': max(quality_scores) if quality_scores else 0
                        },
                        process_capability=process_capability
                    )
                
                # 更新缓存
                self.statistics_cache[cache_key] = {
                    'data': summary,
                    'timestamp': current_time
                }
                
                return summary
                
        except Exception as e:
            self.logger.error(f"获取统计摘要失败: {e}")
            return StatisticalSummary(
                start_time=start_date or datetime.now(),
                end_time=end_date or datetime.now(),
                total_duration_hours=0
            )
    
    def export_data(self, filepath: str, format: DataExportFormat = DataExportFormat.CSV,
                   start_date: datetime = None, end_date: datetime = None) -> bool:
        """导出数据"""
        try:
            with QMutexLocker(self.data_lock):
                # 获取数据
                if start_date and end_date:
                    data = self.db.get_inspections_by_date_range(start_date, end_date)
                else:
                    data = self.db.get_recent_inspections(1000)  # 限制导出数量
                
                if not data:
                    self.logger.warning("没有可导出的数据")
                    return False
                
                # 根据格式导出
                success = False
                
                if format == DataExportFormat.CSV:
                    success = self.exporter.export_to_csv(data, filepath)
                elif format == DataExportFormat.EXCEL:
                    success = self.exporter.export_to_excel(data, filepath)
                elif format == DataExportFormat.JSON:
                    success = self.exporter.export_to_json(data, filepath)
                elif format == DataExportFormat.HTML:
                    summary = self.get_statistical_summary(start_date, end_date)
                    success = self.exporter.export_report(data, summary, filepath, format)
                elif format == DataExportFormat.PDF:
                    summary = self.get_statistical_summary(start_date, end_date)
                    success = self.exporter.export_report(data, summary, filepath, format)
                else:
                    self.logger.error(f"不支持的导出格式: {format}")
                    return False
                
                if success:
                    self.data_exported.emit(filepath, format.value)
                    self.logger.info(f"数据已导出: {filepath} ({format.value})")
                
                return success
                
        except Exception as e:
            error_msg = f"导出数据失败: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def delete_inspection(self, inspection_id: str) -> bool:
        """删除检测结果"""
        try:
            with QMutexLocker(self.data_lock):
                # 获取检测结果
                inspection = self.db.get_inspection(inspection_id)
                if not inspection:
                    self.logger.warning(f"检测结果不存在: {inspection_id}")
                    return False
                
                # 删除图像文件
                self.image_manager.delete_image_files(inspection)
                
                # 从数据库删除
                success = self.db.delete_inspection(inspection_id)
                
                if success:
                    # 从缓存中删除
                    self.recent_inspections = [
                        i for i in self.recent_inspections if i.id != inspection_id
                    ]
                    
                    # 清空统计缓存
                    self.statistics_cache.clear()
                    
                    self.logger.info(f"检测结果已删除: {inspection_id}")
                else:
                    self.logger.error(f"删除检测结果失败: {inspection_id}")
                
                return success
                
        except Exception as e:
            error_msg = f"删除检测结果异常: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """清理旧数据"""
        try:
            with QMutexLocker(self.data_lock):
                # 清理数据库
                db_deleted = self.db.cleanup_old_data(days_to_keep)
                
                # 清理图像文件
                image_deleted = self.image_manager.cleanup_old_images(days_to_keep)
                
                total_deleted = db_deleted + image_deleted
                
                if total_deleted > 0:
                    # 清空缓存
                    self.recent_inspections = self.db.get_recent_inspections(100)
                    self.statistics_cache.clear()
                    
                    # 发出信号
                    self.data_cleaned.emit(total_deleted)
                    
                    self.logger.info(f"清理了 {total_deleted} 条旧记录")
                
                return total_deleted
                
        except Exception as e:
            error_msg = f"清理旧数据异常: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return 0
    
    def backup_database(self, backup_path: str = None) -> bool:
        """备份数据库"""
        try:
            if backup_path is None:
                backup_dir = self.data_dir / "backups"
                backup_dir.mkdir(exist_ok=True)
                backup_path = backup_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            success = self.db.backup_database(backup_path)
            
            if success:
                self.logger.info(f"数据库已备份到: {backup_path}")
            
            return success
            
        except Exception as e:
            error_msg = f"备份数据库异常: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """恢复数据库"""
        try:
            success = self.db.restore_database(backup_path)
            
            if success:
                # 重新加载数据
                self.recent_inspections = self.db.get_recent_inspections(100)
                self.statistics_cache.clear()
                
                self.logger.info(f"数据库已从备份恢复: {backup_path}")
            
            return success
            
        except Exception as e:
            error_msg = f"恢复数据库异常: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        try:
            with QMutexLocker(self.data_lock):
                # 获取数据库大小
                db_size = 0
                db_file = Path(self.database_path)
                if db_file.exists():
                    db_size = db_file.stat().st_size
                
                # 获取记录数
                self.db.cursor.execute('SELECT COUNT(*) as count FROM inspection_results')
                row = self.db.cursor.fetchone()
                record_count = row['count'] if row else 0
                
                # 获取表信息
                self.db.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row['name'] for row in self.db.cursor.fetchall()]
                
                return {
                    'path': str(db_file),
                    'size_bytes': db_size,
                    'size_mb': db_size / (1024 * 1024),
                    'record_count': record_count,
                    'tables': tables,
                    'cache_size': len(self.recent_inspections)
                }
                
        except Exception as e:
            self.logger.error(f"获取数据库信息失败: {e}")
            return {}
    
    def _generate_inspection_id(self, inspection_data: InspectionData) -> str:
        """生成检测ID"""
        timestamp_str = inspection_data.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = hashlib.md5(str(inspection_data.timestamp).encode()).hexdigest()[:6]
        
        return f"INS_{timestamp_str}_{random_suffix}"
    
    def _periodic_cleanup(self):
        """定期清理"""
        try:
            self.cleanup_old_data(30)  # 保留30天数据
        except Exception as e:
            self.logger.error(f"定期清理失败: {e}")
    
    def _periodic_backup(self):
        """定期备份"""
        try:
            self.backup_database()
        except Exception as e:
            self.logger.error(f"定期备份失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("正在清理数据管理器...")
        
        # 停止定时器
        if self.cleanup_timer.isActive():
            self.cleanup_timer.stop()
        
        if self.backup_timer.isActive():
            self.backup_timer.stop()
        
        # 断开数据库连接
        self.db.disconnect()
        
        self.logger.info("数据管理器清理完成")
    
    def __del__(self):
        """析构函数"""
        self.cleanup()