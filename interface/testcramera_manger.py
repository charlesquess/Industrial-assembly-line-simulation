#!/usr/bin/env python3
"""
修复后摄像头测试
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_fixed_camera_manager():
    """测试修复后的CameraManager使用"""
    try:
        from core.camera_manager import CameraManager
        
        print("="*60)
        print("修复后CameraManager测试")
        print("="*60)
        
        # 创建管理器
        manager = CameraManager()
        
        # 获取可用摄像头
        cameras = manager.get_available_cameras()
        print(f"可用摄像头数量: {len(cameras)}")
        
        if not cameras:
            print("没有可用摄像头，测试结束")
            return
        
        # 连接第一个摄像头
        camera_dict = cameras[0]
        camera_id = camera_dict['id']
        print(f"连接摄像头: {camera_id}")
        
        success = manager.connect_camera(camera_id, "usb", device_id=camera_dict.get('device_id', 0))
        print(f"连接结果: {success}")
        
        if success:
            # 获取CameraInfo对象
            camera_info = manager.get_camera_info(camera_id)
            print(f"CameraInfo类型: {type(camera_info)}")
            
            if camera_info:
                # 正确访问CameraInfo对象的属性
                print(f"摄像头名称: {camera_info.name}")
                print(f"设备ID: {camera_info.device_id}")
                print(f"是否连接: {camera_info.is_connected}")
                
                # 访问设置
                if hasattr(camera_info, 'current_settings'):
                    settings = camera_info.current_settings
                    print(f"分辨率: {settings.resolution}")
                    print(f"帧率: {settings.fps}")
                elif hasattr(camera_info, 'capabilities'):
                    caps = camera_info.capabilities
                    print(f"分辨率: {caps.get('width', 'N/A')}x{caps.get('height', 'N/A')}")
            
            # 开始捕获
            print("开始捕获...")
            if manager.start_capture():
                # 获取几帧
                for i in range(3):
                    frame = manager.capture_frame()
                    if frame is not None:
                        print(f"  捕获第{i+1}帧: {frame.shape}")
                    else:
                        print(f"  第{i+1}帧捕获失败")
                
                # 停止捕获
                manager.stop_capture()
            
            # 断开连接
            manager.disconnect_camera(camera_id)
        
        # 清理
        manager.cleanup()
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
    
    print("="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_fixed_camera_manager()
    input("\n按Enter键退出...")