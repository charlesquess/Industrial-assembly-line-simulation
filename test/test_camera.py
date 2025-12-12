#!/usr/bin/env python3
"""
摄像头独立测试程序
用于测试摄像头基本功能，不依赖主界面
"""

import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_camera_direct():
    """直接使用OpenCV测试摄像头"""
    logger.info("=== 开始直接摄像头测试 ===")
    
    # 测试前几个摄像头设备
    for device_id in range(3):
        logger.info(f"测试摄像头设备 {device_id}...")
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            logger.warning(f"摄像头 {device_id} 无法打开")
            cap.release()
            continue
        
        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"摄像头 {device_id} 信息:")
        logger.info(f"  分辨率: {width}x{height}")
        logger.info(f"  帧率: {fps}")
        logger.info(f"  亮度: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
        logger.info(f"  对比度: {cap.get(cv2.CAP_PROP_CONTRAST)}")
        logger.info(f"  饱和度: {cap.get(cv2.CAP_PROP_SATURATION)}")
        
        # 尝试读取几帧
        success_count = 0
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                success_count += 1
                if i == 0:  # 第一帧成功时记录信息
                    logger.info(f"  第一帧形状: {frame.shape}")
        
        logger.info(f"  成功读取 {success_count}/10 帧")
        
        cap.release()
        
        if success_count > 0:
            logger.info(f"摄像头 {device_id} 测试通过 ✓")
        else:
            logger.warning(f"摄像头 {device_id} 无法读取帧 ✗")
    
    logger.info("=== 直接摄像头测试完成 ===")

def test_camera_manager():
    """使用CameraManager测试摄像头"""
    logger.info("=== 开始CameraManager测试 ===")
    
    try:
        # 导入CameraManager
        from core.camera_manager import CameraManager, USBCamera
        
        # 创建摄像头管理器
        logger.info("创建CameraManager...")
        manager = CameraManager()
        
        # 发现摄像头
        logger.info("发现可用摄像头...")
        cameras = manager.discover_cameras()
        
        if not cameras:
            logger.warning("未发现任何摄像头")
            return
        
        logger.info(f"发现 {len(cameras)} 个摄像头:")
        for i, cam in enumerate(cameras):
            logger.info(f"  {i}: {cam.name} (ID: {cam.id}, 设备ID: {cam.device_id})")
            logger.info(f"     型号: {cam.model}, 厂商: {cam.vendor}")
            logger.info(f"     已连接: {cam.is_connected}")
        
        # 测试第一个摄像头
        if cameras:
            first_camera = cameras[0]
            camera_id = first_camera.id
            
            logger.info(f"\n测试摄像头: {first_camera.name}")
            
            # 连接摄像头
            logger.info("连接摄像头...")
            if manager.connect_camera(camera_id, "usb", device_id=first_camera.device_id):
                logger.info("摄像头连接成功")
                
                # 获取摄像头信息
                info = manager.get_camera_info(camera_id)
                if info:
                    logger.info(f"摄像头信息:")
                    logger.info(f"  名称: {info.name}")
                    logger.info(f"  分辨率: {info.current_settings.resolution}")
                    logger.info(f"  帧率: {info.current_settings.fps}")
                
                # 开始捕获
                logger.info("开始视频捕获...")
                if manager.start_capture():
                    logger.info("视频捕获已开始")
                    
                    # 尝试捕获几帧
                    for i in range(5):
                        frame = manager.capture_frame()
                        if frame is not None:
                            logger.info(f"  第 {i+1} 帧: 形状={frame.shape}, 类型={frame.dtype}")
                            
                            # 保存第一帧作为测试
                            if i == 0:
                                test_dir = Path("test_output")
                                test_dir.mkdir(exist_ok=True)
                                test_file = test_dir / "test_frame.jpg"
                                cv2.imwrite(str(test_file), frame)
                                logger.info(f"  测试帧已保存到: {test_file}")
                        else:
                            logger.warning(f"  第 {i+1} 帧: 捕获失败")
                    
                    # 停止捕获
                    logger.info("停止视频捕获...")
                    manager.stop_capture()
                else:
                    logger.error("视频捕获启动失败")
                
                # 断开连接
                logger.info("断开摄像头连接...")
                if manager.disconnect_camera(camera_id):
                    logger.info("摄像头已断开")
                else:
                    logger.error("断开摄像头失败")
            else:
                logger.error("摄像头连接失败")
        
        # 清理
        manager.cleanup()
        logger.info("CameraManager清理完成")
        
    except Exception as e:
        logger.error(f"CameraManager测试失败: {e}", exc_info=True)
    
    logger.info("=== CameraManager测试完成 ===")

def test_usb_camera_direct():
    """直接测试USBCamera类"""
    logger.info("=== 开始USBCamera类直接测试 ===")
    
    try:
        from core.camera_manager import USBCamera
        
        # 测试前几个设备
        for device_id in range(3):
            logger.info(f"\n测试USB摄像头设备 {device_id}...")
            camera_id = f"test_usb_{device_id}"
            
            # 创建摄像头实例
            camera = USBCamera(camera_id, device_id)
            
            # 测试可用性
            if camera.is_available():
                logger.info(f"摄像头 {device_id} 可用")
                
                # 连接
                if camera.connect():
                    logger.info("摄像头连接成功")
                    
                    # 获取信息
                    info = camera.get_info()
                    logger.info(f"摄像头信息: {info.name}, {info.model}")
                    
                    # 获取设置
                    settings = camera.get_settings()
                    logger.info(f"当前设置: 分辨率={settings.resolution}, 帧率={settings.fps}")
                    
                    # 修改设置
                    new_settings = settings
                    new_settings.resolution = (800, 600)
                    new_settings.fps = 15
                    
                    if camera.set_settings(new_settings):
                        logger.info("摄像头设置更新成功")
                    
                    # 开始捕获
                    if camera.start_capture():
                        logger.info("视频捕获开始")
                        
                        # 获取几帧
                        for i in range(3):
                            frame_info = camera.get_frame()
                            if frame_info:
                                logger.info(f"  第 {i+1} 帧: ID={frame_info.frame_id}, 大小={frame_info.frame_size}")
                                
                                # 显示第一帧
                                if i == 0 and frame_info.frame is not None:
                                    # 使用OpenCV显示
                                    cv2.imshow(f"摄像头 {device_id} - 测试", frame_info.frame)
                                    cv2.waitKey(500)  # 显示500ms
                                    cv2.destroyAllWindows()
                            else:
                                logger.warning(f"  第 {i+1} 帧: 无数据")
                        
                        # 停止捕获
                        camera.stop_capture()
                        logger.info("视频捕获停止")
                    else:
                        logger.error("视频捕获启动失败")
                    
                    # 断开连接
                    camera.disconnect()
                    logger.info("摄像头已断开")
                else:
                    logger.error("摄像头连接失败")
            else:
                logger.warning(f"摄像头 {device_id} 不可用")
    
    except Exception as e:
        logger.error(f"USBCamera测试失败: {e}", exc_info=True)
    
    logger.info("=== USBCamera类直接测试完成 ===")

def interactive_test():
    """交互式摄像头测试"""
    logger.info("=== 交互式摄像头测试 ===")
    
    print("\n" + "="*50)
    print("摄像头交互式测试")
    print("="*50)
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 直接OpenCV测试")
    print("2. CameraManager测试")
    print("3. USBCamera类测试")
    print("4. 全部测试")
    print("0. 退出")
    
    choice = input("\n请输入选择 (0-4): ").strip()
    
    if choice == "0":
        print("退出测试")
        return
    elif choice == "1":
        test_camera_direct()
    elif choice == "2":
        test_camera_manager()
    elif choice == "3":
        test_usb_camera_direct()
    elif choice == "4":
        test_camera_direct()
        test_camera_manager()
        test_usb_camera_direct()
    else:
        print("无效选择")
    
    print("\n测试完成！")
    input("按Enter键退出...")

if __name__ == "__main__":
    # 直接运行所有测试
    print("摄像头独立测试程序")
    print("=" * 40)
    
    try:
        # 运行所有测试
        test_camera_direct()
        print()
        test_camera_manager()
        print()
        test_usb_camera_direct()
        
        print("\n所有测试完成！")
        
        # 询问是否进行交互测试
        response = input("\n是否进行交互式测试? (y/n): ").strip().lower()
        if response == 'y':
            interactive_test()
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()