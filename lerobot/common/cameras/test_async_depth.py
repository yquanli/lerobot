import time
import cv2
import numpy as np
from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.common.utils.utils import TimerManager
from lerobot.common.cameras.configs import ColorMode
import logging

# 配置日志记录，以便查看潜在的错误
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_multi_camera_verification():
    # 1. 初始化多个相机配置
    # 在这里添加您所有需要测试的相机
    camera_configs = {
        "cam_1": RealSenseCameraConfig(
            serial_number_or_name="137322074822", 
            fps=30,
            width=640,
            height=480,
            use_depth=True,
            # color_mode=ColorMode.BGR
        ),
        "cam_2": RealSenseCameraConfig(
            serial_number_or_name="137322078934", 
            fps=30,
            width=640,
            height=480,
            use_depth=True,
            # color_mode=ColorMode.BGR
        ),
    }

    cameras = {}
    
    try:
        # --- 连接所有相机 ---
        logging.info("正在连接所有相机...")
        for name, config in camera_configs.items():
            try:
                cam = RealSenseCamera(config)
                cam.connect()
                cameras[name] = cam
                logging.info(f"相机 '{name}' (SN: {config.serial_number_or_name}) 连接成功！")
            except Exception as e:
                logging.error(f"连接相机 '{name}' (SN: {config.serial_number_or_name}) 失败: {e}")
        
        if not cameras:
            logging.error("没有成功连接任何相机，测试终止。")
            return

        # --- 检验 1: 功能正确性 ---
        print("\n--- 检验 1: 功能正确性 ---")
        all_passed = True
        for name, camera in cameras.items():
            try:
                # 预热异步线程
                _ = camera.async_read()
                depth_frame = camera.async_read_depth()
                
                logging.info(f"相机 '{name}': 成功获取深度图！")
                logging.info(f" - 数据类型 (dtype): {depth_frame.dtype}")
                logging.info(f" - 图像形状 (shape): {depth_frame.shape}")

                assert depth_frame.dtype == np.uint16
                assert depth_frame.shape == (camera.height, camera.width)
                logging.info(f"✅ 相机 '{name}': 功能正确性检验通过！")
                
            except Exception as e:
                logging.error(f"❌ 相机 '{name}': 功能正确性检验失败: {e}")
                all_passed = False
        
        if not all_passed:
            return

        # --- 检验 2: 性能优势 ---
        print("\n--- 检验 2: 性能优势对比 ---")
        num_iterations = 100
        
        # 测试同步读取性能
        sync_timer = TimerManager("同步读取")
        for _ in range(num_iterations):
            with sync_timer:
                for camera in cameras.values():
                    _ = camera.read()
                    _ = camera.read_depth()
        print(f"同步模式下 {len(cameras)}个相机的平均总帧率 (FPS): {sync_timer.fps_avg:.2f} Hz")

        # 测试异步读取性能
        async_timer = TimerManager("异步读取")
        for _ in range(num_iterations):
            with async_timer:
                for camera in cameras.values():
                    _ = camera.async_read()
                    _ = camera.async_read_depth()
        print(f"异步模式下 {len(cameras)}个相机的平均总帧率 (FPS): {async_timer.fps_avg:.2f} Hz")
        
        if async_timer.fps_avg > sync_timer.fps_avg * 1.5:
             print("✅ 性能优势检验通过！异步模式显著更快。")
        else:
             print("⚠️ 性能优势不明显或测试失败。")

        # --- 检验 3: 数据同步性 (可视化) ---
        print("\n--- 检验 3: 数据同步性 (可视化) ---")
        print("将显示所有相机的实时图像。快速移动物体在镜头前。")
        print("每个相机的彩色图和深度图应保持同步。按 'q' 键退出。")

        while True:
            display_images = []
            for name, camera in cameras.items():
                color_image_rgb = camera.async_read()
                depth_image = camera.async_read_depth()

                # 为了让 cv2.imshow 正确显示，将 RGB 转换为 BGR
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # 在图像上标注相机名称
                cv2.putText(color_image_bgr, f"{name} Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(depth_colormap, f"{name} Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 将彩色图和深度图水平拼接
                combined = np.hstack((color_image_bgr, depth_colormap))
                display_images.append(combined)

            # 将所有相机的图像垂直拼接起来显示
            if display_images:
                full_display = np.vstack(display_images)
                cv2.imshow('Multi-Camera View - Press Q to quit', full_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("✅ 数据同步性检验完成。")

    finally:
        logging.info("\n正在断开所有相机连接...")
        for name, camera in cameras.items():
            if camera.is_connected:
                camera.disconnect()
                logging.info(f"相机 '{name}' 已断开。")
        cv2.destroyAllWindows()
        print("测试完成，所有资源已释放。")

if __name__ == "__main__":
    run_multi_camera_verification()