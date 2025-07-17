import time
import cv2
import numpy as np
from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.common.utils.utils import TimerManager
import logging

# 配置日志记录，以便查看潜在的错误
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_multi_camera_verification():
    # 1. 初始化多个相机配置
    camera_configs = {
        "cam_1": RealSenseCameraConfig(
            serial_number_or_name="137322074822", # <-- 替换
            fps=30,
            width=640,
            height=480,
            use_depth=True
        ),
        "cam_2": RealSenseCameraConfig(
            serial_number_or_name="137322078934", # <-- 替换
            fps=30,
            width=640,
            height=480,
            use_depth=True
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
                _ = camera.async_read()
                depth_frame = camera.async_read_depth()
                logging.info(f"相机 '{name}': 成功获取深度图！")
                assert depth_frame.dtype == np.uint16
                assert depth_frame.shape == (camera.height, camera.width)
                logging.info(f"✅ 相机 '{name}': 功能正确性检验通过！")
            except Exception as e:
                logging.error(f"❌ 相机 '{name}': 功能正确性检验失败: {e}")
                all_passed = False
        if not all_passed: return

        # --- 检验 2: 性能优势对比 ---
        print("\n--- 检验 2: 性能优势对比 ---")
        num_iterations = 100
        
        # --- 同步模式 (基准) ---
        sync_timer = TimerManager("同步读取")
        for _ in range(num_iterations):
            with sync_timer:
                for camera in cameras.values():
                    _ = camera.read()
                    _ = camera.read_depth()
        print(f"同步模式 (RGB+Depth): {sync_timer.fps_avg:.2f} Hz")

        # --- 异步模式：同时读取RGB和Depth ---
        async_both_timer = TimerManager("异步同时读取")
        for _ in range(num_iterations):
            with async_both_timer:
                for camera in cameras.values():
                    _ = camera.async_read()
                    _ = camera.async_read_depth()
        print(f"异步模式 (RGB+Depth): {async_both_timer.fps_avg:.2f} Hz")
        
        # <<< 新增检验：单独调用 async_read >>>
        print("\n--- 新增检验: 单独调用异步RGB ---")
        async_rgb_timer = TimerManager("单独异步RGB")
        for _ in range(num_iterations * 2): # 运行更多次以稳定观察结果
             with async_rgb_timer:
                for camera in cameras.values():
                    _ = camera.async_read()
        print(f"异步模式 (仅RGB):   {async_rgb_timer.fps_avg:.2f} Hz")

        # <<< 新增检验：单独调用 async_read_depth >>>
        print("\n--- 新增检验: 单独调用异步Depth ---")
        async_depth_timer = TimerManager("单独异步Depth")
        for _ in range(num_iterations * 2):
             with async_depth_timer:
                for camera in cameras.values():
                    _ = camera.async_read_depth()
        print(f"异步模式 (仅Depth): {async_depth_timer.fps_avg:.2f} Hz")
        
        if async_both_timer.fps_avg > sync_timer.fps_avg * 1.5:
             print("\n✅ 性能优势检验通过！异步模式显著更快。")
        else:
             print("\n⚠️ 性能优势不明显或测试失败。")

        # --- 检验 3: 数据同步性与独立帧率可视化 ---
        print("\n--- 检验 3: 数据同步性与独立帧率可视化 ---")
        print("将显示所有相机的实时图像和各自的帧率。按 'q' 键退出。")

        # <<< 新增：为每个流创建独立的FPS计算器 >>>
        fps_calculators = {}
        for name in cameras:
            # 为彩色图和深度图分别创建计时器
            fps_calculators[f"{name}_color"] = {"last_time": time.time(), "frame_count": 0, "fps": 0}
            fps_calculators[f"{name}_depth"] = {"last_time": time.time(), "frame_count": 0, "fps": 0}

        def update_fps(stream_name):
            """一个辅助函数，用于计算和更新特定流的FPS"""
            calc = fps_calculators[stream_name]
            calc["frame_count"] += 1
            current_time = time.time()
            elapsed = current_time - calc["last_time"]
            if elapsed >= 1.0: # 每秒更新一次FPS读数
                calc["fps"] = calc["frame_count"] / elapsed
                calc["frame_count"] = 0
                calc["last_time"] = current_time

        while True:
            display_images = []
            for name, camera in cameras.items():
                # 获取数据
                color_image_rgb = camera.async_read()
                depth_image = camera.async_read_depth()
                
                # 更新各自流的FPS计数
                update_fps(f"{name}_color")
                update_fps(f"{name}_depth")

                # 图像后处理
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # <<< 修改：在各自的子图上绘制FPS >>>
                color_fps_text = f"FPS: {fps_calculators[f'{name}_color']['fps']:.2f}"
                depth_fps_text = f"FPS: {fps_calculators[f'{name}_depth']['fps']:.2f}"
                
                # 将文本绘制在右上角
                cv2.putText(color_image_bgr, color_fps_text, (color_image_bgr.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(depth_colormap, depth_fps_text, (depth_colormap.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 在左上角仍然标注相机名称
                cv2.putText(color_image_bgr, f"{name} Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_colormap, f"{name} Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                combined = np.hstack((color_image_bgr, depth_colormap))
                display_images.append(combined)

            if display_images:
                full_display = np.vstack(display_images)
                cv2.imshow('Multi-Camera View - Press Q to quit', full_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("✅ 可视化检验完成。")

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