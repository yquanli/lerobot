import pyrealsense2 as rs
import numpy as np
import cv2
import time
from typing import Optional

# --- 辅助类与常量定义 ---

class FPSCounter:
    """一个简单的类，用于计算和管理帧率。"""
    def __init__(self):
        self._start_time = time.time()
        self._frame_count = 0
        self.fps = 0

    def tick(self):
        """每处理一帧时调用此方法。"""
        self._frame_count += 1
        elapsed_time = time.time() - self._start_time
        if elapsed_time > 1.0:
            self.fps = self._frame_count / elapsed_time
            self._start_time = time.time()
            self._frame_count = 0

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_GREEN = (0, 255, 0)

# --- 主程序 ---

def select_realsense_device() -> Optional[str]:
    """列出所有可用的RealSense设备，并让用户选择一个。"""
    context = rs.context()
    devices = context.query_devices()
    if not devices:
        print("错误：未检测到RealSense设备。请检查相机连接。")
        return None

    print(f"发现 {len(devices)} 个RealSense设备:")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"  [{i}]: {name} (序列号: {serial})")
    
    while True:
        try:
            choice = int(input(f"请输入您想使用的设备编号 (0-{len(devices)-1}): "))
            if 0 <= choice < len(devices):
                selected_serial = devices[choice].get_info(rs.camera_info.serial_number)
                print(f"\n您已选择: {devices[choice].get_info(rs.camera_info.name)} (S/N: {selected_serial})")
                return selected_serial
            else:
                print("输入无效，请输入列表中的编号。")
        except ValueError:
            print("输入无效，请输入一个数字。")

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    serial_number = select_realsense_device()
    if serial_number is None: return
    config.enable_device(serial_number)

    W, H = 640, 480
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    print("\n正在启动相机管道...")
    pipeline.start(config)
    print("相机管道已成功启动！")

    # (核心修改) 创建两个对齐对象
    align_to_color = rs.align(rs.stream.color)
    align_to_depth = rs.align(rs.stream.depth)
    
    colorizer = rs.colorizer()
    
    window_name = 'RealSense 双向对齐对比 | 按 Q 退出'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, W * 2, H * 2)

    fps_counters = {
        'orig_color': FPSCounter(), 'orig_depth': FPSCounter(),
        'aligned_color_to_depth': FPSCounter(), 'aligned_depth_to_color': FPSCounter()
    }
    last_ts_print_time = time.time()

    try:
        print("\n开始捕获帧...")
        while True:
            frames = pipeline.wait_for_frames()

            # 进行两次对齐处理
            aligned_to_color_frames = align_to_color.process(frames)
            aligned_to_depth_frames = align_to_depth.process(frames)

            frame_dict = {
                'orig_color': frames.get_color_frame(),
                'orig_depth': frames.get_depth_frame(),
                'aligned_color_to_depth': aligned_to_depth_frames.get_color_frame(),
                'aligned_depth_to_color': aligned_to_color_frames.get_depth_frame()
            }

            if not all(frame_dict.values()):
                continue

            images_to_display = {}
            for key, frame in frame_dict.items():
                fps_counters[key].tick()
                if 'depth' in key:
                    images_to_display[key] = np.asanyarray(colorizer.colorize(frame).get_data()).copy()
                else:
                    images_to_display[key] = np.asanyarray(frame.get_data()).copy()

            if time.time() - last_ts_print_time >= 1.0:
                color_ts = frame_dict['orig_color'].get_timestamp()
                depth_ts = frame_dict['orig_depth'].get_timestamp()
                aligned_depth_ts = frame_dict['aligned_depth_to_color'].get_timestamp()
                print(f"彩色帧时间戳: {color_ts:.4f} ms | 深度帧时间戳: {depth_ts:.4f} ms | 差值: {abs(color_ts - depth_ts):.4f} ms | 对齐后差值：{abs(color_ts - aligned_depth_ts):.4f} ms" )
            
                last_ts_print_time = time.time()

            elapsed_time = time.time() - fps_counters['orig_color']._start_time
            if elapsed_time > 1.0:
                for key in fps_counters:
                    fps_counters[key].fps = fps_counters[key]._frame_count / elapsed_time
                    fps_counters[key]._frame_count = 0
                fps_counters['orig_color']._start_time = time.time()

            # (核心修改) 更新所有标签以反映双向对齐
            cv2.putText(images_to_display['orig_color'], f"Original RGB FPS: {fps_counters['orig_color'].fps:.2f}", (10, 30), FONT, FONT_SCALE, TEXT_COLOR_WHITE, FONT_THICKNESS)
            cv2.putText(images_to_display['orig_depth'], f"Original Depth FPS: {fps_counters['orig_depth'].fps:.2f}", (10, 30), FONT, FONT_SCALE, TEXT_COLOR_WHITE, FONT_THICKNESS)
            cv2.putText(images_to_display['aligned_color_to_depth'], f"Aligned Color (to Depth) FPS: {fps_counters['aligned_color_to_depth'].fps:.2f}", (10, 30), FONT, FONT_SCALE, TEXT_COLOR_GREEN, FONT_THICKNESS)
            cv2.putText(images_to_display['aligned_depth_to_color'], f"Aligned Depth (to Color) FPS: {fps_counters['aligned_depth_to_color'].fps:.2f}", (10, 30), FONT, FONT_SCALE, TEXT_COLOR_GREEN, FONT_THICKNESS)
            
            top_row = np.hstack((images_to_display['orig_color'], images_to_display['orig_depth']))
            bottom_row = np.hstack((images_to_display['aligned_color_to_depth'], images_to_display['aligned_depth_to_color']))
            display_image = np.vstack((top_row, bottom_row))

            cv2.imshow(window_name, display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("\n正在停止相机管道...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("测试完成。")

if __name__ == "__main__":
    main()