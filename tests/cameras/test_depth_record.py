import cv2
import argparse
import json
from pathlib import Path

def verify_recorded_depth(dataset_path: Path, episode_index: int = 0):
    """
    一个独立的验证脚本，用于读取LeRobot录制的数据集中的深度视频，
    并进行可视化和数据分析，以验证其正确性。
    """
    print(f"--- LeRobot 录制深度数据验证开始 ---")
    print(f"正在检查数据集: {dataset_path}")

    # --- 1. 定位并检查文件路径 ---
    # (注意：这里的 'wrist' 是默认相机名，如果您的配置不同，请修改此处)
    camera_name = 'wrist' 
    episode_str = f"episode_{episode_index:06d}.mp4"
    depth_video_path = dataset_path / "videos" / "chunk-000" / f"observation.depth.{camera_name}" / episode_str
    episodes_meta_path = dataset_path / "meta" / "episodes.jsonl"

    if not depth_video_path.exists():
        print(f"\n错误：未找到深度视频文件！")
        print(f"预期路径: {depth_video_path}")
        return

    if not episodes_meta_path.exists():
        print(f"\n错误：未找到元数据文件 'episodes.jsonl'！")
        print(f"预期路径: {episodes_meta_path}")
        return
        
    print(f"\n成功定位到深度视频文件: {depth_video_path}")

    # --- 2. 读取视频和元数据 ---
    cap = cv2.VideoCapture(str(depth_video_path))
    if not cap.isOpened():
        print("\n错误：无法使用OpenCV打开深度视频文件。")
        return

    # 获取视频属性
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 从元数据中获取预期的帧数
    with open(episodes_meta_path, 'r') as f:
        episode_info = json.loads(f.readline())
        expected_frame_count = episode_info['length']

    print("\n--- 视频文件属性 ---")
    print(f"分辨率: {video_width}x{video_height}")
    print(f"视频总帧数: {video_frame_count}")
    print(f"元数据记录的帧数: {expected_frame_count}")
    
    if video_frame_count != expected_frame_count:
        print("警告：视频文件中的帧数与元数据记录不匹配！")
    else:
        print("帧数与元数据匹配，检查通过。")

    # --- 3. 逐帧可视化与分析 ---
    print("\n--- 开始逐帧可视化 ---")
    print("将逐帧播放深度视频。请观察灰度变化是否与录制时的物体远近一致。")
    print("按 'q' 键退出播放。")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 视频帧通常是3通道的BGR（即使内容是灰度的），我们转为单通道灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 获取当前帧的数据信息
        min_val, max_val, _, _ = cv2.minMaxLoc(gray_frame)
        
        # 在图像上绘制信息文本
        info_text = f"Frame: {frame_idx+1}/{video_frame_count} | DType: {gray_frame.dtype} | Min/Max: {min_val}/{max_val}"
        cv2.putText(gray_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Recorded Depth Video Verification", gray_frame)
        
        frame_idx += 1
        
        # 按 'q' 键退出
        if cv2.waitKey(33) & 0xFF == ord('q'): # 等待约33ms，模拟30fps播放
            break
            
    print(f"\n播放结束。共处理 {frame_idx} 帧。")

    # --- 清理 ---
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- LeRobot 录制深度数据验证结束 ---")


if __name__ == "__main__":
    # 使用 argparse 来方便地传递数据集路径
    parser = argparse.ArgumentParser(description="验证LeRobot录制的数据集中的深度视频。")
    parser.add_argument("dataset_path", type=str, help="指向录制好的数据集文件夹的路径。")
    args = parser.parse_args()
    
    verify_recorded_depth(Path(args.dataset_path))