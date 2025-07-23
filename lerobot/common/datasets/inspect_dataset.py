import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ==============================================================================
# 1. 配置您的数据集信息
# ==============================================================================
# TODO: 将此处的 'lerobot/aloha_sim_transfer_cube_human' 替换为您自己的数据集仓库ID
# 例如： 'your_hf_username/my_awesome_dataset'
# 如果您的数据集在本地，可以直接提供文件夹路径。
DATASET_REPO_ID = "Sprinng/depth_act_test"  
# ==============================================================================
# 2. 加载数据集
# ==============================================================================
print(f"正在加载数据集: {DATASET_REPO_ID} ...")
try:
    dataset = LeRobotDataset(DATASET_REPO_ID)
    print("数据集加载成功！")
except Exception as e:
    print(f"加载数据集失败，请检查仓库ID或路径是否正确。错误: {e}")
    exit()

# ==============================================================================
# 3. 检查数据集基本信息
# ==============================================================================
print("\n--- 数据集基本信息 ---")
# 获取数据集中的总帧数（frames）
print(f"总帧数: {len(dataset)}")
# LeRobotDataset 实现了 __len__ 方法，返回的是总帧数

# 如果数据集有 episodes 的概念（通常都有），可以这样获取总 episode 数量
if hasattr(dataset, 'num_episodes'):
    print(f"总 Episodes 数量: {dataset.num_episodes}")
else:
    # 另一种获取方式
    print(f"总 Episodes 数量: {len(dataset.episode_data_index['from'])}")


# ==============================================================================
# 4. 检查单个数据点（一个 episode 的第一帧）的详细内容
# ==============================================================================
print("\n--- 单个数据点（第0个episode的第0帧）的详细信息 ---")
# dataset[i] 会返回第 i 帧的数据
# 我们取第 0 帧作为例子
try:
    data_point = dataset[0]

    # data_point 是一个字典，我们遍历它的所有键值对
    for key, value in data_point.items():
        print(f"\n键 (Key): '{key}'")
        
        # 检查值的类型
        if isinstance(value, torch.Tensor):
            # 如果是 Tensor，打印其形状、数据类型和设备
            print(f"  - 类型: torch.Tensor")
            print(f"  - 形状 (Shape): {value.shape}")
            print(f"  - 数据类型 (dtype): {value.dtype}")
            print(f"  - 设备 (device): {value.device}")
        elif isinstance(value, (int, float, str, bool)):
            # 如果是基本数据类型，直接打印
             print(f"  - 类型: {type(value).__name__}")
             print(f"  - 值: {value}")
        else:
            # 其他类型
            print(f"  - 类型: {type(value).__name__}")

except IndexError:
    print("错误：无法访问索引为 0 的数据点，可能数据集为空。")
except Exception as e:
    print(f"访问数据点时发生未知错误: {e}")

print("\n数据集检查完毕。")