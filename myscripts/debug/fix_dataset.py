"""
修复数据集元数据的脚本

检查并修复 dataset_to_index 与实际数据不匹配的问题
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json
from pathlib import Path
import pandas as pd


def fix_dataset_metadata(repo_id: str, root: str | None = None):
    """修复数据集元数据"""
    
    # 加载数据集
    dataset = LeRobotDataset(repo_id, root=root)
    
    print(f"📊 数据集信息:")
    print(f"  Repo ID: {repo_id}")
    print(f"  总帧数: {len(dataset)}")
    print(f"  Episode 数: {len(dataset.meta.episodes)}")
    
    # 检查每个 episode 的索引
    print(f"\n🔍 检查 Episode 索引:")
    
    # ⭐ 修复：判断 episodes 的类型并转换为 DataFrame
    episodes = dataset.meta.episodes
    
    # 检查类型并打印调试信息
    print(f"  Episodes 类型: {type(episodes)}")
    
    if isinstance(episodes, pd.DataFrame):
        # 如果已经是 DataFrame，直接复制
        episodes_df = episodes.copy()
    else:
        # 如果是 HuggingFace Dataset，转换为 DataFrame
        try:
            # 方法 1：使用 to_pandas()
            episodes_df = episodes.to_pandas()
        except AttributeError:
            try:
                # 方法 2：手动构建 DataFrame
                episodes_dict = {
                    'episode_index': episodes['episode_index'],
                    'dataset_from_index': episodes['dataset_from_index'],
                    'dataset_to_index': episodes['dataset_to_index'],
                    'length': episodes['length'],
                }
                episodes_df = pd.DataFrame(episodes_dict)
            except Exception as e:
                print(f"❌ 无法转换 episodes 为 DataFrame: {e}")
                print(f"   Episodes 结构: {episodes}")
                return
    
    print(f"  转换后的 DataFrame 形状: {episodes_df.shape}")
    print(f"  列名: {episodes_df.columns.tolist()}")
    
    has_error = False
    
    for idx, ep in episodes_df.iterrows():
        ep_idx = ep['episode_index']
        from_idx = ep['dataset_from_index']
        to_idx = ep['dataset_to_index']
        length = ep['length']
        
        # 检查 to_idx 是否超出范围
        if to_idx > len(dataset):
            print(f"  ❌ Episode {ep_idx}: to_idx={to_idx} > 总帧数={len(dataset)}")
            has_error = True
            
            # 修复：将 to_idx 设为实际的最大索引
            corrected_to_idx = len(dataset)
            corrected_length = corrected_to_idx - from_idx
            
            print(f"     修复建议: to_idx={corrected_to_idx}, length={corrected_length}")
            
            # 更新 DataFrame
            episodes_df.at[idx, 'dataset_to_index'] = corrected_to_idx
            episodes_df.at[idx, 'length'] = corrected_length
        else:
            print(f"  ✅ Episode {ep_idx}: [{from_idx}, {to_idx}), length={length}")
    
    if has_error:
        print(f"\n⚠️  发现元数据错误，正在修复...")
        
        # 找到数据集的本地路径
        from huggingface_hub import snapshot_download
        local_dir = Path(snapshot_download(repo_id, repo_type='dataset'))
        
        print(f"\n📂 数据集本地路径: {local_dir}")
        
        # 查找 episodes.jsonl 或 meta.json
        possible_files = [
            local_dir / "meta" / "episodes.jsonl",
            local_dir / "episodes.jsonl",
            local_dir / "meta" / "info.json",
            local_dir / "meta.json",
            local_dir / "info.json",
        ]
        
        meta_file = None
        for file_path in possible_files:
            if file_path.exists():
                meta_file = file_path
                print(f"  ✅ 找到元数据文件: {file_path}")
                break
        
        if meta_file is None:
            print(f"  ❌ 未找到元数据文件，尝试的路径:")
            for p in possible_files:
                print(f"     - {p}")
            print(f"\n💡 建议：手动查找或重新生成数据集")
            return
        
        # 备份原始文件
        backup_path = meta_file.with_suffix(meta_file.suffix + '.backup')
        import shutil
        shutil.copy(meta_file, backup_path)
        print(f"  📦 已备份原始文件到: {backup_path}")
        
        # 根据文件类型进行处理
        if meta_file.suffix == '.jsonl':
            # 处理 JSONL 格式（episodes.jsonl）
            print(f"\n  处理 JSONL 格式...")
            
            # 写入修复后的 episodes
            with open(meta_file, 'w') as f:
                for _, row in episodes_df.iterrows():
                    episode_dict = row.to_dict()
                    f.write(json.dumps(episode_dict) + '\n')
            
            print(f"  ✅ 已保存修复后的 episodes.jsonl")
            
        elif meta_file.suffix == '.json':
            # 处理 JSON 格式（meta.json 或 info.json）
            print(f"\n  处理 JSON 格式...")
            
            # 读取原始文件
            with open(meta_file, 'r') as f:
                meta_dict = json.load(f)
            
            # 更新 episodes 信息
            meta_dict['episodes'] = episodes_df.to_dict('records')
            
            # 写回文件
            with open(meta_file, 'w') as f:
                json.dump(meta_dict, f, indent=2)
            
            print(f"  ✅ 已保存修复后的 {meta_file.name}")
        
        print(f"\n🎉 修复完成！")
        print(f"\n⚠️  重要：需要清除 HuggingFace 缓存才能生效！")
        print(f"\n运行以下命令清除缓存:")
        print(f"  rm -rf ~/.cache/huggingface/datasets/{repo_id.replace('/', '___')}")
        print(f"\n然后重新加载数据集即可。")
        
        # 打印修复后的 episodes 信息
        print(f"\n📊 修复后的 Episode 信息:")
        print(episodes_df)
        
    else:
        print(f"\n✅ 元数据检查通过，无需修复。")
        print(f"\n📊 Episode 信息:")
        print(episodes_df)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="修复数据集元数据")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集 repo ID")
    parser.add_argument("--root", type=str, default=None, help="数据集根目录")
    
    args = parser.parse_args()
    
    fix_dataset_metadata(args.repo_id, args.root)