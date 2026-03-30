import pandas as pd
import os
from tqdm import tqdm

def verify_dataset_integrity(csv_path, dataset_root, path_column='video'):
    """
    检查 CSV 中记录的相对路径在物理磁盘上是否存在。
    
    参数:
    - csv_path: CSV 文件的路径
    - dataset_root: 数据的根目录（绝对路径）
    - path_column: CSV 中存储相对路径的列名 (例如 'video' 或 'file_path')
    """
    
    # 1. 加载 CSV
    if not os.path.exists(csv_path):
        print(f"错误: 找不到 CSV 文件 -> {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if path_column not in df.columns:
        print(f"错误: CSV 中不存在列名 '{path_column}'。可选列有: {list(df.columns)}")
        return

    print(f"开始校验... 总记录数: {len(df)}")
    print(f"根目录: {dataset_root}")

    missing_files = []
    total_count = 0
    
    # 2. 遍历检查 (使用 tqdm 显示进度条)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="正在校验路径"):
        relative_path = row[path_column]
        
        # 排除空值
        if pd.isna(relative_path):
            print(f"警告: 第 {index} 行路径为空，已跳过。")
            continue
            
        # 拼接绝对路径
        full_path = os.path.join(dataset_root, relative_path.lstrip('/'))
        
        total_count += 1
        if not os.path.exists(full_path):
            missing_files.append({
                "line": index + 2, # CSV 行号 (1-based + header)
                "path": relative_path
            })

    # 3. 输出报告
    print("\n" + "="*30)
    print("校验完成报告:")
    print(f"检查总数: {total_count}")
    
    if len(missing_files) == 0:
        print("状态: [完美] 所有文件均在磁盘上找到！")
    else:
        print(f"状态: [异常] 发现 {len(missing_files)} 个文件缺失。")
        print("-" * 30)
        print("前 10 个缺失文件示例:")
        for item in missing_files[:10]:
            print(f"行号 {item['line']}: {item['path']}")
        
        # 可选：将缺失列表保存
        if len(missing_files) > 0:
            missing_df = pd.DataFrame(missing_files)
            missing_df.to_csv("missing_files_report.csv", index=False)
            print("-" * 30)
            print(f"完整报告已保存至: missing_files_report.csv")
    print("="*30)

# --- 使用示例 ---
# 假设你的视频存放在 /home/data/panda70m/videos/
# CSV 里的路径是 videos/001.mp4
# 则 dataset_root 应该设为 /home/data/panda70m/
csv_file = "/m2v_intern/chenyuzhuo03/DATASETS/Videos/OpenVid-1M/data/train/OpenVidHD.csv"     # 你的 CSV 路径
root_path = "/m2v_intern/public_datasets/OpenVid-1M/data"   # 你的数据集绝对路径根目录
column_to_check = "video"           # CSV 里的列名

verify_dataset_integrity(csv_file, root_path, column_to_check)