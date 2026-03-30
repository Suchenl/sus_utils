import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def check_exists(args):
    """
    args: (root, rel_path)
    返回布尔值：存在为 True，不存在为 False
    """
    root, rel_path = args
    if not rel_path or pd.isna(rel_path):
        return False
    # 路径清理与拼接
    full_path = os.path.join(root, str(rel_path).lstrip('/'))
    return os.path.exists(full_path)

def fast_parallel_split(input_csv, dataset_root, path_column='video', 
                        available_output='train_manifest.csv', 
                        missing_output='broken_links.csv',
                        max_workers=12):
    
    print(f"正在读取 CSV...")
    df = pd.read_csv(input_csv)
    total = len(df)
    
    # 1. 使用 zip 代替 iterrows (速度快 100 倍)
    paths = df[path_column].tolist()
    # 预准备参数包，减少线程启动时的开销
    args_list = [(dataset_root, p) for p in paths]
    
    print(f"开始校验 {total} 个文件 (使用 {max_workers} 线程)...")
    
    # 2. 使用 executor.map 代替 submit (更节省内存，适合海量任务)
    # chunksize 可以让线程一次领取一批任务，减少交互次数
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # chunksize 建议设置为 100-500
        for res in tqdm(executor.map(check_exists, args_list, chunksize=200), 
                        total=total, desc="校验进度"):
            results.append(res)

    # 3. 快速切分
    print("校验完成，正在生成最终文件...")
    df_available = df[results]
    df_missing = df[~pd.Series(results)]

    df_available.to_csv(available_output, index=False)
    df_missing.to_csv(missing_output, index=False)

    print(f"完成！可用: {len(df_available)}, 缺失: {len(df_missing)}")

if __name__ == "__main__":
    # --- 关键参数建议 ---
    # 如果你的数据在普通硬盘/机械硬盘，线程数设为 4 - 8
    # 如果在 SSD，设为 12 - 24
    # 如果在极速 NVMe，设为 32 - 48
    INPUT_CSV = "/m2v_intern/chenyuzhuo03/DATASETS/Videos/OpenVid-1M/data/train/OpenVid-1M.csv"             # 原始 CSV 文件名
    DATASET_ROOT = "/m2v_intern/public_datasets/OpenVid-1M/data"      # 视频存放的绝对路径根目录
    COLUMN_NAME = "video"           # 路径列
    
    fast_parallel_split(
        input_csv=INPUT_CSV, 
        dataset_root=DATASET_ROOT,
        path_column=COLUMN_NAME,
        max_workers=32  # 建议先从较小的数值开始测
    )