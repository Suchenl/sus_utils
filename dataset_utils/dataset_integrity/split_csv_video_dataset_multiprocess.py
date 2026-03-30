import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_file_status(item):
    """
    单个路径检查函数，供线程池调用
    返回: (原始索引, 是否存在)
    """
    idx, root, rel_path = item
    if pd.isna(rel_path):
        return idx, False
    
    # 拼接路径
    full_path = os.path.join(root, str(rel_path).lstrip('/'))
    return idx, os.path.exists(full_path)

def parallel_split_csv(input_csv, dataset_root, path_column='video', 
                       available_output='train_manifest.csv', 
                       missing_output='broken_links.csv',
                       max_workers=16):
    """
    使用多线程并行检查文件存在性并拆分 CSV
    :param max_workers: 线程数，建议根据硬盘类型设置 (SSD可设 32+, HDD 建议 8-16)
    """
    
    if not os.path.exists(input_csv):
        print(f"错误: 找不到输入 CSV -> {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    print(f"--- 正在并行处理: {input_csv} ---")
    print(f"数据总量: {len(df)} | 线程数: {max_workers}")

    # 准备任务数据：(索引, 根目录, 相对路径)
    tasks = [(i, dataset_root, row[path_column]) for i, row in df.iterrows()]
    
    # 结果占位符
    results_mask = [False] * len(df)

    # 使用线程池执行并行检查
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 包裹 map 以显示进度
        # as_completed 允许结果一出来就处理，但我们需要保持顺序，所以记录索引
        futures = [executor.submit(check_file_status, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="并行校验中"):
            idx, exists = future.result()
            results_mask[idx] = exists

    # 执行拆分
    df_available = df[results_mask]
    df_missing = df[~pd.Series(results_mask)]

    # 保存结果
    df_available.to_csv(available_output, index=False)
    df_missing.to_csv(missing_output, index=False)

    print("\n" + "="*40)
    print("并行处理完成！")
    print(f"1. 可用数据: {len(df_available)} 条")
    print(f"2. 缺失数据: {len(df_missing)} 条")
    print(f"可用率: {(len(df_available) / len(df) * 100):.2f}%")
    print(f"结果已存至: {available_output} 和 {missing_output}")
    print("="*40)

# --- 运行设置 ---
if __name__ == "__main__":
    INPUT_CSV = "/m2v_intern/chenyuzhuo03/DATASETS/Videos/OpenVid-1M/data/train/OpenVid-1M.csv"             # 原始 CSV 文件名
    DATASET_ROOT = "/m2v_intern/public_datasets/OpenVid-1M/data"      # 视频存放的绝对路径根目录
    COLUMN_NAME = "video"           # 路径列
    
    # 线程数设置建议：
    # - 普通机械硬盘 (HDD): 8-16 (多了会因为磁头寻道变慢)
    # - 固态硬盘 (SSD): 32-64 (速度极快)
    # - 网络存储 (NFS/NAS): 16-32 (取决于网络带宽)
    parallel_split_csv(INPUT_CSV, DATASET_ROOT, COLUMN_NAME, max_workers=32)
