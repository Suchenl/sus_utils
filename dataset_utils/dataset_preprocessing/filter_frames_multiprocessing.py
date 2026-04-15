import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def filter_chunk(chunk_data):
    """
    进程池中的子进程执行函数
    """
    df_chunk, frame_threshold, frame_column = chunk_data
    # 过滤大于等于阈值的
    valid = df_chunk[df_chunk[frame_column] >= frame_threshold]
    # 过滤小于阈值的
    short = df_chunk[df_chunk[frame_column] < frame_threshold]
    return valid, short

def parallel_filter_csv(input_csv, output_valid='valid_frames.csv', 
                        output_short='short_frames.csv', 
                        frame_column='frame', 
                        threshold=81, 
                        num_processes=None):
    """
    并行过滤 CSV 文件
    :param num_processes: 进程数，默认使用所有 CPU 核心
    """
    if not os.path.exists(input_csv):
        print(f"Error: File not found -> {input_csv}")
        return

    # 1. 估算进程数
    if num_processes is None:
        num_processes = cpu_count()

    print(f"--- Starting Parallel Filtering ---")
    print(f"Input: {input_csv}")
    print(f"Threshold: >= {threshold} frames")
    print(f"Using {num_processes} processes...")

    # 2. 分块读取 CSV
    # 对于超大文件，我们不一次性读入，而是分块处理
    chunk_size = 100000  # 每块 10 万行
    reader = pd.read_csv(input_csv, chunksize=chunk_size)
    
    # 准备任务
    tasks = []
    for chunk in reader:
        tasks.append((chunk, threshold, frame_column))

    # 3. 启动进程池
    valid_chunks = []
    short_chunks = []
    
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示分块处理进度
        for v_chunk, s_chunk in tqdm(pool.imap(filter_chunk, tasks), total=len(tasks), desc="Processing chunks"):
            valid_chunks.append(v_chunk)
            short_chunks.append(s_chunk)

    # 4. 合并并保存结果
    print("Merging and saving results...")
    
    df_valid_final = pd.concat(valid_chunks, ignore_index=True)
    df_short_final = pd.concat(short_chunks, ignore_index=True)

    df_valid_final.to_csv(output_valid, index=False)
    df_short_final.to_csv(output_short, index=False)

    # 5. 输出总结报告
    print("\n" + "="*40)
    print("Filtering Task Completed!")
    print(f"Total processed: {len(df_valid_final) + len(df_short_final)}")
    print(f"Valid (>= {threshold}): {len(df_valid_final)} rows -> Saved to {output_valid}")
    print(f"Short (< {threshold}): {len(df_short_final)} rows -> Saved to {output_short}")
    print(f"Retention Rate: {(len(df_valid_final) / (len(df_valid_final) + len(df_short_final)) * 100):.2f}%")
    print("="*40)

# --- 运行设置 ---
if __name__ == "__main__":
    # 配置你的文件路径
    INPUT_CSV = "/m2v_intern/chenyuzhuo03/DATASETS/Videos/OpenVid-1M/data/metadata/OpenVid-1M-available.csv" 
    
    # 执行过滤
    parallel_filter_csv(
        input_csv=INPUT_CSV,
        output_valid="OpenVid-1M-available-33plus.csv", # 满足要求的存这里
        output_short="OpenVid-1M-available-shorter_than_33.csv", # 太短的存这里
        frame_column="frame", # 你 CSV 中的帧数键名
        threshold=33,         # 阈值
        num_processes=32      # 进程数
    )