import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm
from collections import Counter

def get_joint_data_chunk(chunk_data):
    """
    联合提取多列数据
    """
    df_chunk, columns = chunk_data
    # 确保列都存在并删除空值
    valid_data = df_chunk[columns].dropna()
    # 转换为元组列表
    return [tuple(x) for x in valid_data.values]

def parallel_joint_analysis(input_csv, columns=['width', 'height'], num_processes=None, top_n=30):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes to analyze: {columns}")
    
    # 1. 并行读取数据
    chunk_size = 250000 
    reader = pd.read_csv(input_csv, chunksize=chunk_size)
    tasks = [(chunk, columns) for chunk in reader]

    all_samples = []
    with Pool(processes=num_processes) as pool:
        for row_list in tqdm(pool.imap_unordered(get_joint_data_chunk, tasks), total=len(tasks), desc="Extracting Data"):
            all_samples.extend(row_list)

    if not all_samples:
        print("No data extracted.")
        return

    # 2. 统计频率
    # 自动处理任意数量的键，整数去.0，浮点留2位
    formatted_results = []
    for row in all_samples:
        parts = []
        for val in row:
            v_float = float(val)
            if v_float.is_integer():
                parts.append(str(int(v_float)))
            else:
                parts.append(str(round(v_float, 2)))
        formatted_results.append(" x ".join(parts))
    
    counter = Counter(formatted_results)
    total_count = len(formatted_results)
    
    # 动态生成列名：例如 "width_height_aspect ratio"
    combined_col_name = " x ".join(columns)
    df_stats = pd.DataFrame(counter.most_common(), columns=[combined_col_name, 'count'])
    df_stats['percentage'] = (df_stats['count'] / total_count) * 100

    # 保存统计结果
    file_base = os.path.basename(input_csv).replace('.csv', '')
    if not os.path.exists(file_base):
        os.makedirs(file_base)
    cols_str = "_".join(columns).replace(" ", "_")
    csv_save_name = f"{file_base}/{cols_str}_joint_statistics.csv"
    df_stats.to_csv(csv_save_name, index=False)
    print(f"\nFull statistics saved to: {csv_save_name}")

    # 3. 打印 Top N (动态表头)
    print(f"\n{'='*70}")
    print(f"Top {top_n} Common Combinations for: {columns}")
    header_val = combined_col_name
    print(f"{'Rank':<5} | {header_val:>25} | {'Count':>10} | {'Percentage':>10}")
    for i, row in df_stats.head(top_n).iterrows():
        print(f"{i+1:<5} | {row[combined_col_name]:>25} | {int(row['count']):>10d} | {row['percentage']:>9.2f}%")
    print(f"{'='*70}\n")

    # 4. 可视化
    plot_top_n_joint(df_stats, top_n, columns, combined_col_name, file_base, total_count)

def plot_top_n_joint(df_stats, top_n, columns, combined_col_name, file_base, total_count):
    # 取前 N 个数据进行绘图
    plot_df = df_stats.head(top_n).copy()
    
    # --- 关键修改：根据 top_n 动态计算宽度 ---
    # 设定每个条形占用 0.5 英寸，最小宽度 15 英寸，最大宽度可以设为 100 以防溢出
    dynamic_width = max(15, top_n * 0.5) 
    plt.figure(figsize=(dynamic_width, 10))
    # ---------------------------------------
    
    # 使用条形图
    bars = plt.bar(plot_df[combined_col_name], plot_df['count'], color='teal', alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    max_y = max(plot_df['count'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max_y * 0.01),
                 f'{int(height)}', ha='center', va='bottom', rotation=90, fontsize=9, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    # 动态 X 轴标签
    plt.xlabel(f"Combined Keys ({combined_col_name})", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    
    # 动态标题
    display_cols = " & ".join([c.upper() for c in columns])
    title = f"Top {top_n} Joint Analysis: {display_cols}\n(Total Samples: {total_count})"
    plt.title(title, fontsize=16, pad=20)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 防止底部标签过长被截断
    plt.tight_layout()

    cols_str = "_".join(columns).replace(" ", "_")
    img_save_name = f"{file_base}/{cols_str}_top_{top_n}.png"
    plt.savefig(img_save_name, dpi=300, bbox_inches='tight') # 添加 bbox_inches='tight' 确保保存完整
    print(f"Joint distribution plot saved to: {img_save_name}")
    plt.show()

if __name__ == "__main__":
    CSV_PATH = "/m2v_intern/chenyuzhuo03/DATASETS/Videos/OpenVid-1M/preprocessing/OpenVid-1M_with_resolution-filtered_by_minimum_size.csv"
    
    # 你现在可以传入任意数量的键，系统会自动适配
    parallel_joint_analysis(
        input_csv=CSV_PATH,
        columns=["seconds", "fps"],  # 你可以换成任何你想要的列组合， 尝试改为 "seconds", "frame", "width", "height", "fps", "aspect ratio" 等查看不同结果
        top_n=100
    )