import pandas as pd
import os
from tqdm import tqdm

def split_csv_by_existence(input_csv, dataset_root, path_column='video', 
                          available_output='available_data.csv', 
                          missing_output='missing_data.csv'):
    """
    根据磁盘上文件的真实存在性，将 CSV 拆分为两个文件。
    """
    
    # 1. 加载原始 CSV
    if not os.path.exists(input_csv):
        print(f"错误: 找不到输入 CSV -> {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    
    if path_column not in df.columns:
        print(f"错误: CSV 中不存在列名 '{path_column}'。可选列有: {list(df.columns)}")
        return

    print(f"--- 正在处理: {input_csv} ---")
    print(f"数据总量: {len(df)}")
    print(f"数据集根目录: {dataset_root}")

    # 2. 定义检查逻辑
    # 我们创建一个布尔列表来记录每一行是否可用
    is_available = []

    # 使用 tqdm 显示实时处理进度
    for index, row in tqdm(df.iterrows(), total=len(df), desc="检查文件状态"):
        relative_path = row[path_column]
        
        # 处理空路径
        if pd.isna(relative_path):
            is_available.append(False)
            continue
            
        # 拼接并检查
        # lstrip('/') 防止因为绝对路径符号导致拼接失败
        full_path = os.path.join(dataset_root, str(relative_path).lstrip('/'))
        
        if os.path.exists(full_path):
            is_available.append(True)
        else:
            is_available.append(False)

    # 3. 执行拆分
    # 利用布尔索引快速过滤数据
    df_available = df[is_available]
    df_missing = df[~pd.Series(is_available)] # ~ 表示取反

    # 4. 保存结果
    df_available.to_csv(available_output, index=False)
    df_missing.to_csv(missing_output, index=False)

    # 5. 输出总结报告
    print("\n" + "="*40)
    print("CSV 拆分任务完成！")
    print(f"1. 可用数据 (已存至 {available_output}): {len(df_available)} 条")
    print(f"2. 缺失数据 (已存至 {missing_output}): {len(df_missing)} 条")
    print(f"可用率: {(len(df_available) / len(df) * 100):.2f}%")
    print("="*40)

# --- 使用示例 ---
# 请根据你的实际情况修改以下路径
INPUT_CSV = "metadata.csv"             # 原始 CSV 文件名
DATASET_ROOT = "/mnt/data/videos/"     # 视频存放的绝对路径根目录
COLUMN_NAME = "video"                   # CSV 中记录相对路径的列名

split_csv_by_existence(
    input_csv=INPUT_CSV, 
    dataset_root=DATASET_ROOT, 
    path_column=COLUMN_NAME,
    available_output="train_manifest.csv",  # 筛选后的可用数据
    missing_output="broken_links.csv"       # 记录的坏链数据
)