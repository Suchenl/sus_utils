import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def check_single_file(args):
    """
    工作线程函数：校验单个文件是否存在
    参数 args: (行索引, 相对路径, 根目录)
    返回值: 如果缺失返回 dict, 否则返回 None
    """
    index, relative_path, dataset_root = args
    
    # 排除空值
    if pd.isna(relative_path):
        return None
        
    # 拼接绝对路径
    full_path = os.path.join(dataset_root, str(relative_path).lstrip('/'))
    
    if not os.path.exists(full_path):
        return {
            "line": index + 2, # CSV 行号 (1-based + header)
            "path": relative_path
        }
    return True # 存在标志

def verify_dataset_integrity_parallel(csv_path, dataset_root, path_column='video', max_workers=32):
    """
    多线程检查 CSV 中记录的相对路径在物理磁盘上是否存在。
    
    参数:
    - csv_path: CSV 文件的路径
    - dataset_root: 数据的根目录（绝对路径）
    - path_column: CSV 中存储相对路径的列名
    - max_workers: 线程池大小。SSD建议 32-64，机械硬盘或网络挂载建议 12-24。
    """
    
    # 1. 加载 CSV
    if not os.path.exists(csv_path):
        print(f"错误: 找不到 CSV 文件 -> {csv_path}")
        return
    
    print(f"正在加载 CSV: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if path_column not in df.columns:
        print(f"错误: CSV 中不存在列名 '{path_column}'。可选列有: {list(df.columns)}")
        return

    total_records = len(df)
    print(f"开始并行校验... 总记录数: {total_records} | 线程数: {max_workers}")
    print(f"根目录: {dataset_root}")

    missing_files = []
    valid_count = 0
    skipped_count = 0

    # 2. 准备任务
    # 将 DataFrame 转换为元组列表以加快多线程分发速度
    tasks = [(i, row[path_column], dataset_root) for i, row in df.iterrows()]

    # 3. 执行多线程校验
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(check_single_file, task) for task in tasks]
        
        # 使用 tqdm 实时显示进度
        for future in tqdm(as_completed(futures), total=total_records, desc="校验进度"):
            result = future.result()
            
            if result is True:
                valid_count += 1
            elif result is None:
                skipped_count += 1
            else:
                missing_files.append(result)

    # 4. 输出报告
    print("\n" + "="*40)
    print("校验完成报告 (Multi-threaded):")
    print(f"检查总数: {total_records}")
    print(f"文件存在: {valid_count}")
    print(f"跳过空行: {skipped_count}")
    
    if len(missing_files) == 0:
        print("状态: [完美] 所有文件均在磁盘上找到！")
    else:
        print(f"状态: [异常] 发现 {len(missing_files)} 个文件缺失。")
        print("-" * 40)
        
        # 按行号排序以便查看
        missing_files.sort(key=lambda x: x['line'])
        
        print("前 10 个缺失文件示例:")
        for item in missing_files[:10]:
            print(f"行号 {item['line']}: {item['path']}")
        
        # 将缺失列表保存
        missing_df = pd.DataFrame(missing_files)
        report_name = "missing_files_report.csv"
        missing_df.to_csv(report_name, index=False)
        print("-" * 40)
        print(f"完整缺失报告已保存至: {report_name}")
    print("="*40)

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置信息
    CSV_FILE = "/m2v_intern/chenyuzhuo03/DATASETS/Videos/OpenVid-1M/data/train/OpenVid-1M-available.csv"
    ROOT_PATH = "/m2v_intern/public_datasets/OpenVid-1M/data"
    COLUMN_NAME = "video"
    
    # 线程数设置建议：
    # 如果数据在本地 SSD: 64
    # 如果数据在 NFS/NAS 网络挂载: 24 ~ 32 (避免并发请求过多导致网络拥塞)
    verify_dataset_integrity_parallel(CSV_FILE, ROOT_PATH, COLUMN_NAME, max_workers=32)