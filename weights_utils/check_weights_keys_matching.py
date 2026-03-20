import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def load_pytorch_weights(weight_path: str) -> Dict[str, Any]:
    """
    加载PyTorch权重文件
    
    Args:
        weight_path: 权重文件路径
    
    Returns:
        Dict[str, Any]: 权重字典
    
    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 加载失败
    """
    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    try:
        weights = torch.load(weight_path, map_location='cpu')
        # 如果是checkpoint格式(包含'state_dict'等键),则提取state_dict
        if isinstance(weights, dict) and 'state_dict' in weights:
            weights = weights['state_dict']
        return weights
    except Exception as e:
        raise RuntimeError(f"加载权重文件失败 {weight_path}: {str(e)}")


def save_model_structure(weights: Dict[str, Any], output_path: str, title: str = "模型结构"):
    """
    保存模型权重结构到文件
    
    Args:
        weights: 权重字典
        output_path: 输出文件路径
        title: 标题
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"{title}\n")
        f.write(f"{'='*80}\n")
        f.write(f"总参数数量: {len(weights)}\n\n")
        
        # 按层级组织键
        structure = {}
        for key in sorted(weights.keys()):
            parts = key.split('.')
            current = structure
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # 最后一个部分存储形状信息
            tensor = weights[key]
            if hasattr(tensor, 'shape'):
                shape_info = f"{key}: {tuple(tensor.shape)}"
            else:
                shape_info = f"{key}: {type(tensor).__name__}"
            
            if parts[-1] not in current:
                current[parts[-1]] = []
            current[parts[-1]].append(shape_info)
        
        # 递归写入结构
        def write_dict(d, indent=0):
            for key in sorted(d.keys()):
                if isinstance(d[key], dict):
                    f.write("  " * indent + f"├─ {key}/\n")
                    write_dict(d[key], indent + 1)
                elif isinstance(d[key], list):
                    for item in d[key]:
                        f.write("  " * indent + f"└─ {item}\n")
        
        write_dict(structure)
        f.write(f"{'='*80}\n")


def check_weights_keys_matching(
    weight_path1: str,
    weight_path2: str,
    verbose: bool = True,
    check_shapes: bool = True,
    save_structure: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    检查两个PyTorch权重文件的键是否匹配
    
    Args:
        weight_path1: 第一个权重文件路径
        weight_path2: 第二个权重文件路径
        verbose: 是否打印详细信息
        check_shapes: 是否检查对应键的张量形状是否一致
        save_structure: 保存模型结构的目录路径,如果指定则保存两个模型的结构到该目录
    
    Returns:
        Tuple[bool, Dict[str, Any]]: 
            - bool: 键是否完全匹配
            - Dict: 包含匹配详情的字典
                - 'all_match': 所有键是否完全匹配
                - 'common_keys': 共同的键列表
                - 'only_in_weight1': 仅在权重1中的键
                - 'only_in_weight2': 仅在权重2中的键
                - 'shape_mismatches': 形状不匹配的键(如果check_shapes=True)
                - 'total_keys_weight1': 权重1的总键数
                - 'total_keys_weight2': 权重2的总键数
                - 'structure_files': 保存的结构文件路径列表(如果save_structure不为None)
    """
    # 加载两个权重文件
    if verbose:
        print(f"加载权重文件1: {weight_path1}")
    weights1 = load_pytorch_weights(weight_path1)
    
    if verbose:
        print(f"加载权重文件2: {weight_path2}")
    weights2 = load_pytorch_weights(weight_path2)
    
    # 保存模型结构到文件
    structure_files = []
    if save_structure:
        save_dir = Path(save_structure)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        weight1_name = Path(weight_path1).stem
        weight2_name = Path(weight_path2).stem
        
        structure_file1 = save_dir / f"{weight1_name}_structure.txt"
        structure_file2 = save_dir / f"{weight2_name}_structure.txt"
        
        if verbose:
            print(f"\n保存模型1结构到: {structure_file1}")
        save_model_structure(weights1, str(structure_file1), f"模型1结构 ({weight_path1})")
        structure_files.append(str(structure_file1))
        
        if verbose:
            print(f"保存模型2结构到: {structure_file2}")
        save_model_structure(weights2, str(structure_file2), f"模型2结构 ({weight_path2})")
        structure_files.append(str(structure_file2))
    
    # 获取键集合
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    
    # 计算共同键和差异键
    common_keys = sorted(list(keys1 & keys2))
    only_in_weight1 = sorted(list(keys1 - keys2))
    only_in_weight2 = sorted(list(keys2 - keys1))
    
    # 检查形状匹配
    shape_mismatches = []
    if check_shapes and common_keys:
        for key in common_keys:
            tensor1 = weights1[key]
            tensor2 = weights2[key]
            if hasattr(tensor1, 'shape') and hasattr(tensor2, 'shape'):
                if tensor1.shape != tensor2.shape:
                    shape_mismatches.append({
                        'key': key,
                        'shape1': tuple(tensor1.shape),
                        'shape2': tuple(tensor2.shape)
                    })
    
    # 判断是否完全匹配
    all_match = (len(only_in_weight1) == 0 and 
                 len(only_in_weight2) == 0 and 
                 len(shape_mismatches) == 0)
    
    # 构建结果字典
    result = {
        'all_match': all_match,
        'common_keys': common_keys,
        'only_in_weight1': only_in_weight1,
        'only_in_weight2': only_in_weight2,
        'shape_mismatches': shape_mismatches,
        'total_keys_weight1': len(keys1),
        'total_keys_weight2': len(keys2),
    }
    
    if structure_files:
        result['structure_files'] = structure_files
    
    # 打印详细信息
    if verbose:
        print("\n" + "="*80)
        print("权重键匹配检查结果")
        print("="*80)
        print(f"权重1总键数: {len(keys1)}")
        print(f"权重2总键数: {len(keys2)}")
        print(f"共同键数量: {len(common_keys)}")
        print(f"仅在权重1中的键数量: {len(only_in_weight1)}")
        print(f"仅在权重2中的键数量: {len(only_in_weight2)}")
        
        if only_in_weight1:
            print("\n仅在权重1中的键:")
            for key in only_in_weight1[:10]:  # 最多显示10个
                print(f"  - {key}")
            if len(only_in_weight1) > 10:
                print(f"  ... 还有 {len(only_in_weight1) - 10} 个键")
        
        if only_in_weight2:
            print("\n仅在权重2中的键:")
            for key in only_in_weight2[:10]:  # 最多显示10个
                print(f"  - {key}")
            if len(only_in_weight2) > 10:
                print(f"  ... 还有 {len(only_in_weight2) - 10} 个键")
        
        if check_shapes and shape_mismatches:
            print("\n形状不匹配的键:")
            for mismatch in shape_mismatches[:10]:  # 最多显示10个
                print(f"  - {mismatch['key']}: {mismatch['shape1']} vs {mismatch['shape2']}")
            if len(shape_mismatches) > 10:
                print(f"  ... 还有 {len(shape_mismatches) - 10} 个不匹配")
        
        print("\n" + "="*80)
        if all_match:
            print("✓ 所有键完全匹配!")
        else:
            print("✗ 键不完全匹配")
        print("="*80)
    
    return all_match, result


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查两个PyTorch权重文件的键是否匹配')
    parser.add_argument('weight1', type=str, help='第一个权重文件路径')
    parser.add_argument('weight2', type=str, help='第二个权重文件路径')
    parser.add_argument('--no-verbose', action='store_true', help='不打印详细信息')
    parser.add_argument('--no-check-shapes', action='store_true', help='不检查张量形状')
    parser.add_argument('--save-structure', type=str, default=None, 
                        help='保存模型结构的目录路径(例如: ./structures)')
    
    args = parser.parse_args()
    
    match, details = check_weights_keys_matching(
        args.weight1,
        args.weight2,
        verbose=not args.no_verbose,
        check_shapes=not args.no_check_shapes,
        save_structure=args.save_structure
    )
    
    return 0 if match else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())