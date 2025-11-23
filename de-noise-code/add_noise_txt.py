# -*- coding: utf-8 -*-


import random
import os
from tqdm import tqdm

# --- 全局参数配置 ---
# noise_type = 'sequence_content'
# noise_type = 'sequence_order'

# (请确保这些文件名与您的文件一致)
# DATASET_NAME='Arts_Crafts_and_Sewing'
DATASET_NAME='Baby_Products'
DATASET_NAME='Goodreads'
DATASET_NAME='Movies_and_TV'    
DATASET_NAME='Sports_and_Outdoors'
DATASET_NAME='Video_Games'

# 数据文件路径配置
if DATASET_NAME == 'Goodreads':
    TRAIN_FILE = f'data/{DATASET_NAME}/clean/train_data.txt'
    VAL_FILE = f'data/{DATASET_NAME}/clean/val_data.txt'
    TEST_FILE = f'data/{DATASET_NAME}/clean/test_data.txt' 
else:
    TRAIN_FILE = f'data/{DATASET_NAME}/5-core/downstream/train_data.txt'
    VAL_FILE = f'data/{DATASET_NAME}/5-core/downstream/val_data.txt'
    TEST_FILE = f'data/{DATASET_NAME}/5-core/downstream/test_data.txt'

# 目标文件：我们实际要注入噪声的文件
TARGET_FILE = TEST_FILE

# 物品池来源：用于构建“物品宇宙”的所有文件
ITEM_POOL_SOURCE_FILES = [TRAIN_FILE, VAL_FILE, TEST_FILE]

# 噪声比例：将有100%的数据行被注入噪声
NOISE_RATIO = 1

# --- 1. 数据I/O与物品池构建 ---

def load_data_from_txt(filepath):
    """
    从TXT文件中加载数据。
    每行: "id1 id2 ... idN label"
    返回: 列表，每个元素是 {'sequence': [id1, ... idN], 'label': label}
    """
    print(f"正在加载目标数据: {filepath} ...")
    data = []
    if not os.path.exists(filepath):
        print(f"错误: 目标文件 {filepath} 未找到。")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()

            # attention!!! 
            if len(parts) < 3:  # 至少需要2个序列项和1个标签
                continue
            
            try:
                # 最后一个是label，前面所有都是sequence
                label = int(parts[-1])
                sequence = [int(item) for item in parts[:-1]]
                data.append({'sequence': sequence, 'label': label})
            except ValueError:
                print(f"警告: 无法解析行: {line}")
                continue
    print(f"加载完成，共 {len(data)} 条数据。")
    return data

def save_data_to_txt(filepath, data):
    """
    将加噪后的数据保存回TXT格式。
    """
    print(f"正在保存加噪数据到: {filepath} ...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            # 将序列中的所有int转换为str
            seq_str = " ".join(map(str, item['sequence']))
            # 写入 "seq label\n"
            f.write(f"{seq_str} {item['label']}\n")

def build_item_pool(filepaths):
    """
    从所有数据文件中构建一个完整的物品ID池。
    """
    print(f"正在从 {len(filepaths)} 个文件中构建物品池...")
    item_pool_set = set()
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"警告: 物品池源文件 {filepath} 未找到，已跳过。")
            continue
        
        print(f"  ... 正在扫描 {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                try:
                    # 将该行的 *所有* ID 都添加到池中
                    item_ids = [int(item) for item in parts]
                    item_pool_set.update(item_ids)
                except ValueError:
                    continue # 忽略格式错误的行
                    
    all_item_ids = list(item_pool_set)
    if not all_item_ids:
        print("错误: 物品池为空！请检查数据文件。")
        return None
        
    print(f"物品池构建完毕，共 {len(all_item_ids)} 个独立物品ID。")
    return all_item_ids

# --- 2. 噪声注入函数 ---

def add_sequence_content_noise(data, noise_ratio, all_item_ids):
    """
    噪声类型1: 序列内容噪声 (插入, 删除, 替换)
    [注意: 此函数 *就地修改* data 列表]
    """
    print(f"开始注入 序列内容噪声 (Sequence Content Noise)...")
    num_to_noise = int(len(data) * noise_ratio)
    indices_to_noise = random.sample(range(len(data)), num_to_noise)
    log = [f"噪声类型: Sequence Content Noise (Ratio: {noise_ratio})"]
    
    for idx in tqdm(indices_to_noise, desc="注入内容噪声"):
        # 随机选择一种操作
        # sub_type = random.choice(['insert', 'delete', 'replace'])
        sub_type = 'delete'  # 仅测试删除操作
        # 获取序列的 *引用*
        sequence = data[idx]['sequence']
        seq_len = len(sequence)
        
        if sub_type == 'insert':
            # 随机选择一个新物品
            new_item_id = random.choice(all_item_ids)
            # 随机选择插入位置
            insert_pos = random.randint(0, seq_len)
            
            # 就地插入
            sequence.insert(insert_pos, new_item_id)
            log.append(f"Row {idx} [INSERT]: 插入物品 {new_item_id} at pos {insert_pos}")

        elif sub_type == 'delete':
            # 随机选择删除位置
            delete_pos = random.randint(0, seq_len - 1)
            deleted_id = sequence[delete_pos]
            
            # 就地删除
            sequence.pop(delete_pos)
            log.append(f"Row {idx} [DELETE]: 删除物品 {deleted_id} from pos {delete_pos}")
            
        elif sub_type == 'replace' and seq_len > 0:
            # 随机选择替换位置
            replace_pos = random.randint(0, seq_len - 1)
            original_id = sequence[replace_pos]
            
            # 随机选择一个新物品 (确保和原物品不同)
            new_item_id = random.choice(all_item_ids)
            while new_item_id == original_id and len(all_item_ids) > 1:
                new_item_id = random.choice(all_item_ids)
                
            # 就地替换
            sequence[replace_pos] = new_item_id
            log.append(f"Row {idx} [REPLACE]: 替换物品 {original_id} -> {new_item_id} at pos {replace_pos}")
            
    return log

def add_order_noise(data, noise_ratio):
    """
    噪声类型2: 序列顺序噪声 (随机交换)
    [注意: 此函数 *就地修改* data 列表]
    """
    print(f"开始注入 序列顺序噪声 (Sequence Order Noise)...")
    num_to_noise = int(len(data) * noise_ratio)
    indices_to_noise = random.sample(range(len(data)), num_to_noise)
    log = [f"噪声类型: Sequence Order Noise (Ratio: {noise_ratio})"]
    
    for idx in tqdm(indices_to_noise, desc="注入顺序噪声"):
        # 获取序列的 *引用*
        sequence = data[idx]['sequence']
        seq_len = len(sequence)
        
        if seq_len < 2:
            # 序列长度不足2，无法交换
            continue
            
        # 随机选择两个 *不同* 的位置
        pos1, pos2 = random.sample(range(seq_len), 2)
        
        # 就地交换 (Python的原子操作，非常安全)
        sequence[pos1], sequence[pos2] = sequence[pos2], sequence[pos1]
        
        log.append(f"Row {idx}: 交换位置 {pos1} <-> {pos2}")
        
    return log

# --- 3. 主执行函数 ---

def main():
    """
    主执行逻辑
    """
    # 1. 构建 *完整* 物品池 (使用所有源)
    all_item_ids = build_item_pool(ITEM_POOL_SOURCE_FILES)
    if not all_item_ids:
        print("执行中止。")
        return
        
   

    # 3. 选择一种噪声类型
    # for noise_type in ['sequence_content', 'sequence_order']:
    for noise_type in ['sequence_content']:
        # 2. 加载和预处理 *目标* 数据 (我们要加噪的文件)
        data = load_data_from_txt(TARGET_FILE)
        if not data:
            print("执行中止。")
            return
        
        print(f"\n--- 已选择噪声类型: {noise_type} ---")
        
        log = []

        # 4. 执行选中的噪声注入 (函数会就地修改 'data')
        if noise_type == 'sequence_content':
            log = add_sequence_content_noise(data, NOISE_RATIO, all_item_ids)
        elif noise_type == 'sequence_order':
            log = add_order_noise(data, NOISE_RATIO)
            
        # 5. 保存结果
        if log: # 检查log是否为空，以确认噪声已注入
            # 定义输出文件名
            base_name = os.path.basename(TARGET_FILE).rsplit('.', 1)[0] # rsplit确保只分割最后一个.
            output_txt_file = f"noise_test_data_txt/{DATASET_NAME}/{base_name}_noised_{noise_type}.txt"
            output_log_file = f"noise_test_data_txt/{DATASET_NAME}/{base_name}_noised_{noise_type}_log.txt"
            
            # 保存加噪后的TXT
            save_data_to_txt(output_txt_file, data)
            
            # 保存日志文件
            print(f"正在保存日志到: {output_log_file}")
            try:
                with open(output_log_file, 'w', encoding='utf-8') as f:
                    f.write("\n".join(log))
                print("--- 噪声注入完成 ---")
            except IOError as e:
                print(f"保存日志失败: {e}")
        else:
            print("错误: 噪声注入失败，未生成数据。")


if __name__ == "__main__":
    print('DATASET_NAME:', DATASET_NAME )
    main()
