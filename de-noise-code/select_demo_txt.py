# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
import os

# --- 1. 全局配置 ---
noise_type = 'sequence_content'
# noise_type = 'sequence_order'

# (请确保这些文件名与您的文件一致)
DATASET_NAME='Arts_Crafts_and_Sewing'
# DATASET_NAME='Baby_Products'
# DATASET_NAME='Goodreads'
# DATASET_NAME='Movies_and_TV'    
# DATASET_NAME='Sports_and_Outdoors'
# DATASET_NAME='Video_Games'

TOP_K = 5      # 您希望检索的K值

# (λ) Lambda: 用于平衡余弦相似度和长度奖励的权重
# 先不用
LAMBDA_WEIGHT = 0

# 数据文件路径配置
if DATASET_NAME == 'Goodreads':
    TRAIN_FILE = f'data/{DATASET_NAME}/clean/train_data.txt'
    VAL_FILE = f'data/{DATASET_NAME}/clean/val_data.txt'
    TEST_FILE = f'data/{DATASET_NAME}/clean/test_data.txt' 
else:
    TRAIN_FILE = f'data/{DATASET_NAME}/5-core/downstream/train_data.txt'
    VAL_FILE = f'data/{DATASET_NAME}/5-core/downstream/val_data.txt'
    TEST_FILE = f'data/{DATASET_NAME}/5-core/downstream/test_data.txt'

# noise文件路径
# TEST_FILE = f'noise_test_data_txt/{DATASET_NAME}/test_data_noised_{noise_type}.txt'
# OUTPUT_FILE = f'demo_set/{DATASET_NAME}/noise_{noise_type}_retrieval_map_len_aware_lambda_{LAMBDA_WEIGHT}.json'

#clean文件路径
# TEST_FILE 不变
OUTPUT_FILE = f'demo_set/{DATASET_NAME}/clean_retrieval_map_len_aware_lambda_{LAMBDA_WEIGHT}.json'

# --- 2. 辅助函数 ---

def find_max_item_id(files):
    """
    扫描所有数据文件，找到最大的item ID，以确定词汇表大小 (V)。
    """
    print(f"正在扫描文件 {files} 以确定物品词汇表大小...")
    max_id = -1
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path}，跳过。")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # 将所有ID转换为整数
                    ids = [int(i) for i in line.split()]
                    if ids:
                        max_in_line = max(ids)
                        if max_in_line > max_id:
                            max_id = max_in_line
                except ValueError as e:
                    print(f"警告: 在 {file_path} 中发现非数字行: {line} -> {e}")
                    
    return max_id

def load_and_vectorize(file_path, vocab_size):
    """
    加载 .txt 文件，并将其转换为一个 Scipy 稀疏矩阵 和 历史长度数组。
    每一行是一个多热编码向量。
    
    返回:
    csr_matrix: (N, V) 的稀疏矩阵
    np.array: (N,) 的历史长度数组
    """
    data = []           # 存储 '1'
    row_indices = []    # 存储行号 (i)
    col_indices = []    # 存储 item_id (j)
    history_lengths = [] # 存储每行的历史序列长度
    
    line_count = 0
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 找不到文件: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用tqdm显示进度条
        for i, line in enumerate(tqdm(f, desc=f"Vectorizing {file_path}")):
            line_count = i + 1
            line = line.strip()
            
            if not line:
                history_lengths.append(0) # 空行，长度为0
                continue
            
            try:
                parts = [int(i) for i in line.split()]
            except ValueError:
                history_lengths.append(0) # 包含非数字字符的行，长度为0
                continue
                
            if len(parts) < 2:
                # 至少需要1个历史item和1个label
                history_lengths.append(0) # 序列过短，有效历史为0
                continue
                
            # 历史 = 除最后一个item外的所有item
            history_ids = parts[:-1]
            
            # 使用 set() 确保每个item ID在向量中只出现一次 (多热编码)
            unique_history_ids = set(history_ids)
            
            # 记录该行的历史长度 (基于unique items，匹配多热向量)
            history_lengths.append(len(unique_history_ids))
            
            for item_id in unique_history_ids:
                if item_id >= vocab_size:
                    # 这不应该发生，但作为安全检查
                    print(f"警告: Item ID {item_id} 在行 {i} 超出词汇表范围。")
                    continue
                
                data.append(1)
                row_indices.append(i) # 当前行索引 (i)
                col_indices.append(item_id) # 物品ID作为列索引
    
    # 构建稀疏矩阵
    # 形状 = (总行数, 词汇表大小)
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(line_count, vocab_size))
    
    # 确保 history_lengths 的长度与矩阵的行数完全一致
    assert len(history_lengths) == line_count, "长度数组和矩阵行数不匹配!"
    
    return sparse_matrix, np.array(history_lengths)

def process_txt_data(file_path):
    """
    读取txt文件，每行去掉最后一个数字，存入一个大数组中。
    """
    master_array = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 1. .strip() 去除首尾换行符和空白
                # 2. .split() 按空白字符（空格、Tab等）分割成列表
                parts = line.strip().split()

                # 严谨性检查：确保该行不是空行
                if parts:
                    current_row = [int(x) for x in parts]
                    master_array.append(current_row)
                else:
                    print(f"提示：第 {line_num} 行是空行，已跳过。")

        return master_array

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}，请检查路径。")
        return []
    except Exception as e:
        print(f"读取过程中发生其他错误: {e}")
        return []
    
# --- 3. 主执行函数 ---

def main():
    """主执行逻辑"""
    print('noise_type is:',noise_type)
    # 1. 确定词汇表大小 (V)
    print("[1/5] 正在确定物品词汇表大小 (V)...")
    files_to_scan = [TRAIN_FILE,VAL_FILE, TEST_FILE]
    max_id = find_max_item_id(files_to_scan)
    
    if max_id == -1:
        print("错误: 在所有文件中都找不到任何物品ID。")
        return
        
    vocab_size = max_id + 1
    print(f"...完成。Max Item ID = {max_id}。向量维度 V = {vocab_size}。")

    # 2. 加载和向量化训练集
    print(f"[2/5] 正在加载和向量化训练集: {TRAIN_FILE}...")
    try:
        train_matrix, train_lengths = load_and_vectorize(TRAIN_FILE, vocab_size)
        print(f"...完成。训练矩阵 (train_matrix) 形状: {train_matrix.shape}")
        print(f"...完成。训练集长度 (train_lengths) 形状: {train_lengths.shape}")
    except FileNotFoundError as e:
        print(e)
        return

    # 3. 加载和向量化测试集
    print(f"[3/5] 正在加载和向量化测试集: {TEST_FILE}...")
    try:
        test_matrix, test_lengths = load_and_vectorize(TEST_FILE, vocab_size)
        print(f"...完成。测试矩阵 (test_matrix) 形状: {test_matrix.shape}")
        print(f"...完成。测试集长度 (test_lengths) 形状: {test_lengths.shape}")
    except FileNotFoundError as e:
        print(e)
        return
        
    # 4. 计算余弦相似度
    print("[4/5] 正在计算余弦相似度 (Test x Train)...")
    print("       (这可能是计算密集型步骤，请稍候...)")
    
    # 核心计算步骤：
    # similarity_matrix[i, j] = test[i] 和 train[j] 之间的相似度
    # 形状: (N_test, N_train)
    similarity_matrix = cosine_similarity(test_matrix, train_matrix)
    
    print(f"...完成。相似度矩阵形状: {similarity_matrix.shape}")

    # 5. 查找 Top-K 并保存结果
    print(f"[5/5] 正在为 {test_matrix.shape[0]} 个测试样本查找 Top-{TOP_K}...")
    
    retrieval_map = {} # 最终的 {test_index: [train_index_list]}
    # 应用拉普拉斯平滑 (+1)
    train_lengths_smooth = train_lengths.astype(float) + 1.0
    # --- MODIFICATION END ---


    #map to train data
    train_data_list = process_txt_data(TRAIN_FILE)
    
    for test_index in tqdm(range(similarity_matrix.shape[0]), desc="检索 Top-K"):
        
        sim_cosine_row = similarity_matrix[test_index] # (N_train,)
        
        # --- MODIFICATION START (v4) ---
        # 步骤 1: 获取平滑后的 test 长度 (标量)
        len_test_current_smooth = test_lengths[test_index] + 1.0
        
        # 步骤 2: 计算平滑后的长度比例 (向量, N_train,)
        # (L_train+1) / (L_test+1)
        len_ratio_vector = train_lengths_smooth / len_test_current_smooth
        
        # 步骤 3: 计算对数比例 (向量, N_train,)
        # log( (L_train+1) / (L_test+1) )
        # np.log 在 len_ratio_vector > 0 时工作正常
        log_ratio_vector = np.log(len_ratio_vector)
        
        # 步骤 4: 计算最终的长度缩放因子 (向量, N_train,)
        # 1 + λ * log_ratio
        scaling_factor_vector = 1.0 + LAMBDA_WEIGHT * log_ratio_vector
        
        # 步骤 5: 安全截断 (Clipping)
        # 防止极短序列导致 factor < 0，从而颠倒排序
        scaling_factor_vector = np.maximum(0.0, scaling_factor_vector)
        
        # 步骤 6: 计算最终得分
        # S_final = S_cosine * Factor
        final_score_row = sim_cosine_row * scaling_factor_vector
        # --- MODIFICATION END ---
        
        # 按最终的 S_final 排序
        top_k_train_indices = np.argsort(final_score_row)[-TOP_K:][::-1]

        # 存入字典
        top_k_train_indices_list=top_k_train_indices.tolist()
        top_k_train_data_list = []
        for i in top_k_train_indices_list:
            top_k_train_data_list.append(train_data_list[i])


        retrieval_map[test_index] = top_k_train_data_list


    # 6. 保存到JSON
    print(f"...完成。正在将检索映射保存到: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(retrieval_map, f, indent=4)
        
        print("--- 检索任务全部完成 ---")
        
    except Exception as e:
        print(f"保存JSON文件失败: {e}")


if __name__ == "__main__":
    main()
