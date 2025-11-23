import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Heiti TC'  # 替换为你选择的字体

def load_data_file(filepath):
    """
    读取数据文件 (test.txt 或 train.txt)，解析每一行，
    返回一个 {行索引: 序列} 的字典。
    行索引从0开始。
    """
    data_sequences = {}
    try:
        with open(filepath, 'r') as f:
            # 假设行号（index）与JSON中的key是对应的
            for i, line in enumerate(f):
                parts = line.strip().split()
                if parts:
                    # 假设最后一个是label，前面的是序列
                    try:
                        # 将所有部分（除了最后一个）转换为整数
                        sequence = [int(x) for x in parts[:-1]]
                        data_sequences[i] = sequence
                    except ValueError as e:
                        print(f"Warning: Skipped line {i} in {filepath}. Error parsing numbers: {e}")
                else:
                     print(f"Warning: Skipped empty line {i} in {filepath}.")
    except FileNotFoundError:
        # 这个错误会在主函数中被捕获和处理
        raise
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return data_sequences

def calculate_overlap(seq1, seq2, method='jaccard'):
    """
    计算两个序列之间的重叠程度。
    默认方法: 'jaccard' (交集 / 并集)
    可选方法: 'intersection' (交集大小)
    """
    if not isinstance(seq1, list) or not isinstance(seq2, list):
        print(f"Warning: Invalid input for overlap calculation. Seq1: {type(seq1)}, Seq2: {type(seq2)}")
        return None

    set1 = set(seq1)
    set2 = set(seq2)
    
    intersection_count = len(set1.intersection(set2))
    
    if method == 'jaccard':
        union_count = len(set1.union(set2))
        if union_count == 0:
            # 两个都是空序列
            return 1.0 if intersection_count == 0 else 0.0
        return intersection_count / union_count
    elif method == 'intersection':
        return intersection_count
    else:
        # 默认回退到 Jaccard
        union_count = len(set1.union(set2))
        if union_count == 0:
            return 1.0
        return intersection_count / union_count

def analyze_overlaps(retrieval_map_file, test_file, train_file, output_file):
    """
    主分析函数
    """
    try:
        # 1. 加载 retrieval map
        print(f"Loading retrieval map from {retrieval_map_file}...")
        with open(retrieval_map_file, 'r') as f:
            retrieval_map = json.load(f)
        print(f"Successfully loaded retrieval map. Found {len(retrieval_map)} keys.")

        # 2. 加载数据文件
        print(f"Loading test data from {test_file}...")
        test_data = load_data_file(test_file)
        print(f"Loaded {len(test_data)} sequences from {test_file}.")
        
        print(f"Loading train data from {train_file}...")
        train_data = load_data_file(train_file)
        print(f"Loaded {len(train_data)} sequences from {train_file}.")

        # 3. 计算重叠度
        results = {}
        print("Calculating overlaps...")
        
        # 确保数据已加载
        if not test_data:
            print(f"Error: No data loaded from {test_file}. Aborting.")
            return
        if not train_data:
            print(f"Error: No data loaded from {train_file}. Aborting.")
            return

        total_keys = len(retrieval_map)
        processed_count = 0
        
        for test_idx_str, train_indices in retrieval_map.items():
            test_idx = int(test_idx_str) # 将JSON的字符串key转为整数
            
            if test_idx not in test_data:
                print(f"Warning: Test index {test_idx} not found in {test_file}. Skipping.")
                results[test_idx_str] = {"error": "Test index not found"}
                continue
                
            test_seq = test_data.get(test_idx)
            
            overlap_scores = []
            
            for train_idx in train_indices:
                if train_idx not in train_data:
                    # 仅在第一次遇到时警告，避免刷屏
                    if processed_count < 100: # 仅在前100个key中显示详细警告
                        print(f"Warning: Train index {train_idx} (for test key {test_idx_str}) not found in {train_file}.")
                    overlap_scores.append(None) # 用 None 标记缺失的数据
                    continue
                
                train_seq = train_data.get(train_idx)
                
                # 计算 Jaccard 相似度
                score = calculate_overlap(test_seq, train_seq, method='jaccard')
                if score is not None:
                    overlap_scores.append(score)
                else:
                    overlap_scores.append(None)
            
            # 计算平均重叠度（只计算有效分数）
            valid_scores = [s for s in overlap_scores if s is not None]
            if valid_scores:
                average_overlap = sum(valid_scores) / len(valid_scores)
            else:
                average_overlap = 0.0
                
            results[test_idx_str] = {
                "individual_overlaps_jaccard": overlap_scores,
                "average_overlap_jaccard": average_overlap,
                "retrieved_indices": train_indices
            }
            
            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total_keys:
                print(f"Processed {processed_count}/{total_keys} keys...")
            
        # 4. 保存结果
        print(f"Calculation complete. Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Done. Results saved to {output_file}.")

    except FileNotFoundError as e:
        print(f"************************************************************")
        print(f"错误：必要的数据文件未找到。")
        print(f"文件 '{e.filename}' 缺失。")
        print(f"************************************************************")
    except Exception as e:
        print(f"发生了一个意外错误: {e}")
        import traceback
        traceback.print_exc()








def analyze_jaccard_results(input_file, summary_file_path):
    """
    加载 overlap_results.json，执行详细的统计分析，
    并同时打印到控制台和保存到 summary_file_path 文件。
    """
    
    # 打开日志文件，使用 'with' 语句确保它在结束时被正确关闭
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f_summary:
            
            def log(message):
                """一个辅助函数，同时打印到控制台和写入文件"""
                print(message)
                f_summary.write(message + '\n')

            log(f"开始分析: {input_file}")
            log(f"统计总结将保存到: {summary_file_path}")
            
            log(f"\nLoading overlap results from {input_file}...")
            try:
                with open(input_file, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                log(f"************************************************************")
                log(f"错误: 文件 {input_file} 未找到。请确保文件已上传。")
                log(f"************************************************************")
                return
            except json.JSONDecodeError:
                log(f"************************************************************")
                log(f"错误: 文件 {input_file} 不是一个有效的JSON。")
                log(f"************************************************************")
                return
                
            log(f"加载成功。共找到 {len(data)} 个 test 索引。")

            average_overlaps = []
            max_individual_overlaps = []
            all_individual_overlaps = []
            
            # 存储 test_idx: avg_overlap 键值对，用于后续排序
            avg_overlap_by_idx = {}

            log("Processing data...")
            valid_entries = 0
            error_entries = 0

            for test_idx, results in data.items():
                if "error" in results:
                    error_entries += 1
                    continue
                
                valid_entries += 1
                avg_score = results.get("average_overlap_jaccard", 0.0)
                individual_scores = results.get("individual_overlaps_jaccard", [])
                
                # 过滤掉可能的 None 值 (例如，如果train索引无效)
                valid_individual_scores = [s for s in individual_scores if s is not None]

                average_overlaps.append(avg_score)
                avg_overlap_by_idx[test_idx] = avg_score
                
                if valid_individual_scores:
                    # 记录5个中最好的那一个
                    max_individual_overlaps.append(max(valid_individual_scores))
                    # 记录所有独立的对比分数
                    all_individual_overlaps.extend(valid_individual_scores)
                else:
                    # 如果没有有效分数（例如，所有retrieval都无效）
                    max_individual_overlaps.append(0.0)

            log(f"处理完成: {valid_entries} 个有效条目, {error_entries} 个错误条目被跳过。")

            # --- 1. 启动统计分析 (Numpy) ---
            log("\n" + "="*30)
            log("--- 统计分析结果 ---")
            log("="*30)
            
            if valid_entries == 0:
                log("未找到有效的分析数据。")
                return

            # 转换为Numpy数组以便于统计
            avg_scores_np = np.array(average_overlaps)
            max_scores_np = np.array(max_individual_overlaps)
            
            # --- 分析A: 平均重叠度 (系统整体性能) ---
            log("\n[分析A: 平均重叠度 (基于每个Test样本的5个检索的平均值)]")
            log("这是对系统“平均”性能的最佳评估。")
            log(f"  总均值 (Mean of Averages):   {np.mean(avg_scores_np):.4f}")
            log(f"  中位数 (Median):             {np.median(avg_scores_np):.4f}")
            log(f"  标准差 (Std Dev):            {np.std(avg_scores_np):.4f}")
            log(f"  最小值 (Min):                {np.min(avg_scores_np):.4f}")
            log(f"  最大值 (Max):                {np.max(avg_scores_np):.4f}")
            
            # 关键指标：完全失败的 Test 样本
            zero_avg_count = np.sum(avg_scores_np == 0.0)
            log(f"  *关键指标*:")
            log(f"    Test样本的平均重叠度为 0.0 的数量:  {zero_avg_count} (占总体的 {(zero_avg_count/valid_entries)*100:.2f}%)")
            
            # 关键指标：高重叠
            high_overlap_threshold = 0.5
            high_avg_count = np.sum(avg_scores_np > high_overlap_threshold)
            log(f"    Test样本的平均重叠度 > {high_overlap_threshold} 的数量: {high_avg_count} (占总体的 {(high_avg_count/valid_entries)*100:.2f}%)")

            # --- 分析B: 最大重叠度 (系统潜力) ---
            log("\n[分析B: 最大重叠度 (基于每个Test样本5个检索中的最佳值)]")
            log("这反映了系统“最好能做到什么程度”。")
            log(f"  总均值 (Mean of Maxes):      {np.mean(max_scores_np):.4f}")
            
            # 关键指标：硬失败（连最好的那个都是0）
            zero_max_count = np.sum(max_scores_np == 0.0)
            log(f"  *关键指标*:")
            log(f"    Test样本的5个检索重叠度*全部*为 0.0 的数量: {zero_max_count} (占总体的 {(zero_max_count/valid_entries)*100:.2f}%)")
            
            high_max_count = np.sum(max_scores_np > high_overlap_threshold)
            log(f"    Test样本*至少有1个*检索的重叠度 > {high_overlap_threshold} 的数量: {high_max_count} (占总体的 {(high_max_count/valid_entries)*100:.2f}%)")

            # --- 3. Top/Bottom 列表 ---
            log("\n" + "="*30)
            log("--- 最佳/最差 检索样本 ---")
            log("="*30)
            sorted_items = sorted(avg_overlap_by_idx.items(), key=lambda item: item[1], reverse=True)
            
            log("\nTop 5 Test 索引 (按平均重叠度):")
            for i, (idx, score) in enumerate(sorted_items[:5]):
                log(f"  {i+1}. Test 索引 {idx}: {score:.4f}")
                
            log("\nBottom 5 Test 索引 (按平均重叠度):")
            # 确保我们只展示列表的最后5个
            bottom_5 = sorted_items[-5:]
            bottom_5.reverse() # 逆序显示，从最低的开始
            for i, (idx, score) in enumerate(bottom_5):
                log(f"  {valid_entries-i}. Test 索引 {idx}: {score:.4f}")

            # --- 4. 可视化 (Matplotlib) ---
            log("\n" + "="*30)
            log("--- 生成可视化图表 ---")
            log("="*30)
            
            # 图 1: 平均重叠度的分布
            plt.figure(figsize=(9, 6))
            plt.hist(avg_scores_np, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
            plt.title('Average Jaccard similarity distribution ', fontsize=20)
            plt.xlabel('Average Jaccard similarity', fontsize=18)
            plt.ylabel('Number of Test Samples', fontsize=18)
            plt.xticks(fontsize=14)  # 设置 X 轴刻度数字的大小
            plt.yticks(fontsize=14)  # 设置 Y 轴刻度数字的大小
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xlim(-0.01, 1.01) #  slight padding
            plt.axvline(np.mean(avg_scores_np), color='red', linestyle='dashed', linewidth=2, label=f'mean: {np.mean(avg_scores_np):.3f}')
            plt.axvline(np.median(avg_scores_np), color='blue', linestyle='dashed', linewidth=2, label=f'middle: {np.median(avg_scores_np):.3f}')
            plt.legend()
            plt.savefig('average_overlap_distribution.png',dpi=600)
            log("已保存 'average_overlap_distribution.png'")

            # 图 2: 最大重叠度的分布
            # plt.figure(figsize=(12, 7))
            # plt.hist(max_scores_np, bins=50, range=(0, 1), edgecolor='black', alpha=0.7, color='green')
            # plt.title('图2: 最佳Jaccard重叠度分布 (top-K个检索中的最大值)', fontsize=16)
            # plt.xlabel('最大 Jaccard 相似度 (top-K个检索中的最佳值)', fontsize=12)
            # plt.ylabel('Test 样本数量', fontsize=12)
            # plt.grid(axis='y', linestyle='--', alpha=0.7)
            # plt.xlim(-0.01, 1.01)
            # plt.axvline(np.mean(max_scores_np), color='red', linestyle='dashed', linewidth=2, label=f'均值: {np.mean(max_scores_np):.3f}')
            # plt.axvline(np.median(max_scores_np), color='blue', linestyle='dashed', linewidth=2, label=f'中位数: {np.median(max_scores_np):.3f}')
            # plt.legend()
            # plt.savefig('max_overlap_distribution.png')
            # log("已保存 'max_overlap_distribution.png'")
                    
            log(f"\n分析完成。")

    except Exception as e:
        # 如果在打开 summary_file 之前就出错，则只打印到控制台
        print(f"发生了一个意外错误: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    # --- 执行主函数 ---
    # 提供的JSON文件名
    retrieval_map_filename = 'demo_set/Arts_Crafts_and_Sewing/sequence_content_retrieval_map_len_aware_lambda_0.json'
    test_filename = 'noise_test_data_txt/Arts_Crafts_and_Sewing/test_data_noised_sequence_content.txt'
    train_filename = 'data/Arts_Crafts_and_Sewing/5-core/downstream/train_data.txt'
    output_filename = 'overlap_results.json'

    # analyze_overlaps(retrieval_map_filename, test_filename, train_filename, output_filename)

    summary_txt = 'analysis_summary.txt' # 定义输出的txt文件名
    analyze_jaccard_results(output_filename, summary_txt)