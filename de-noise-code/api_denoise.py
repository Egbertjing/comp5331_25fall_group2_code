# -*- coding: utf-8 -*-

import json
import os
import time
import openai
from tqdm import tqdm
import multiprocessing
from functools import partial
from typing import List, Dict, Any, Optional
import logging
import difflib
# --- 1. 全局配置 ---

# --- 假设的文件名 (请根据您的实际情况修改) ---
DATASET_NAME='Arts_Crafts_and_Sewing'
# DATASET_NAME='Baby_Products'
# DATASET_NAME='Goodreads'
# DATASET_NAME='Movies_and_TV'    
# DATASET_NAME='Sports_and_Outdoors'
# DATASET_NAME='Video_Games'

# 我们使用 logging 模块来“弹出报错”，而不是直接 raise Exception
# 这样程序可以继续运行，同时在控制台清晰地显示所有匹配问题
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

# 数据文件
# 数据文件路径配置
if DATASET_NAME == 'Goodreads':
    TRAIN_FILE = f'data/{DATASET_NAME}/clean/train_data.txt'
    VAL_FILE = f'data/{DATASET_NAME}/clean/val_data.txt'
    TEST_FILE = f'data/{DATASET_NAME}/clean/test_data.txt' 
    ITEM_MAP_FILE = f'data/{DATASET_NAME}/clean/item_titles.json'
else:
    TRAIN_FILE = f'data/{DATASET_NAME}/5-core/downstream/train_data.txt'
    VAL_FILE = f'data/{DATASET_NAME}/5-core/downstream/val_data.txt'
    TEST_FILE = f'data/{DATASET_NAME}/5-core/downstream/test_data.txt'
    ITEM_MAP_FILE = f'data/{DATASET_NAME}/5-core/downstream/item_titles.json' 

LAMBDA_WEIGHT= 0  # 选择demo时用于长度惩罚的权重
# 任务1: 内容去噪
TEST_FILE_CONTENT = f'noise_test_data_txt/{DATASET_NAME}/test_data_noised_sequence_content.txt'
RETRIEVAL_MAP_CONTENT = f'demo_set/{DATASET_NAME}/sequence_content_retrieval_map_len_aware_lambda_{LAMBDA_WEIGHT}.json' # 为内容噪声生成的RAG文件
PROMPT_FILE_CONTENT = 'prompts/prompt_content_del_noise.txt'

# 任务2: 顺序去噪
TEST_FILE_ORDER = f'noise_test_data_txt/{DATASET_NAME}/test_data_noised_sequence_order.txt'
RETRIEVAL_MAP_ORDER = f'demo_set/{DATASET_NAME}/sequence_content_retrieval_map_len_aware_lambda_{LAMBDA_WEIGHT}.json' # 为顺序噪声生成的RAG文件
PROMPT_FILE_ORDER = 'prompts/prompt_order_noise.txt'

# --- 2. 辅助加载函数 ---

def load_item_map(filepath):
    """加载 ID -> Name 的JSON映射。"""
    print(f"Loading item map from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"关键错误: 找不到物品映射文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        item_map = json.load(f)
    
    # 键可能是整数，也可能是字符串。统一转换为字符串键以便查找。
    return {str(k): v for k, v in item_map.items()}

def load_data_file_as_list(filepath):
    """将 .txt 数据文件加载到内存中的列表，以便快速索引。"""
    print(f"Loading data file into memory: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"关键错误: 找不到数据文件: {filepath}")
        
    data_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # 将每行的ID存储为整数列表
                    data_list.append([int(i) for i in line.split()])
                except ValueError:
                    print(f"警告: 跳过非数字行: {line}")
    return data_list

def load_retrieval_map(filepath):
    """加载 RAG 检索到的 {test_idx: [train_idx...]} 映射。"""
    print(f"Loading retrieval map from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"关键错误: 找不到RAG映射文件: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_prompt_template(filepath):
    """从文件加载Prompt模板字符串"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"关键错误: 找不到Prompt模板文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# --- 3. 核心 "翻译" 与 "调用" 逻辑 ---

def translate_sequence(id_list, item_map, unknown_item_placeholder="[Unknown Item ID: {id}]"):
    """
    将一串ID列表转换为一串名称列表。
    """
    name_list = []
    for item_id in id_list:
        # 使用 .get() 安全地处理在映射中找不到的ID
        name = item_map.get(str(item_id), unknown_item_placeholder.format(id=item_id))
        name_list.append(name)
    return name_list


def generate_denoising_prompts(task_name, test_file_path, retrieval_map_path, prompt_template_path, item_map, train_data_list):
    """
    执行一个完整的去噪任务 (例如 "内容去噪" 或 "顺序去噪")。
    """
    print(f"\n--- 启动去噪任务: {task_name} ---")
    
    # 1. 加载此任务所需的文件
    try:
        test_data_list = load_data_file_as_list(test_file_path)
        retrieval_map = load_retrieval_map(retrieval_map_path)
        prompt_template = load_prompt_template(prompt_template_path)
    except FileNotFoundError as e:
        print(f"任务 {task_name} 失败: {e}")
        return

    final_prompt_list = []
    # 2. 遍历测试集中的每一条噪声数据
    # (为演示，我们只处理前 10 个)
    print(f"\n--- [Task: {task_name}] 正在处理 Test Sample ---")
    # for test_index in tqdm(range(min(10, len(test_data_list)))):
    for test_index in tqdm(range(len(test_data_list)), desc="生成去噪Prompt"): 
        # 3. 获取噪声数据 (ID格式)
        noisy_id_seq = test_data_list[test_index]
        if not noisy_id_seq:
            print(f"跳过空的测试行 {test_index}")
            continue
            
        # 4. "翻译" 噪声数据 (ID -> 名称)
        noisy_name_history = translate_sequence(noisy_id_seq[:-1], item_map)
        noisy_name_label = translate_sequence([noisy_id_seq[-1]], item_map)[0]
        
        target_sample_for_json = {
            "history": noisy_name_history
        }
        
        # 5. 获取RAG检索到的上下文 (ID格式)
        # 确保使用字符串键来查询JSON
        retrieved_train_indices = retrieval_map.get(str(test_index))
        if not retrieved_train_indices:
            print(f"警告: 在RAG映射中找不到 test_index {test_index} 的上下文。")
            continue
            
        # 6. "翻译" 上下文 (ID -> 名称)
        context_samples_for_json = []
        for train_index in retrieved_train_indices:
            clean_id_seq = train_data_list[train_index]
            clean_name_history = translate_sequence(clean_id_seq[:-1], item_map)
            clean_name_label = translate_sequence([clean_id_seq[-1]], item_map)[0]
            
            context_samples_for_json.append({
                "history": clean_name_history,
                "target_label": clean_name_label
            })
            
        # 7. 序列化为JSON字符串
        context_str = json.dumps(context_samples_for_json, indent=2)
        target_str = json.dumps(target_sample_for_json, indent=2)
        
        # 8. 组装最终Prompt
        # --- BUG FIX (2025-11-03) ---
        # 1. 预先转义模板中的所有 { 和 }，以防止 .format()
        #    错误地解析Prompt模板中包含的 *示例JSON*。
        escaped_template = prompt_template.replace("{", "{{").replace("}", "}}")
        
        # 2. "取消转义" (un-escape) 我们 *真正* 想要格式化的占位符
        escaped_template = escaped_template.replace("{{historical_context_json}}", "{historical_context_json}")
        escaped_template = escaped_template.replace("{{target_sample_json}}", "{target_sample_json}")
        # --- BUG FIX END ---
        try:
            final_prompt = escaped_template.format(
                historical_context_json=context_str,
                target_sample_json=target_str
            )
            final_prompt_list.append((test_index, final_prompt))
        except KeyError as e:
            print(f"错误: Prompt模板中的占位符 {e} 未找到。")
            continue
        

    return final_prompt_list
        

def prompt_gpt_single(client_params, template, item):
    """Process a single QA pair for judging"""
    client = openai.OpenAI(**client_params)
    max_retries = 5
    retry_count = 0
    # print(item[1])
    
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                {
                    "role": "user",
                    "content": template % item[1]
                }
                ],
                temperature=0,
                max_tokens=1024,
                top_p=0,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            content = response.choices[0].message.content

            return {
                "index": item[0],
                "content": content
            }
            
        except Exception as err:
            retry_count += 1
            print(f'Exception occurs when calling GPT-4 for judge (attempt {retry_count}/{max_retries}): {err}')
            if retry_count < max_retries:
                sleep_time = 2 * retry_count
                print(f'Will sleep for {sleep_time} seconds before retry...')
                time.sleep(sleep_time)
            else:
                print(f'Failed after {max_retries} attempts for {item[0]}')
                return {
                    "content": None,
                    "score": None, 
                    "reason": f"Failed after {max_retries} attempts: {str(err)}",
                    "Item": item[0]
                }

def prompt_gpt(client, items, num_processes=None):
    template = """
    %s
    """

    # Get client parameters for creating new clients in each process
    client_params = {
        "api_key": client.api_key, 
        "base_url": client.base_url
    }
    
    # Determine number of processes to use
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), len(items))
    
    # Create a partial function with fixed parameters
    process_fn = partial(prompt_gpt_single, client_params, template)
    
    # Process QA pairs in parallel using a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(
                process_fn, 
                items
            ),
            total=len(items),
            desc="Evaluating responses"
        ))
    
    # Sort results by index to maintain original order
    # results.sort(key=lambda x: x["idx"])
    
    # Print results
    # for i, result in enumerate(results):
    #     print(f'----------- question {i+1} ----------')
    #     print('User Instruction:', result['QApair'][0])
    #     print('Model Response:', result['QApair'][1])
    #     print('reason:', result['reason'])
    #     print('score:', result['score'])
    
    return results


def call_denoising_llm_api(task_name, prompt_list):
    """
    对一组Prompt调用LLM API以执行去噪分析。
    """
    print(f"\n--- [Task: {task_name}] 调用LLM API 处理 ---")

    api_key="sk-u29QLl0aWNCoUGnPH8dufHdd5tZ0gNl62uDGxz5OO1kjajaU"
    client = openai.OpenAI(api_key=api_key, base_url='https://xiaoai.plus/v1',)
    num_processes=64

    result = prompt_gpt(client, prompt_list, num_processes=num_processes)
    return result

def create_reverse_map(id_to_name_map):
    """
    创建 Name -> ID 的反向映射。
    """
    print("Creating reverse (Name -> ID) map...")
    name_to_id_map = {name: id for id, name in id_to_name_map.items()}
    
    # 科研严谨性检查：检查名称是否有歧义
    if len(name_to_id_map) != len(id_to_name_map):
        print(f"警告: 物品名称存在歧义 (重复的名称)。")
        print(f"  ID->Name 映射有 {len(id_to_name_map)} 个条目")
        print(f"  Name->ID 反向映射只有 {len(name_to_id_map)} 个条目")
        print("  反向翻译可能不完全准确。")
        
    return name_to_id_map


def parse_and_map_history(llm_output_str: str, 
                          name_to_id_map: Dict[str, int],
                          fuzzy_match_cutoff: float = 0.3) -> Optional[List[Optional[int]]]:
    """
    解析LLM生成的JSON字符串，提取history列表，并将其映射到数字ID。

    Args:
        llm_output_str: LLM 返回的原始字符串（包含JSON）。
        name_to_id_map: 标准名称到数字ID的映射字典。
        fuzzy_match_cutoff: 模糊匹配的相似度阈值 (0.0 到 1.0)。
                            阈值越高，匹配越严格。

    Returns:
        一个包含数字ID的列表。如果某个item无法匹配，则对应位置为 None。
        如果发生严重的解析错误，则返回 None。
    """
    
    # --- 步骤 1: 解析最外层的 JSON 字符串 ---
    try:
        data = json.loads(llm_output_str)
    except json.JSONDecodeError as e:
        logging.error(f"严重错误: 无法解析 LLM 的顶层 JSON 字符串。错误: {e}")
        logging.error(f"原始字符串 (前200字符): {llm_output_str[:200]}...")
        return []

    # --- 步骤 2: 提取 "corrected_final_result" ---
    # 我们需要处理两种情况：
    # 1. LLM 输出了一个嵌套的 JSON 对象 (如你的例子所示)
    # 2. LLM 严格遵守了 prompt 的示例，输出了一个字符串 (需要二次解析)
    
    result_data = data.get("corrected_final_result")
    if not result_data:
        logging.error("严重错误: 在 LLM 输出中未找到 'corrected_final_result' 键。")
        return []

    inner_data = {}
    if isinstance(result_data, str):
        # 情况2：值是一个字符串，需要再次解析
        try:
            inner_data = json.loads(result_data)
        except json.JSONDecodeError as e:
            logging.error(f"严重错误: 无法解析 'corrected_final_result' 中的内部 JSON 字符串。错误: {e}")
            return []
    elif isinstance(result_data, dict):
        # 情况1：值已经是一个字典 (如你的例子)
        inner_data = result_data
    else:
        logging.error(f"严重错误: 'corrected_final_result' 既不是字符串也不是字典，类型为 {type(result_data)}。")
        return []

    # --- 步骤 3: 提取 "history" 列表 ---
    history_item_names = inner_data.get("history")
    if not isinstance(history_item_names, list):
        logging.error(f"严重错误: 'history' 键不存在，或者其值不是一个列表。")
        return []

    logging.info(f"成功提取 {len(history_item_names)} 个 item 名称，准备开始映射...")

    # --- 步骤 4: 循环映射，处理严格匹配与模糊匹配 ---
    mapped_ids = []
    all_known_names = list(name_to_id_map.keys()) # 用于模糊匹配的“词典”

    for item_name in history_item_names:
        # 步骤 4.1: 尝试严格匹配 (最高效)
        item_id = name_to_id_map.get(item_name)

        if item_id is not None:
            # 严格匹配成功
            mapped_ids.append(item_id)
        else:
            # 步骤 4.2: 严格匹配失败，触发警告和模糊匹配
            logging.warning(f"匹配警告: 无法严格匹配 '{item_name}'")
            
            # 使用 difflib 寻找最接近的匹配
            # get_close_matches 返回一个列表，我们取第一个 (n=1)
            close_matches = difflib.get_close_matches(item_name, 
                                                      all_known_names, 
                                                      n=1, 
                                                      cutoff=fuzzy_match_cutoff)
            
            if close_matches:
                best_match_name = close_matches[0]
                matched_id = name_to_id_map[best_match_name]
                logging.warning(f"--> 已执行模糊匹配: 自动修正为 '{best_match_name}' (ID: {matched_id})")
                mapped_ids.append(matched_id)
            else:
                # 步骤 4.3: 模糊匹配也失败了
                logging.error(f"--> 匹配失败: 无法为 '{item_name}' 找到相似度 > {fuzzy_match_cutoff} 的匹配项。")

                # mapped_ids.append(None) # 在结果中插入 None 作为失败标记

    return mapped_ids

def save_results(result, final_prompt_list, name_to_id_map, test_file_path, task_name):
    # save
    print("\n--- 内容去噪任务执行完毕 ---")
    save_path = f'denoised_test_data/{DATASET_NAME}/{task_name}_noise_denoising_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"原始内容去噪结果已保存到: {save_path}")

    # 解析并翻译结果
    print("\n--- 解析并翻译内容去噪结果 ---")
    ind_2_ids_map = {}
    for res in tqdm(result):
        index = res['index']
        history_as_ids = parse_and_map_history(res['content'], name_to_id_map)
        if history_as_ids is not None:
            ind_2_ids_map[index] = history_as_ids

    #transfer ind_2_ids_map to list and add test labels
    test_data_list = load_data_file_as_list(test_file_path)
    # assert len(test_data_list) == len(final_prompt_list), "测试数据长度应等于Prompt列表长度"
    denoised_data_list = []
    for i in range(len(final_prompt_list)):
        if i in ind_2_ids_map:
            noisy_id_seq_label = test_data_list[i][-1]
            denoised_data_list.append(ind_2_ids_map[i]+[noisy_id_seq_label])
        else:
            denoised_data_list.append([]) # empty list for missing

    # save denoised data
    print("\n--- 保存内容去噪后的数据 ---")
    save_denoised_path = f'denoised_test_data/{DATASET_NAME}/test_data_denoised_sequence_{task_name}.txt'
    with open(save_denoised_path, 'w', encoding='utf-8') as f:
        for id_list in denoised_data_list:
            line = ' '.join(str(i) for i in id_list)
            f.write(line + '\n')
    print(f"内容去噪后的数据已保存到: {save_denoised_path}")


# --- 4. 主执行函数 ---
def main():
    """主执行逻辑"""
    print('DATASET_NAME = ', DATASET_NAME)
    try:
        # 1. 加载共享资源 (一次性加载)
        item_map = load_item_map(ITEM_MAP_FILE)
        name_to_id_map = create_reverse_map(item_map) # [新功能]
        train_data_list = load_data_file_as_list(TRAIN_FILE)
        print(f"\n--- 共享资源加载完毕 ---")
        print(f"物品映射 (Item Map) 已加载: {len(item_map)} 个条目")
        print(f"训练数据 (Train Data) 已加载: {len(train_data_list)} 个样本")

        # 2. 执行内容去噪任务
        final_prompt_list = generate_denoising_prompts(
            task_name="Content Noise Denoising",
            test_file_path=TEST_FILE_CONTENT,
            retrieval_map_path=RETRIEVAL_MAP_CONTENT,
            prompt_template_path=PROMPT_FILE_CONTENT,
            item_map=item_map,
            train_data_list=train_data_list
        )
        # 调用LLM API
        result = call_denoising_llm_api(
            task_name="Content Noise Denoising",
            prompt_list=final_prompt_list
        )
        save_results(result, final_prompt_list, name_to_id_map, TEST_FILE_CONTENT, task_name='content')

        
        # # 3. 执行顺序去噪任务
        # final_prompt_list = generate_denoising_prompts(
        #     task_name="Order Noise Denoising",
        #     test_file_path=TEST_FILE_ORDER,
        #     retrieval_map_path=RETRIEVAL_MAP_ORDER,
        #     prompt_template_path=PROMPT_FILE_ORDER,
        #     item_map=item_map,
        #     train_data_list=train_data_list
        # )
        # # 调用LLM API
        # result = call_denoising_llm_api(
        #     task_name="Order Noise Denoising",
        #     prompt_list=final_prompt_list
        # )
        # save_results(result, final_prompt_list, name_to_id_map, TEST_FILE_CONTENT,task_name='order')
        
        print("\n--- 所有去噪任务执行完毕 ---")

    except FileNotFoundError as e:
        print(f"\n!!! 关键错误，程序中止: {e}")
    except Exception as e:
        print(f"\n!!! 发生未知错误: {e}")

if __name__ == "__main__":
    main()

    # item_map = load_item_map(ITEM_MAP_FILE)
    # name_to_id_map = create_reverse_map(item_map) # [新功能]
    # llm_str = "{\n  \"Interest Profile (from Context)\": \"User is focused on model kits, painting supplies, and related accessories for scale modeling.\",\n  \"Noise Identification\": \"The item 'Handi Quilter Bobbin Box' is identified as noise. It is thematically inconsistent with the user's interest in model kits and painting supplies, as it pertains to quilting rather than scale modeling.\",\n  \"Suggested Correction\": \"Remove 'Handi Quilter Bobbin Box' from the history sequence.\",\n  \"Suggested Corrected final result\": '{\n  \"history\": [\n    \"Revell 1:48 P38J Lightning\",\n    \"Revell of Germany 03986 Spitfire MK.lla Model Kit\",\n    \"9pcs Round Pointed Tip Pony Hair Artists Filbert Paintbrushes, Marrywindix Watercolor Paint Brush Set Acrylic Oil Painting Brush Black\",\n    \"TAMIYA 1/48 Republic P-47D Thunderbolt - Razorback\",\n    \"Cordless Mini Mixer Model Painting Mixing Color Mixing Mini Portable Electric Mixer Stocking Stuffer Scale Model Paint Vallejo Tamiya\",\n    \"Revell 1:48 P - 40B Tiger Shark Plastic Model Kit, 12 years old and up, Camo\",\n    \"TAMIYA 35127 1/35 Israeli Merkava MBT Tank Plastic Model Kit\",\n    \"Revell 85-7546 P-61 Black Widow 1:48 Scale 130-Piece Skill Level 5 Model Airplane Building Kit\",\n    \"TAMIYA 300035068 35068 BR.Chieftain MK.5 Tank\"\n  ]\n}'\n}"

    # mapped_ids = parse_and_map_history(llm_str, name_to_id_map, fuzzy_match_cutoff=0.8)
    # print("Mapped IDs:", mapped_ids)