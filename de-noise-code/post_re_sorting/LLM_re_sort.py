# -*- coding: utf-8 -*-

import json
import time
import openai
from tqdm import tqdm
import multiprocessing
from functools import partial
import difflib
import re
import ast

# =========================================
# 1. 配置与模板定义
# =========================================

# 建议将此模板单独保存在一个文件中或配置类中，方便迭代修改
RAG_RERANK_PROMPT_TEMPLATE_DEMO = """
# Role
You are an expert Recommender System capable of understanding complex user preferences based on their historical interactions.

# Task
Your task is to re-rank a list of candidate items for a target user. You will be provided with the user's historical interaction sequence and a list of 50 candidate items.
To assist you, I will provide examples of similar users, showing their histories and the items they actually chose next.

# Knowledge Base (Similar User Examples)
Learning from how similar users made choices can help you understand the current user's potential interests.
{demo_block}

# Current Target User
- Interaction History (Ordered by time, latest at the end):
{current_user_history}

# Candidate Items (Top-50)
Please re-rank the following candidate items based on how likely the target user will interact with them next.
{candidate_items_list}

# Output Requirements
1. **RANKING IS CRITICAL**: The output list MUST be ordered by the likelihood of interaction. The FIRST item in the list should be the MOST likely next item, the SECOND item the second most likely, and so on (descending order of probability).
2. STRICTLY select the top 20 most likely items from the provided 'Candidate Items' list only.
3. Output matched items exactly as they appear in the candidate list. Do not alter names.
4. Present the final result purely as a list of strings, NO other text.
Example: ["Most Likely Item", "Second Most Likely Item", ..., "10th Likely Item"]

# Final Answer
"""

# ZERO_SHOT_RERANK_PROMPT_TEMPLATE = """
# # Role
# You are an expert Recommender System capable of understanding complex user preferences based on their historical interactions.

# # Task
# Your task is to re-rank a list of candidate items for a target user. You will be provided with the user's historical interaction sequence and a list of 50 candidate items.

# # Current Target User
# - Interaction History (Ordered by time, latest at the end):
# {current_user_history}

# # Candidate Items
# Please re-rank the following candidate items based on how likely the target user will interact with them next.
# {candidate_items_list}

# # Output Requirements
# 1. **RANKING IS CRITICAL**: The output list MUST be ordered by the likelihood of interaction. The FIRST item in the list should be the MOST likely next item, the SECOND item the second most likely, and so on (descending order of probability).
# 2. STRICTLY select the top 20 most likely items from the provided 'Candidate Items' list only.
# 3. Output matched items exactly as they appear in the candidate list. Do not alter names.
# 4. Present the final result purely as a list of strings, NO other text.
# Example: ["Most Likely Item", "Second Most Likely Item", ..., "10th Likely Item"]

# # Final Answer
# """

ZERO_SHOT_RERANK_PROMPT_TEMPLATE ="""
# Role
You are a world-class Recommender System Engine. Your goal is to predict the user's next likely interaction by analyzing their behavior sequence.

# Task Objective
Re-rank the provided list of [Candidate Items] based on the user's [Target User History]. You must identify the user's core interests and immediate needs to determine the most probable next items.

# Ranking Guidelines (Critical for Success)
When determining the ranking order, apply the following criteria:
1. **Sequential Coherence**: Give higher priority to items that logically follow the user's MOST RECENT interactions (e.g., complementary accessories, functional next steps). Recent history is more indicative of immediate intent.
2. **Interest Consistency**: Items that semantically align with the user's dominant long-term interests should be ranked higher.
3. **Specificity Matching**: Prefer items that match the specific granular attributes (e.g., specific brand, material, or sub-category) seen in the history, rather than generic related items.

# Input Data
## Target User History (Time-ordered sequence, latest at the bottom)
{current_user_history}

## Candidate Items (to be re-ranked)
{candidate_items_list}

# Output Constraints (MUST FOLLOW)
1. **Logical Ranking**: The output MUST be a strictly ordered list where index 0 is the MOST likely next item, index 1 is the second most likely, etc.
2. **Top-20 Cutoff**: You must output exactly the top 20 items selected from the Candidate Items.
3. **No Hallucination**: ONLY select items from the provided [Candidate Items] list. If an item is not in the candidate list, it MUST NOT appear in your output.
4. **Exact Match**: Output the item names exactly as they appear in the candidate list (including punctuation and case).
5. **Pure JSON Format**: The final output must be a standard JSON list of strings only. Do not include any explaining text, markdown formatting, or notes.

# Final Output
"""

# =========================================
# 2. 数据加载辅助函数
# =========================================
def load_json(filepath):
    print(f"正在加载: {filepath} ...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        return {} if 'map' in filepath or 'titles' in filepath else []

# =========================================
# 3. 核心处理逻辑
# =========================================
def generate_prompts(demo_map_path, item_titles_path, predict_data_path,using_demo,candaidate_num):
    # --- A. 加载数据 ---
    # 请确保这些文件在您的当前工作目录下
    demo_map = load_json(demo_map_path)
    item_titles = load_json(item_titles_path)
    predict_data = load_json(predict_data_path)

    if not predict_data:
        print("没有预测数据，终止处理。")
        return []

    # --- B. 辅助 Helper ---
    # 这是一个闭包函数，用于快速查找物品名称，缺失时返回原始ID字符串
    def get_name(item_id):
        return item_titles.get(str(item_id), str(item_id))

    formatted_prompts = []
    total_samples = len(predict_data)
    print(f"开始构建 Prompt，共 {total_samples} 条数据...")

    # --- C. 主循环 ---
    # 使用 enumerate 获取 index，因为 demo_map 的 key 是数据的索引
    for idx, test_item in enumerate(predict_data):
        # 1. 构建当前用户历史 (Current History)
        # predict_infos 中的 history_sequences 已经是列表
        cur_history_str = " -> ".join([get_name(iid) for iid in test_item.get('history_sequences', [])])

        # 2. 构建候选列表 (Candidates)
        # 建议：在实际实验中，这里可以加入 random.shuffle(candidate_names) 来消除位置偏差
        candidate_ids = test_item.get('candidate_preds', [])
        candidate_names = [get_name(iid) for iid in candidate_ids]
        candidate_names = candidate_names[:candaidate_num]  
        # 使用 json.dumps 格式化为标准的 JSON 字符串列表，清晰且易于 LLM 理解
        candidates_str = json.dumps(candidate_names, ensure_ascii=False, indent=2)

        if using_demo:
            # 3. 构建 Demo 模块 (RAG Block)
            idx_str = str(idx) # demo_map 的 key 是字符串类型的索引
            demos = demo_map.get(idx_str, [])
            
            demo_texts = []
            # 取前5个demo (如果存在)
            for i, demo_seq in enumerate(demos[:5]):
                # demo_seq 结构: [hist1, hist2, ..., label]
                if len(demo_seq) < 2: continue # 防御性编程：确保至少有一条历史和一个标签
                
                demo_history_ids = demo_seq[:-1]
                demo_label_id = demo_seq[-1]
                
                demo_hist_str = " -> ".join([get_name(iid) for iid in demo_history_ids])
                demo_label_str = get_name(demo_label_id)
                
                demo_texts.append(
                    f"[Example {i+1}]\nHistory: {demo_hist_str}\nActual Choice: {demo_label_str}"
                )
            
            demo_block_str = "\n\n".join(demo_texts) if demo_texts else "No similar user examples available."

            # 4. 填充完整 Prompt
            full_prompt = RAG_RERANK_PROMPT_TEMPLATE_DEMO.format(
                demo_block=demo_block_str,
                current_user_history=cur_history_str,
                candidate_items_list=candidates_str
            )
        else:
            full_prompt = ZERO_SHOT_RERANK_PROMPT_TEMPLATE.format(
                current_user_history=cur_history_str,
                candidate_items_list=candidates_str
            )


        formatted_prompts.append(full_prompt)

        if (idx + 1) % 3000 == 0:
            print(f"已处理 {idx + 1}/{total_samples} 条...")

    print("Prompt 构建完成！")
    return formatted_prompts



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
                # model="gpt-4.1-mini",
                # model="gpt-5-nano",

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

# =========================================
# 1. 核心解析类 (Utility Class for Robust Mapping)
# =========================================
class LLMOutputResolver:
    def __init__(self, item_titles_path):
        self.id_to_name = self._load_json(item_titles_path)
        print(len(self.id_to_name), "item titles loaded.")
        # 确保 ID 都是字符串类型，方便统一处理
        self.id_to_name = {str(k): str(v) for k, v in self.id_to_name.items()}
        # self.name_to_id = {v: k for k, v in self.id_to_name.items()}
        # print(len(self.name_to_id)," len(self.name_to_id)")
        # self.valid_candidates = [k for k in self.name_to_id.keys()]

    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
            return {}

    def resolve_output(self, idx, llm_ranked_names, original_candidate_ids):
        """
        将 LLM 输出的 Name 列表映射回 ID 列表。
        仅在 original_candidate_ids 范围内进行匹配，确保高效且准确。
        """
        # 1. 构建当前样本的局部反向映射表 (Local Reverse Map)
        # 这比全局搜索快得多，也更安全
        local_name_to_id = {}
        valid_candidates = []
        
        for cid in original_candidate_ids:
            cid_str = str(cid)
            if cid_str in self.id_to_name:
                name = self.id_to_name[cid_str]
                local_name_to_id[name] = cid_str
                valid_candidates.append(name)
            else:
                # 极端情况：候选ID在title表中不存在，科研中需注意这种脏数据
                pass 
        # local_name_to_id = self.name_to_id
        # valid_candidates = self.valid_candidates

        resolved_ids = []
        for name_from_llm in llm_ranked_names:
            # --- A. 精确匹配 (Exact Match) ---

            if name_from_llm in local_name_to_id:
                resolved_ids.append(local_name_to_id[name_from_llm])
                continue

            # --- B. 模糊匹配 (Fuzzy Match via difflib) ---
            # 在 50 个候选项中寻找最接近的
            # cutoff=0.6 是一个经验值，可以根据实际情况调整 (0.0-1.0，越高越严格)
            # print(f"Attempting fuzzy match for: '{name_from_llm}'")
            matches = difflib.get_close_matches(name_from_llm, valid_candidates, n=1, cutoff=0.3)
            
            if matches:
                best_match = matches[0]
                resolved_ids.append(local_name_to_id[best_match])
                # 可选：打印日志看看模糊匹配的情况，用于调整 cutoff
                # print(f"[Fuzzy Match] '{name_from_llm}' -> '{best_match}'")
            else:
                print(f"Warning: In index {idx}, Could not map LLM output '{name_from_llm}' back to any candidate.")
                # 此时可以选择跳过，或者用某种默认值填充

        return resolved_ids
    
def parse_llm_json_output(llm_response_str):
    """
    科研级鲁棒解析器 v3.0:
    1. 标准 JSON 解析 (json.loads)
    2. 提取最外层 [...] 后标准 JSON 解析
    3. Python AST 解析 (ast.literal_eval, 处理单引号)
    4. [NEW] 终极回退：基于行的正则提取 (处理未转义引号等严重语法错误)
    """
    cleaned_str = llm_response_str.strip()
    # --- 预处理 markdown ---
    if cleaned_str.startswith("```"):
        cleaned_str = re.sub(r'^```[a-zA-Z]*\n', '', cleaned_str)
        if cleaned_str.endswith("```"):
            cleaned_str = cleaned_str[:-3]
    cleaned_str = cleaned_str.strip()

    # === Level 1: 标准尝试 ===
    try:
        return json.loads(cleaned_str)
    except Exception:
        pass

    # === Level 2: 提取列表主体 ===
    match = re.search(r'\[.*\]', cleaned_str, re.DOTALL)
    if match:
        list_body = match.group(0)
        # Sub-attempt 2.1: JSON again
        try:
            return json.loads(list_body)
        except Exception:
            pass
        # Sub-attempt 2.2: AST (处理单引号)
        try:
            return ast.literal_eval(list_body)
        except Exception:
            pass

    # === Level 3: 终极回退 (Regex Line Scraper) ===
    # 针对您提供的这种换行分隔的列表，我们直接提取每一行双引号中间的内容。
    # 这个正则的意思是：寻找被双引号包裹的内容，忽略行首行尾的空白和可能的逗号。
    # 它可以容忍中间出现奇怪的未转义符号，因为它强行匹配到行尾最后一个双引号。
    print("[Warning] Standard parsing failed. Attempting regex fallback...")
    
    # 策略：假设每一项都在单独的一行上，且被 "" 包裹
    # 我们寻找形如 `  "任意内容",` 或 `  "任意内容"` 的行
    fallback_items = []
    # 用换行符切分，逐行处理
    lines = cleaned_str.split('\n')
    for line in lines:
        line = line.strip()
        # 跳过纯开头 [ 和纯结尾 ]
        if line in ['[', ']']: 
            continue
        
        # 尝试匹配被双引号包裹的内容。
        # 这是一个非贪婪匹配，但为了应对未转义的结尾引号，我们可能需要更复杂的逻辑。
        # 针对您的数据，简单点：掐头去尾。
        
        # 如果行以 " 开头
        if line.startswith('"'):
            # 去掉开头的 "
            content = line[1:]
            # 去掉结尾可能的 ,
            if content.endswith(','):
                content = content[:-1]
            # 去掉结尾的 "
            if content.endswith('"'):
                content = content[:-1]
            
            # 此时 content 就是我们要的，虽然可能还带有一些转义符（如 \"），
            # 我们可以手动处理一下常见的转义以恢复原始文本
            content = content.replace('\\"', '"')
            
            fallback_items.append(content)
            
    if fallback_items:
        return fallback_items

    return []       


def call_llm_re_ranking_and_resolve(demo_map_path, item_titles_path, predict_data_path, output_dir, candaidate_num):
    # 执行构建
    all_prompts = generate_prompts(demo_map_path, item_titles_path, predict_data_path,using_demo=False,candaidate_num=candaidate_num)

    # 打印第一条生成的 Prompt 进行检查
    # if all_prompts:
    #     print("\n--- Sample Prompt Preview [Index 0] ---")
    #     print(all_prompts[0])
    #     print("---------------------------------------")
    #     print(f"总共生成了 {len(all_prompts)} 个 Prompt。")
    #     return

    # all_prompts=all_prompts[:2]
    # 调用 LLM API 进行处理
    llm_re_ranking_results = call_denoising_llm_api("RAG based LLM Re-ranking", [(idx, prompt) for idx, prompt in enumerate(all_prompts)])
    # print(llm_re_ranking_results[0])
    output_path = f'{output_dir}/raw_llm_re_ranked_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(llm_re_ranking_results, f, ensure_ascii=False, indent=2)
    print(f"\nLLM输出的重排序结果已保存至 {output_path}")


    # llm_re_ranking_results = json.load(open(output_path, 'r'))
    

    # 解析 LLM 输出
    resolver = LLMOutputResolver(item_titles_path)
    final_mapped_results = {}

    predict_data = load_json(predict_data_path)
    re_ranked_results = []
    for res in tqdm(llm_re_ranking_results):
        idx = res['index']
        # if (idx + 1) % 1000 == 0:
        #     print(f"已处理 {idx + 1}/{len(llm_re_ranking_results)} 条...")
        # print(f"Processing index: {idx}")
        if res['content'] is None:
            print(f"Skipping index {idx} due to None content.")
            continue
        LLM_response = parse_llm_json_output(res['content'])
        if LLM_response is None:
            print(f"Skipping index {idx} due to failed LLM response.")
            continue
        # print(len(LLM_response))
        # print(predict_data[idx])
        original_candidate_ids = predict_data[idx]['candidate_preds']
        # print("--- 开始解析 LLM 输出 ---")
        resolved_ids = resolver.resolve_output(idx, LLM_response,original_candidate_ids)
    
        # print("\nOriginal Candidate IDs:", original_candidate_ids)
        # print("Resolved IDs:", resolved_ids)
        cur_re_ranked_result = {
            "history_sequences": predict_data[idx]['history_sequences'],
            "label": predict_data[idx]['label'],
            "candidate_preds": original_candidate_ids,
            "re_ranked_candidates": resolved_ids
        }
        re_ranked_results.append(cur_re_ranked_result)
    # 保存最终结果
    output_path = f'{output_dir}/llm_re_ranked_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(re_ranked_results, f, ensure_ascii=False, indent=2)
    print(f"\n最终重排序结果已保存至 {output_path}")
    


if __name__ == '__main__':
    # dataset = 'Arts_Crafts_and_Sewing'
    # dataset = 'Baby_Products'
    # dataset = 'Goodreads'
    # dataset = 'Movies_and_TV'
    # dataset = 'Sports_and_Outdoors'
    # dataset = 'Video_Games'
    dataset_list = [
        'Baby_Products',
        'Goodreads'
                    ]
    
    
    # model_name = 'Qwen2-0.5'
    model_name = 'Smollm2-135'

    with_noise =  False

    # candaidate_num=40
    candaidate_num=20

    # 定义文件路径
    # demo_map_path = '/Users/yangzaifei/Desktop/courses/COMP 5331/project/demo_set/Arts_Crafts_and_Sewing/clean_retrieval_map_len_aware_lambda_0.json'
    demo_map_path=''

    for dataset in dataset_list:
        if dataset=='Goodreads':
            item_titles_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/data/{dataset}/clean/item_titles.json'
        else:
            item_titles_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/data/{dataset}/5-core/downstream/item_titles.json'

        if with_noise:
            final_name = 'noise_predict_infos.json'
        else:
            final_name = 'predict_infos.json'


        if dataset=='Arts_Crafts_and_Sewing':
            predict_data_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/post_re_sorting/{model_name}/Arts_5core/{final_name}'
        elif dataset=='Baby_Products':
            predict_data_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/post_re_sorting/{model_name}/Baby_5core/{final_name}'
        elif dataset=='Goodreads':
            predict_data_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/post_re_sorting/{model_name}/Goodreads/{final_name}'
        elif dataset=='Movies_and_TV':
            predict_data_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/post_re_sorting/{model_name}/Movies_5core/{final_name}'
        elif dataset=='Sports_and_Outdoors':
            predict_data_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/post_re_sorting/{model_name}/Sports_5core/{final_name}'
        elif dataset=='Video_Games':
            predict_data_path = f'/Users/yangzaifei/Desktop/courses/COMP 5331/project/post_re_sorting/{model_name}/Games_5core/{final_name}'

        print(f"预测数据路径: {predict_data_path}")

        if with_noise:
            output_dir = f're_ranked/{model_name}/{dataset}/noise_data'
        else:
            output_dir = f're_ranked/{model_name}/{dataset}/clean_data'
        call_llm_re_ranking_and_resolve(demo_map_path, item_titles_path, predict_data_path, output_dir,candaidate_num=candaidate_num)
        
