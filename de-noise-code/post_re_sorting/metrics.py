import numpy as np
import json
def recall_at_k(predict_infos, k, ori=True):
    """Compute Recall@k."""
    hits = 0
    for info in predict_infos:
        label = str(info["label"])
        if ori:
            preds = [str(x) for x in info["candidate_preds"][:k]]
        else:
            preds = [str(x) for x in info["re_ranked_candidates"][:k]]
        if label in preds:
            hits += 1
    return hits / len(predict_infos)


def ndcg_at_k(predict_infos, k, ori=True):
    """Compute NDCG@k."""
    ndcg = 0.0
    for info in predict_infos:
        label = str(info["label"])
        if ori:
            preds = [str(x) for x in info["candidate_preds"][:k]]
        else:
            preds = [str(x) for x in info["re_ranked_candidates"][:k]]
        if label in preds:
            rank = preds.index(label)
            ndcg += 1 / np.log2(rank + 2)  # +2 since rank starts from 0
    return ndcg / len(predict_infos)


if __name__ == "__main__":
    # print("4o-mini_demo_0_llm_re_ranked_results.json")
    # predict_infos = json.load(open("4o-mini_demo_0_llm_re_ranked_results.json", "r"))
    
    # print("4o-mini_demo_5_llm_re_ranked_results.json")
    # predict_infos = json.load(open("4o-mini_demo_5_llm_re_ranked_results.json", "r"))

    # print("4o-mini_demo_0_new_llm_re_ranked_results.json")
    # predict_infos = json.load(open("4o-mini_demo_0_new_llm_re_ranked_results.json", "r"))

    # print("4.1-mini_demo_0_new_llm_re_ranked_results.json")
    # predict_infos = json.load(open("4.1-mini_demo_0_new_llm_re_ranked_results.json", "r"))
    
    # print("4o-mini_demo_0_new_full_llm_re_ranked_results.json")
    # predict_infos = json.load(open("4o-mini_demo_0_new_full_llm_re_ranked_results.json", "r"))


    # print("top40_llm_re_ranked_results.json")
    # predict_infos = json.load(open("top40_llm_re_ranked_results.json", "r"))

    # model_name = 'Qwen2-0.5'
    model_name = 'Smollm2-135'

    # dataset= 'Arts_Crafts_and_Sewing'
    dataset= 'Baby_Products'
    dataset= 'Goodreads'
    # dataset= 'Movies_and_TV'
    # dataset= 'Sports_and_Outdoors'
    # dataset= 'Video_Games'
    with_noise = 0

    if with_noise:
        output_dir = f're_ranked/{model_name}/{dataset}/noise_data'
    else:
        output_dir = f're_ranked/{model_name}/{dataset}/clean_data'

    # if with_noise:
    #     output_dir = f're_ranked/{model_name}/{dataset}/noise_data'
    # else:
    #     output_dir = f're_ranked/{model_name}/{dataset}/clean_data/top40'

    
    
    print('model name: ', model_name)
    print('dataset name: ', dataset)
    print('with noise: ', with_noise)
    print(output_dir)
    predict_infos = json.load(open(f"{output_dir}/llm_re_ranked_results.json", "r"))
    
    # predict_infos = predict_infos[:1000]  # 只评估前1000条，节省时间
    
    print(f"Evaluating {len(predict_infos)} samples...")

    print("Original Ranking:")
    num_list = []
    for k in [10, 20]:
        r = recall_at_k(predict_infos, k)
        n = ndcg_at_k(predict_infos, k)
        print(f"Recall@{k}: {r:.4f}, NDCG@{k}: {n:.4f}")
        num_list.append(str(round(r,4)))
        num_list.append(str(round(n,4)))
    print(' & '.join(num_list) )       

    num_list = []
    print("\nLLM Re-Ranked:")
    for k in [10, 20]:
        r = recall_at_k(predict_infos, k, ori=False)
        n = ndcg_at_k(predict_infos, k, ori=False)
        print(f"Recall@{k}: {r:.4f}, NDCG@{k}: {n:.4f}")

        num_list.append(str(round(r,4)))
        num_list.append(str(round(n,4)))
    print(' & '.join(num_list) )  
