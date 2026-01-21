

import json
import os
import sys

# Try imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not found. SBERT filtering will be skipped.")

def parse_winner(judge_response):
    """Robustly parse the winner from the judge's response."""
    if not isinstance(judge_response, str):
        return 'error'
    
    response_lower = judge_response.lower()
    
    # Priority on explicit tags
    if '[[a]]' in response_lower: 
        return 'model_a'
    elif '[[b]]' in response_lower: 
        return 'model_b'
    elif '[[c]]' in response_lower or 'tie' in response_lower: 
        return 'tie'
        
    return 'error'

def calculate_biased_self_preference(data_file_path, results_file_path, label):
    """
    Calculate the Biased Self-Preference Rate (SPR).
    SPR = (Judge Picks Self when Human Picked Opponent) / (Total times Human Picked Opponent)
    """
    # Fix paths
    if not os.path.exists(data_file_path):
        base_dir = os.path.dirname(__file__)
        # Fallback attempts
        attempts = [
            os.path.join(base_dir, '../../', data_file_path),
            os.path.join(base_dir, '../..', data_file_path),
            data_file_path
        ]
        for p in attempts:
            if os.path.exists(p):
                data_file_path = p
                break
                
    if not os.path.exists(results_file_path):
        base_dir = os.path.dirname(__file__)
        attempts = [
            os.path.join(base_dir, '../../', results_file_path),
            os.path.join(base_dir, '../..', results_file_path),
            results_file_path
        ]
        for p in attempts:
            if os.path.exists(p):
                results_file_path = p
                break

    print(f"\nProcessing {label}...")
    
    # Load Metadata
    metadata = {}
    with open(data_file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                metadata[item['conversation_id']] = item
            except: pass
            
    # Load Results
    results = []
    with open(results_file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                results.append(item)
            except: pass
            
    # Calculate Metrics per Judge
    judges = ['deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022']
    
    spr_values = []
    
    print(f"  {'Judge':<20} {'SPR':<10} {'(N=)'}")
    print("  " + "-"*40)
    
    for judge_name in judges:
        biased = 0
        opportunities = 0
        
        for result in results:
            conv_id = result.get('conversation_id')
            if conv_id not in metadata:
                continue
            
            meta = metadata[conv_id]
            
            # Judge matching
            if result.get('judge_model') != judge_name:
                continue
            
            # Participation check
            judge_is_a = (meta.get('model_a_name') == judge_name)
            judge_is_b = (meta.get('model_b_name') == judge_name)
            if not (judge_is_a or judge_is_b):
                continue
            
            # Winner determination
            judge_winner = parse_winner(result.get('judge_response', ''))
            human_winner = meta.get('human_winner')
            
            # EXCLUDE TIES
            if judge_winner == 'tie' or judge_winner == 'error' or human_winner == 'tie':
                continue
                
            # Dictionary logic for A/B
            judge_is_model_a = judge_is_a
            
            # Bias Condition: Human Picked Opponent
            human_picked_opponent = False
            if judge_is_model_a and human_winner == 'model_b':
                human_picked_opponent = True
            elif (not judge_is_model_a) and human_winner == 'model_a':
                human_picked_opponent = True
                
            if human_picked_opponent:
                opportunities += 1
                
                # Did judge overrule?
                judge_picked_self = False
                if judge_is_model_a and judge_winner == 'model_a':
                    judge_picked_self = True
                elif (not judge_is_model_a) and judge_winner == 'model_b':
                    judge_picked_self = True
                    
                if judge_picked_self:
                    biased += 1
        
        rate = (biased / opportunities * 100) if opportunities > 0 else 0.0
        spr_values.append(rate)
        
        print(f"  {judge_name:<20} {rate:6.2f}%    ({biased}/{opportunities})")
        
    avg_spr = sum(spr_values) / len(spr_values) if spr_values else 0
    print(f"  AVERAGE              {avg_spr:6.2f}%")
    
    return {
        'avg': avg_spr,
        'deepseek': spr_values[0] if len(spr_values) > 0 else 0,
        'chatgpt': spr_values[1] if len(spr_values) > 1 else 0,
        'claude': spr_values[2] if len(spr_values) > 2 else 0
    }

def filter_with_sbert(data_file_path, temp_file_path):
    """
    Applies SBERT filtering (0.70-0.99) to a dataset and saves a temp filtered version.
    """
    if not SBERT_AVAILABLE:
        return data_file_path
        
    print(f"\n[SBERT] Filtering {os.path.basename(data_file_path)} (Sweet Spot 0.70-0.99)...")
    
    # Validate input path
    if not os.path.exists(data_file_path):
         base_dir = os.path.dirname(__file__)
         # Fallback
         possible = [os.path.join(base_dir, '../../', data_file_path), data_file_path]
         for p in possible:
             if os.path.exists(p):
                 data_file_path = p
                 break
                 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    data = []
    with open(data_file_path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: pass
            
    # Batch Encode
    pairs = []
    for item in data:
        cid = item['conversation_id']
        pair_entry = {'id': cid, 'valid': True}
        if 'original_model_a_response' in item:
            pair_entry['orig_a'] = item['original_model_a_response']
            pair_entry['para_a'] = item['model_a_response']
        if 'original_model_b_response' in item:
            pair_entry['orig_b'] = item['original_model_b_response']
            pair_entry['para_b'] = item['model_b_response']
        pairs.append(pair_entry)
        
    valid_ids = set()
    batch_size = 64
    
    aa_orig = [p['orig_a'] for p in pairs if 'orig_a' in p]
    aa_para = [p['para_a'] for p in pairs if 'para_a' in p]
    
    bb_orig = [p['orig_b'] for p in pairs if 'orig_b' in p]
    bb_para = [p['para_b'] for p in pairs if 'para_b' in p]
    
    # Encode with progress bar hidden to reduce noise, or visible
    # We'll just run it.
    
    emb_a_o, emb_a_p = [], []
    emb_b_o, emb_b_p = [], []
    
    if aa_orig:
        emb_a_o = model.encode(aa_orig, batch_size=batch_size, show_progress_bar=False)
        emb_a_p = model.encode(aa_para, batch_size=batch_size, show_progress_bar=False)
    if bb_orig:
        emb_b_o = model.encode(bb_orig, batch_size=batch_size, show_progress_bar=False)
        emb_b_p = model.encode(bb_para, batch_size=batch_size, show_progress_bar=False)
        
    a_idx = 0
    b_idx = 0
    kept_count = 0
    
    for p in pairs:
        is_valid = True
        if 'orig_a' in p:
            if a_idx < len(emb_a_o):
                sim = cosine_similarity([emb_a_o[a_idx]], [emb_a_p[a_idx]])[0][0]
                if not (0.70 <= sim < 0.99): is_valid = False
                a_idx += 1
        if 'orig_b' in p:
            if b_idx < len(emb_b_o):
                sim = cosine_similarity([emb_b_o[b_idx]], [emb_b_p[b_idx]])[0][0]
                if not (0.70 <= sim < 0.99): is_valid = False
                b_idx += 1
            
        if is_valid:
            valid_ids.add(p['id'])
            kept_count += 1
            
    print(f"  Kept {kept_count}/{len(pairs)} battles ({kept_count/len(pairs)*100:.1f}%)")
    
    filtered_data = [d for d in data if d['conversation_id'] in valid_ids]
    with open(temp_file_path, 'w') as f:
        for item in filtered_data:
            f.write(json.dumps(item) + '\n')
            
    return temp_file_path

print("="*80)
print("COMPREHENSIVE RESULTS VERIFICATION (WITH SBERT FILTERING)")
print("="*80)

baseline = calculate_biased_self_preference(
    'data/raw/judge_samples.jsonl',
    'data/raw/judge_results.jsonl',
    "Baseline"
)

markdown = calculate_biased_self_preference(
    'data/processed/intervention_markdown_strip.jsonl',
    'data/judging_results/intervention_markdown_strip_results.jsonl',
    "Markdown Removal"
)

# Filtered Qwen BT
qwen_bt_temp = 'temp_qwen_bt.jsonl'
path_bt = filter_with_sbert('data/processed/intervention_qwen_chinese.jsonl', qwen_bt_temp)
qwen_bt = calculate_biased_self_preference(
    path_bt, 
    'data/judging_results/intervention_qwen_results.jsonl', 
    "Qwen Back-Translation"
)

# Filtered Qwen Para
qwen_para_temp = 'temp_qwen_para.jsonl'
path_para = filter_with_sbert('data/processed/intervention_qwen_paraphrase.jsonl', qwen_para_temp)
qwen_para = calculate_biased_self_preference(
    path_para, 
    'data/judging_results/intervention_qwen_paraphrase_results.jsonl', 
    "Qwen Paraphrasing"
)

# Cleanup
if os.path.exists(qwen_bt_temp): os.remove(qwen_bt_temp)
if os.path.exists(qwen_para_temp): os.remove(qwen_para_temp)

print("\n" + "="*105)
print("FINAL VERIFIED RESULTS TABLE (SWEET SPOT METHODOLOGY)")
print("="*105)
print(f"{'Judge':<15} | {'Baseline':<10} | {'Markdown':<10} {'(Delta)':<8} | {'BackTrans':<10} {'(Delta)':<8} | {'Paraphrase':<10} {'(Delta)':<8}")
print("-" * 105)

judges = ['DeepSeek', 'ChatGPT', 'Claude']
keys = ['deepseek', 'chatgpt', 'claude']
for i, judge in enumerate(judges):
    base = baseline[keys[i]]
    md = markdown[keys[i]]
    bt = qwen_bt[keys[i]]
    pa = qwen_para[keys[i]]
    
    print(f"{judge:<15} | {base:6.2f}%    | {md:6.2f}%    ({md-base:+.2f}pp) | {bt:6.2f}%    ({bt-base:+.2f}pp) | {pa:6.2f}%    ({pa-base:+.2f}pp)")

print("-" * 105)
av_b = baseline['avg']
av_m = markdown['avg']
av_bt = qwen_bt['avg']
av_p = qwen_para['avg']
print(f"{'AVERAGE':<15} | {av_b:6.2f}%    | {av_m:6.2f}%    ({av_m-av_b:+.2f}pp) | {av_bt:6.2f}%    ({av_bt-av_b:+.2f}pp) | {av_p:6.2f}%    ({av_p-av_b:+.2f}pp)")
print("-" * 105)
# Save to JSON for plotting
results_json = {
    'Baseline': baseline,
    'Markdown Removal': markdown,
    'Back-Translation': qwen_bt,
    'Paraphrasing': qwen_para
}

with open('results/metrics/spr_results.json', 'w') as f:
    json.dump(results_json, f, indent=4)
print(f"\nDone: SPR results saved to results/metrics/spr_results.json")
