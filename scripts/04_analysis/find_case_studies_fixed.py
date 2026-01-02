import json
import re

def extract_winner(judge_response):
    """Extract winner from judge response text"""
    response_lower = judge_response.lower()
    
    # Look for common patterns
    if 'assistant a' in response_lower or 'response a' in response_lower or '[[a]]' in response_lower:
        return 'model_a'
    elif 'assistant b' in response_lower or 'response b' in response_lower or '[[b]]' in response_lower:
        return 'model_b'
    elif 'tie' in response_lower:
        return 'tie'
    
    # Try to find [[A]] or [[B]] pattern
    match = re.search(r'\[\[([AB])\]\]', judge_response, re.IGNORECASE)
    if match:
        return 'model_a' if match.group(1).upper() == 'A' else 'model_b'
    
    return None

# Load data
print("Loading data...")
with open('judge_samples.jsonl', 'r') as f:
    baseline_meta = {json.loads(line)['conversation_id']: json.loads(line) for line in f}

with open('intervention_qwen_paraphrase.jsonl', 'r') as f:
    para_meta = {json.loads(line)['conversation_id']: json.loads(line) for line in f}

with open('judge_results.jsonl', 'r') as f:
    baseline_results = {json.loads(line)['conversation_id']: json.loads(line) for line in f}

with open('intervention_qwen_paraphrase_results.jsonl', 'r') as f:
    para_results = {json.loads(line)['conversation_id']: json.loads(line) for line in f}

print(f"Loaded {len(baseline_meta)} baseline metadata")
print(f"Loaded {len(para_meta)} paraphrased metadata")
print(f"Loaded {len(baseline_results)} baseline results")
print(f"Loaded {len(para_results)} paraphrased results")

# Find biased self-preference cases
print("\nSearching for biased self-preference cases...")

case_studies = []

for conv_id in para_results:
    if conv_id not in para_meta or conv_id not in baseline_results:
        continue
    
    pm = para_meta[conv_id]
    pr = para_results[conv_id]
    br = baseline_results[conv_id]
    
    # Get judge model
    judge = pr['judge_model']
    
    # Get human winner from baseline
    human_winner = br.get('original_winner')
    if not human_winner:
        continue
    
    # Get judge's verdict
    judge_winner = extract_winner(pr['judge_response'])
    if not judge_winner or judge_winner == 'tie':
        continue
    
    # Check if judge is model_a or model_b
    if judge == pm['model_a_name']:
        judge_position = 'model_a'
        # Biased self-preference: human picked B, judge picked A
        if human_winner == 'model_b' and judge_winner == 'model_a':
            if conv_id in baseline_meta:
                case_studies.append({
                    'conv_id': conv_id,
                    'judge': judge,
                    'judge_position': 'model_a',
                    'human_picked': 'model_b',
                    'judge_picked': 'model_a',
                    'baseline_meta': baseline_meta.get(conv_id),
                    'para_meta': pm,
                    'baseline_result': br,
                    'para_result': pr
                })
    
    elif judge == pm['model_b_name']:
        judge_position = 'model_b'
        # Biased self-preference: human picked A, judge picked B
        if human_winner == 'model_a' and judge_winner == 'model_b':
            if conv_id in baseline_meta:
                case_studies.append({
                    'conv_id': conv_id,
                    'judge': judge,
                    'judge_position': 'model_b',
                    'human_picked': 'model_a',
                    'judge_picked': 'model_b',
                    'baseline_meta': baseline_meta.get(conv_id),
                    'para_meta': pm,
                    'baseline_result': br,
                    'para_result': pr
                })

print(f"Found {len(case_studies)} biased self-preference cases")

# Group by judge
by_judge = {}
for case in case_studies:
    judge = case['judge']
    if judge not in by_judge:
        by_judge[judge] = []
    by_judge[judge].append(case)

print("\nBy judge:")
for judge, cases in by_judge.items():
    print(f"  {judge}: {len(cases)} cases")

# Select diverse examples
selected = []
for judge in ['deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022']:
    if judge in by_judge and len(by_judge[judge]) > 0:
        selected.extend(by_judge[judge][:2])

selected = selected[:5]

print(f"\nSelected {len(selected)} case studies")

# Write detailed case studies
with open('case_studies_persistent_bias.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CASE STUDIES: PERSISTENT BIAS DESPITE PARAPHRASING\n")
    f.write("="*80 + "\n\n")
    
    for i, case in enumerate(selected, 1):
        f.write("="*80 + "\n")
        f.write(f"CASE STUDY {i}/{len(selected)}\n")
        f.write("="*80 + "\n")
        f.write(f"Judge: {case['judge']}\n")
        f.write(f"Human Picked: {case['human_picked']}\n")
        f.write(f"Judge Picked: {case['judge_picked']} (ITSELF - BIASED)\n\n")
        
        # User prompt
        f.write("USER PROMPT:\n")
        f.write("-" * 80 + "\n")
        f.write(case['para_meta']['user_prompt'][:400] + "...\n\n")
        
        # Judge's response
        if case['judge_position'] == 'model_a':
            orig = case['baseline_meta']['model_a_response']
            para = case['para_meta']['model_a_response']
            opp = case['para_meta']['model_b_response']
            opp_name = case['para_meta']['model_b_name']
        else:
            orig = case['baseline_meta']['model_b_response']
            para = case['para_meta']['model_b_response']
            opp = case['para_meta']['model_a_response']
            opp_name = case['para_meta']['model_a_name']
        
        f.write(f"JUDGE'S RESPONSE ({case['judge']}) - ORIGINAL:\n")
        f.write("-" * 80 + "\n")
        f.write(orig[:600] + "...\n\n")
        
        f.write(f"JUDGE'S RESPONSE ({case['judge']}) - PARAPHRASED:\n")
        f.write("-" * 80 + "\n")
        f.write(para[:600] + "...\n\n")
        
        f.write(f"OPPONENT'S RESPONSE ({opp_name}):\n")
        f.write("-" * 80 + "\n")
        f.write(opp[:600] + "...\n\n")
        
        f.write("ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write("Despite Qwen paraphrasing to remove style, judge STILL picked itself.\n")
        f.write("This reveals persistent patterns:\n")
        f.write("- Deep linguistic structures\n")
        f.write("- Semantic framing\n")
        f.write("- Domain vocabulary\n")
        f.write("- Argumentation style\n\n")

print(f"\nWrote case studies to: case_studies_persistent_bias.txt")
