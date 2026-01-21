from datasets import load_dataset
import pandas as pd
import numpy as np
import json

# Judge models
JUDGE_MODELS = [
    'chatgpt-4o-latest-20250326',
    'deepseek-r1-0528',
    'claude-3-5-haiku-20241022'
]

print("="*70)
print("Step 1: Loading judge_samples.jsonl to get excluded IDs")
print("="*70)

# Load judge samples to get conversation IDs to exclude
judge_data = []
with open('judge_samples.jsonl', 'r') as f:
    for line in f:
        try:
            judge_data.append(json.loads(line))
        except:
            pass

df_judge = pd.DataFrame(judge_data)
excluded_ids = set(df_judge['conversation_id'].unique())

print(f"Loaded {len(df_judge)} judge samples")
print(f"Unique conversation IDs to exclude: {len(excluded_ids)}")

print("\n" + "="*70)
print("Step 2: Loading LMSYS Arena dataset")
print("="*70)

ds = load_dataset("lmarena-ai/arena-human-preference-140k")
df = pd.DataFrame(ds['train'])

print(f"Initial dataset size: {len(df):,}")

# Apply filters
if 'language' in df.columns:
    df = df[df['language'] == 'en'].copy()
    print(f"After English filter: {len(df):,}")

if 'is_code' in df.columns:
    df = df[df['is_code'] == False].copy()
    print(f"After code filter: {len(df):,}")

def is_math(tag):
    if isinstance(tag, dict):
        math_tag = tag.get('math_v0.1', {})
        if isinstance(math_tag, dict) and math_tag.get('math', False):
            return True
    return False

if 'category_tag' in df.columns:
    mask = df['category_tag'].apply(is_math)
    df = df[~mask].copy()
    print(f"After math filter: {len(df):,}")

# Exclude judge samples
df = df[~df['id'].isin(excluded_ids)].copy()
print(f"After excluding judge samples: {len(df):,}")

print("\n" + "="*70)
print("Step 3: Sampling 1,000 battles per judge for classifier training")
print("="*70)

def extract_text_content(content_list):
    """Extracts text from content list"""
    if isinstance(content_list, np.ndarray):
        content_list = list(content_list)
    elif not isinstance(content_list, list):
        return ""
    
    extracted_text = []
    for item in content_list:
        if isinstance(item, dict) and 'text' in item and item['text'] is not None:
            extracted_text.append(item['text'])
    return " ".join(extracted_text).strip()

all_samples = []

for judge in JUDGE_MODELS:
    # Find battles where this judge is either model_a or model_b
    judge_battles = df[
        (df['model_a'] == judge) |
        (df['model_b'] == judge)
    ].copy()
    
    available = len(judge_battles)
    print(f"\n{judge}:")
    print(f"  Available battles: {available:,}")
    
    if available < 1000:
        print(f"  WARNING: Only {available} available (requested 1,000). Taking all.")
        sample = judge_battles
    else:
        sample = judge_battles.sample(n=1000, random_state=42)
        print(f"  Sampled: 1,000")
    
    # Process conversations
    for _, row in sample.iterrows():
        user_prompt = ""
        model_a_response = ""
        model_b_response = ""
        
        full_conversation_data = row['full_conversation']
        
        # Handle numpy array
        if isinstance(full_conversation_data, np.ndarray):
            if full_conversation_data.ndim == 0:
                full_conversation_data = full_conversation_data.item()
            elif full_conversation_data.ndim == 1 and full_conversation_data.size > 0:
                full_conversation_data = full_conversation_data[0]
            else:
                full_conversation_data = []
        
        if not isinstance(full_conversation_data, list):
            full_conversation_data = []
        
        if full_conversation_data and len(full_conversation_data) > 0:
            first_turn = full_conversation_data[0]
            user_prompt = extract_text_content(first_turn.get('user', {}).get('content', []))
            model_a_response = extract_text_content(first_turn.get('model_side_a', {}).get('content', []))
            model_b_response = extract_text_content(first_turn.get('model_side_b', {}).get('content', []))
        
        all_samples.append({
            'judge_model': judge,
            'conversation_id': row['id'],
            'user_prompt': user_prompt,
            'model_a_name': row['model_a'],
            'model_a_response': model_a_response,
            'model_b_name': row['model_b'],
            'model_b_response': model_b_response,
            'human_winner': row['winner']
        })

print("\n" + "="*70)
print("Step 4: Saving classifier training data")
print("="*70)

df_final = pd.DataFrame(all_samples)

print(f"\nFinal dataset breakdown:")
print(df_final['judge_model'].value_counts())
print(f"\nTotal battles: {len(df_final)}")

# Verify no overlap with judge_samples
overlap = set(df_final['conversation_id'].unique()).intersection(excluded_ids)
if overlap:
    print(f"\n⚠ WARNING: {len(overlap)} conversation IDs overlap with judge_samples!")
else:
    print(f"\nDone: No overlap with judge_samples.jsonl")

# Save
output_file = "classifier_training_samples.jsonl"
df_final.to_json(output_file, orient='records', lines=True)

print(f"\nDone: Saved to {output_file}")
print(f"\nThis data will be used to train the McGovern classifier.")
print(f"The classifier will then be tested on:")
print(f"  1. judge_samples.jsonl (baseline)")
print(f"  2. backtranslated_samples.jsonl (intervention)")
print(f"  3. paraphrased_samples.jsonl (intervention)")
