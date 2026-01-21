#!/usr/bin/env python3
"""
Calculate per-model classifier accuracy under each intervention.
This shows which models' fingerprints are most/least affected by each intervention.
"""

import sys
import os
sys.path.append('scripts/01_training')

import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from train_mcgovern_classifier import PosTagExtractor
import numpy as np

# Load classifier
print("Loading baseline classifier...")
with open('models/mcgovern_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

def load_intervention_data(file_path):
    """Load intervention data and extract samples"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    df = pd.DataFrame(data)
    
    # Extract samples where model is judge
    texts = []
    labels = []
    for _, row in df.iterrows():
        if row['model_a_name'] == row['judge_model']:
            texts.append(row['model_a_response'])
            labels.append(row['model_a_name'])
        if row['model_b_name'] == row['judge_model']:
            texts.append(row['model_b_response'])
            labels.append(row['model_b_name'])
    
    return texts, labels

# Test on all interventions
interventions = [
    ('Baseline', 'data/raw/judge_samples.jsonl'),
    ('Markdown Removal', 'data/processed/intervention_markdown_strip.jsonl'),
    ('Back-Translation', 'data/processed/intervention_qwen_chinese.jsonl'),
    ('Paraphrasing', 'data/processed/intervention_qwen_paraphrase.jsonl')
]

print("\n" + "="*80)
print("PER-MODEL CLASSIFIER ACCURACY UNDER EACH INTERVENTION")
print("="*80)

results = {}

for intervention_name, file_path in interventions:
    print(f"\nTesting: {intervention_name}...")
    texts, labels = load_intervention_data(file_path)
    
    # Predict
    y_pred = clf.predict(texts)
    y_pred_decoded = clf.label_encoder_.inverse_transform(y_pred)
    
    # Overall accuracy
    overall_acc = accuracy_score(labels, y_pred_decoded)
    
    # Per-model accuracy
    model_accuracies = {}
    for model in clf.label_encoder_.classes_:
        # Get samples for this model
        model_mask = np.array(labels) == model
        if model_mask.sum() > 0:
            model_acc = accuracy_score(
                np.array(labels)[model_mask],
                y_pred_decoded[model_mask]
            )
            model_accuracies[model] = model_acc
    
    results[intervention_name] = {
        'overall': overall_acc,
        'per_model': model_accuracies
    }
    
    print(f"  Overall: {overall_acc*100:.2f}%")
    for model, acc in model_accuracies.items():
        model_short = model.split('-')[0].title()
        print(f"    {model_short}: {acc*100:.2f}%")

# Create summary table
print("\n" + "="*80)
print("SUMMARY TABLE: PER-MODEL ACCURACY UNDER EACH INTERVENTION")
print("="*80)

model_names = {
    'chatgpt-4o-latest-20250326': 'ChatGPT-4o',
    'claude-3-5-haiku-20241022': 'Claude-3.5-Haiku',
    'deepseek-r1-0528': 'DeepSeek-R1'
}

# Print header
print(f"\n{'Intervention':<25} {'Overall':<10} {'ChatGPT':<10} {'Claude':<10} {'DeepSeek':<10}")
print("-" * 80)

baseline_overall = results['Baseline']['overall']
baseline_per_model = results['Baseline']['per_model']

for intervention_name in ['Baseline', 'Markdown Removal', 'Back-Translation', 'Paraphrasing']:
    overall = results[intervention_name]['overall']
    per_model = results[intervention_name]['per_model']
    
    # Format with drops
    if intervention_name == 'Baseline':
        overall_str = f"{overall*100:.2f}%"
        chatgpt_str = f"{per_model['chatgpt-4o-latest-20250326']*100:.2f}%"
        claude_str = f"{per_model['claude-3-5-haiku-20241022']*100:.2f}%"
        deepseek_str = f"{per_model['deepseek-r1-0528']*100:.2f}%"
    else:
        overall_drop = (baseline_overall - overall) * 100
        chatgpt_drop = (baseline_per_model['chatgpt-4o-latest-20250326'] - per_model['chatgpt-4o-latest-20250326']) * 100
        claude_drop = (baseline_per_model['claude-3-5-haiku-20241022'] - per_model['claude-3-5-haiku-20241022']) * 100
        deepseek_drop = (baseline_per_model['deepseek-r1-0528'] - per_model['deepseek-r1-0528']) * 100
        
        overall_str = f"{overall*100:.2f}% ({overall_drop:+.1f}pp)"
        chatgpt_str = f"{per_model['chatgpt-4o-latest-20250326']*100:.2f}% ({chatgpt_drop:+.1f}pp)"
        claude_str = f"{per_model['claude-3-5-haiku-20241022']*100:.2f}% ({claude_drop:+.1f}pp)"
        deepseek_str = f"{per_model['deepseek-r1-0528']*100:.2f}% ({deepseek_drop:+.1f}pp)"
    
    print(f"{intervention_name:<25} {overall_str:<20} {chatgpt_str:<20} {claude_str:<20} {deepseek_str:<20}")

# Save results
output_file = 'results/metrics/per_model_intervention_effects.txt'
with open(output_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PER-MODEL CLASSIFIER ACCURACY UNDER EACH INTERVENTION\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Intervention':<25} {'Overall':<20} {'ChatGPT':<20} {'Claude':<20} {'DeepSeek':<20}\n")
    f.write("-" * 120 + "\n")
    
    for intervention_name in ['Baseline', 'Markdown Removal', 'Back-Translation', 'Paraphrasing']:
        overall = results[intervention_name]['overall']
        per_model = results[intervention_name]['per_model']
        
        if intervention_name == 'Baseline':
            f.write(f"{intervention_name:<25} {overall*100:.2f}%{'':<15} ")
            f.write(f"{per_model['chatgpt-4o-latest-20250326']*100:.2f}%{'':<15} ")
            f.write(f"{per_model['claude-3-5-haiku-20241022']*100:.2f}%{'':<15} ")
            f.write(f"{per_model['deepseek-r1-0528']*100:.2f}%\n")
        else:
            overall_drop = (baseline_overall - overall) * 100
            chatgpt_drop = (baseline_per_model['chatgpt-4o-latest-20250326'] - per_model['chatgpt-4o-latest-20250326']) * 100
            claude_drop = (baseline_per_model['claude-3-5-haiku-20241022'] - per_model['claude-3-5-haiku-20241022']) * 100
            deepseek_drop = (baseline_per_model['deepseek-r1-0528'] - per_model['deepseek-r1-0528']) * 100
            
            f.write(f"{intervention_name:<25} {overall*100:.2f}% ({overall_drop:+.1f}pp){'':<5} ")
            f.write(f"{per_model['chatgpt-4o-latest-20250326']*100:.2f}% ({chatgpt_drop:+.1f}pp){'':<5} ")
            f.write(f"{per_model['claude-3-5-haiku-20241022']*100:.2f}% ({claude_drop:+.1f}pp){'':<5} ")
            f.write(f"{per_model['deepseek-r1-0528']*100:.2f}% ({deepseek_drop:+.1f}pp)\n")

print(f"\nDone: Results saved to {output_file}")

# Save data for plotting
import json
plot_data = {}
for intervention_name in ['Baseline', 'Markdown Removal', 'Back-Translation', 'Paraphrasing']:
    plot_data[intervention_name] = {
        'overall': results[intervention_name]['overall'],
        'chatgpt': results[intervention_name]['per_model']['chatgpt-4o-latest-20250326'],
        'claude': results[intervention_name]['per_model']['claude-3-5-haiku-20241022'],
        'deepseek': results[intervention_name]['per_model']['deepseek-r1-0528']
    }

with open('results/metrics/per_model_intervention_data.json', 'w') as f:
    json.dump(plot_data, f, indent=2)

print("Done: Plot data saved to results/metrics/per_model_intervention_data.json")
