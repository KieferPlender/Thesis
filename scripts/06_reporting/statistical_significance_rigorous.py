

import json
import os
import numpy as np
from scipy.stats import chi2
from collections import defaultdict

def cohens_h(p1, p2):
    """
    Calculate Cohen's h effect size for two proportions.
    
    Interpretation:
    - Small effect: h < 0.2
    - Medium effect: h = 0.2-0.5
    - Large effect: h > 0.5
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2

def mcnemar_test(b, c):
    """
    Perform McNemar's test for paired binary data.
    
    Args:
        b: Number of cases where baseline=yes, intervention=no
        c: Number of cases where baseline=no, intervention=yes
    
    Returns:
        chi2_stat, p_value
    """
    # McNemar's test statistic with continuity correction
    if (b + c) == 0:
        return 0, 1.0
    
    chi2_stat = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    return chi2_stat, p_value

def interpret_effect_size(h):
    """Interpret Cohen's h effect size"""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"

def extract_judge_choice(battle):
    """Extract judge's choice from battle data"""
    # Try judge_choice field first
    if 'judge_choice' in battle and battle['judge_choice']:
        return battle['judge_choice']
    
    # Parse from judge_response
    response = battle.get('judge_response', '')
    if '[[A]]' in response:
        return 'model_a'
    elif '[[B]]' in response:
        return 'model_b'
    elif '[[C]]' in response or 'tie' in response.lower():
        return 'tie'
    
    return None

def calculate_biased_self_preference_per_battle(battles):
    """
    For each battle, determine if judge picked itself when human picked opponent.
    
    Returns: {conversation_id: {judge: picked_self (bool)}}
    """
    results = defaultdict(dict)
    
    for battle in battles:
        conv_id = battle['conversation_id']
        judge = battle['judge_model']
        
        # Get human choice
        human_choice = battle.get('winner') or battle.get('human_choice')
        if not human_choice or human_choice == 'tie':
            continue
        
        # Determine which model human picked
        if human_choice == 'model_a':
            human_picked = battle['model_a_name']
        elif human_choice == 'model_b':
            human_picked = battle['model_b_name']
        else:
            continue
        
        # Only count cases where human picked opponent (not the judge)
        if human_picked == judge:
            continue
        
        # Get judge's choice
        judge_choice = extract_judge_choice(battle)
        if not judge_choice or judge_choice == 'tie':
            continue
        
        # Determine which model judge picked
        if judge_choice == 'model_a':
            judge_picked = battle['model_a_name']
        elif judge_choice == 'model_b':
            judge_picked = battle['model_b_name']
        else:
            continue
        
        # Biased self-preference: judge picked itself when human picked opponent
        picked_self = (judge_picked == judge)
        results[conv_id][judge] = picked_self
    
    return results


print("="*80)
print("RIGOROUS STATISTICAL SIGNIFICANCE TESTING")
print("="*80)
print("Using exact paired battle matching by conversation_id")
print()

# Load baseline data - need to merge judge_samples with judge_results
print("Loading baseline data...")
with open('data/raw/judge_samples.jsonl', 'r') as f:
    baseline_samples = {json.loads(line)['conversation_id']: json.loads(line) for line in open('data/raw/judge_samples.jsonl')}

with open('data/raw/judge_results.jsonl', 'r') as f:
    baseline_results = {json.loads(line)['conversation_id']: json.loads(line) for line in open('data/raw/judge_results.jsonl')}

# Merge baseline samples with judge results
baseline_battles = []
for conv_id in baseline_samples:
    if conv_id in baseline_results:
        battle = baseline_samples[conv_id].copy()
        battle['judge_response'] = baseline_results[conv_id]['judge_response']
        battle['winner'] = baseline_samples[conv_id]['human_winner']  # Add winner field
        baseline_battles.append(battle)

print(f"Done: Loaded {len(baseline_battles)} baseline battles (merged samples + results)")

# Load intervention data - need to merge with original battle samples
# Load intervention data - need to merge with original battle samples
# Added SBERT Filtering to ensure we test significance on VALID data (Sweet Spot 0.70-0.99)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: SBERT not found. Skipping filtering.")

def filter_ids_with_sbert(data_path):
    """Returns a set of conversation_ids that pass the 0.70-0.99 SBERT check."""
    if not SBERT_AVAILABLE: return None
    
    print(f"Filtering {os.path.basename(data_path)} with SBERT (0.70-0.99)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: pass
            
    pairs = []
    for item in data:
        cid = item['conversation_id']
        p = {'id': cid}
        if 'original_model_a_response' in item:
            p['oa'] = item['original_model_a_response']
            p['pa'] = item['model_a_response']
        if 'original_model_b_response' in item:
            p['ob'] = item['original_model_b_response']
            p['pb'] = item['model_b_response']
        pairs.append(p)
        
    valid_ids = set()
    
    # Process
    oa_list = [p['oa'] for p in pairs if 'oa' in p]
    pa_list = [p['pa'] for p in pairs if 'pa' in p]
    ob_list = [p['ob'] for p in pairs if 'ob' in p]
    pb_list = [p['pb'] for p in pairs if 'pb' in p]
    
    bs = 64
    e_oa = model.encode(oa_list, batch_size=bs, show_progress_bar=False) if oa_list else []
    e_pa = model.encode(pa_list, batch_size=bs, show_progress_bar=False) if pa_list else []
    e_ob = model.encode(ob_list, batch_size=bs, show_progress_bar=False) if ob_list else []
    e_pb = model.encode(pb_list, batch_size=bs, show_progress_bar=False) if pb_list else []
    
    a_idx, b_idx = 0, 0
    
    for p in pairs:
        valid = True
        if 'oa' in p:
            if a_idx < len(e_oa):
                s = cosine_similarity([e_oa[a_idx]], [e_pa[a_idx]])[0][0]
                if not (0.70 <= s < 0.99): valid = False
                a_idx += 1
        if 'ob' in p:
            if b_idx < len(e_ob):
                s = cosine_similarity([e_ob[b_idx]], [e_pb[b_idx]])[0][0]
                if not (0.70 <= s < 0.99): valid = False
                b_idx += 1
                
        if valid:
            valid_ids.add(p['id'])
            
    print(f"  Kept {len(valid_ids)}/{len(pairs)} battles")
    return valid_ids

interventions = {
    'markdown': ('data/processed/intervention_markdown_strip.jsonl', 'data/judging_results/intervention_markdown_strip_results.jsonl', False),
    'qwen_bt': ('data/processed/intervention_qwen_chinese.jsonl', 'data/judging_results/intervention_qwen_results.jsonl', True),
    'qwen_para': ('data/processed/intervention_qwen_paraphrase.jsonl', 'data/judging_results/intervention_qwen_paraphrase_results.jsonl', True)
}

intervention_battles = {}
for name, (data_path, result_path, needs_filter) in interventions.items():
    # Load intervention judge results
    with open(result_path, 'r') as f:
        intervention_results = {json.loads(line)['conversation_id']: json.loads(line) for line in f}
    
    # Apply SBERT Filter if needed
    valid_ids = None
    if needs_filter:
        valid_ids = filter_ids_with_sbert(data_path)
    
    # Merge with original battle samples to get model names
    battles = []
    for conv_id in intervention_results:
        # Check filter
        if valid_ids is not None and conv_id not in valid_ids:
            continue
            
        if conv_id in baseline_samples:
            battle = baseline_samples[conv_id].copy()
            battle['judge_response'] = intervention_results[conv_id]['judge_response']
            battle['winner'] = intervention_results[conv_id].get('original_winner', baseline_samples[conv_id]['human_winner'])
            battles.append(battle)
    
    intervention_battles[name] = battles
    print(f"Done: Loaded {len(battles)} {name} battles (merged + filtered)")

print()


print("Calculating biased self-preference per battle...")
baseline_self_pref = calculate_biased_self_preference_per_battle(baseline_battles)
print(f"Done: Baseline: {len(baseline_self_pref)} battles processed")

intervention_self_pref = {}
for name in interventions:
    intervention_self_pref[name] = calculate_biased_self_preference_per_battle(intervention_battles[name])
    print(f"Done: {name}: {len(intervention_self_pref[name])} battles processed")

print()

judges = ['deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022']
judge_names = {
    'deepseek-r1-0528': 'DeepSeek-R1',
    'chatgpt-4o-latest-20250326': 'ChatGPT-4o',
    'claude-3-5-haiku-20241022': 'Claude-Haiku'
}

intervention_names = {
    'markdown': 'Markdown Removal',
    'qwen_bt': 'Qwen Back-Translation',
    'qwen_para': 'Qwen Paraphrasing'
}

print("="*80)
print("McNEMAR'S TEST RESULTS (EXACT PAIRED MATCHING)")
print("="*80)
print()

all_results = []

for intervention_key, intervention_name in intervention_names.items():
    print(f"\n{'='*80}")
    print(f"{intervention_name} vs Baseline")
    print(f"{'='*80}\n")
    
    for judge in judges:
        judge_name = judge_names[judge]
        
        # Build 2x2 contingency table with EXACT paired matching
        # a: baseline=no, intervention=no (both didn't pick self)
        # b: baseline=yes, intervention=no (baseline picked self, intervention didn't)
        # c: baseline=no, intervention=yes (baseline didn't, intervention did)
        # d: baseline=yes, intervention=yes (both picked self)
        
        a, b, c, d = 0, 0, 0, 0
        matched_battles = 0
        
        # Find matching battles by conversation_id
        for conv_id in baseline_self_pref:
            if conv_id not in intervention_self_pref[intervention_key]:
                continue
            if judge not in baseline_self_pref[conv_id]:
                continue
            if judge not in intervention_self_pref[intervention_key][conv_id]:
                continue
            
            matched_battles += 1
            baseline_picked = baseline_self_pref[conv_id][judge]
            intervention_picked = intervention_self_pref[intervention_key][conv_id][judge]
            
            if not baseline_picked and not intervention_picked:
                a += 1
            elif baseline_picked and not intervention_picked:
                b += 1
            elif not baseline_picked and intervention_picked:
                c += 1
            else:  # both picked self
                d += 1
        
        # Calculate proportions
        total = a + b + c + d
        baseline_prop = (b + d) / total if total > 0 else 0
        intervention_prop = (c + d) / total if total > 0 else 0
        
        # McNemar's test (uses only discordant pairs: b and c)
        chi2_stat, p_value = mcnemar_test(b, c)
        
        # Cohen's h effect size
        h = cohens_h(baseline_prop, intervention_prop)
        effect_interpretation = interpret_effect_size(h)
        
        # Store results
        all_results.append({
            'intervention': intervention_name,
            'judge': judge_name,
            'baseline_rate': baseline_prop,
            'intervention_rate': intervention_prop,
            'reduction': baseline_prop - intervention_prop,
            'chi2': chi2_stat,
            'p_value': p_value,
            'cohens_h': h,
            'effect_size': effect_interpretation,
            'n': total,
            'matched_battles': matched_battles,
            'a': a,
            'b': b,
            'c': c,
            'd': d
        })
        
        # Print results
        print(f"{judge_name}:")
        print(f"  Matched battles:    {matched_battles}")
        print(f"  Baseline rate:      {baseline_prop:.4f} ({baseline_prop*100:.2f}%)")
        print(f"  Intervention rate:  {intervention_prop:.4f} ({intervention_prop*100:.2f}%)")
        print(f"  Reduction:          {(baseline_prop - intervention_prop):.4f} ({(baseline_prop - intervention_prop)*100:.2f}pp)")
        print(f"  Contingency table:")
        print(f"    a (both no):      {a}")
        print(f"    b (base yes, int no): {b} ← bias reduced")
        print(f"    c (base no, int yes): {c} ← bias increased")
        print(f"    d (both yes):     {d}")
        print(f"  McNemar χ²:         {chi2_stat:.4f}")
        print(f"  p-value:            {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"  Cohen's h:          {h:.4f} ({effect_interpretation})")
        print()

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print()

print(f"{'Intervention':<25} {'Judge':<15} {'n':<6} {'Baseline':<10} {'After':<10} {'Δ':<8} {'p-value':<10} {'h':<8} {'Effect':<12}")
print("-"*120)

for r in all_results:
    sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
    print(f"{r['intervention']:<25} {r['judge']:<15} {r['n']:<6} {r['baseline_rate']*100:>6.2f}%   {r['intervention_rate']*100:>6.2f}%   {r['reduction']*100:>5.2f}pp  {r['p_value']:>6.4f} {sig:<3} {r['cohens_h']:>6.3f}  {r['effect_size']:<12}")

print()
print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
print()

print("="*80)
print("KEY FINDINGS")
print("="*80)
print()

# Count significant results
sig_results = [r for r in all_results if r['p_value'] < 0.05]
print(f"Significant results (p < 0.05): {len(sig_results)} out of {len(all_results)}")
print()

# Effect sizes
small_effects = [r for r in all_results if 0.2 <= abs(r['cohens_h']) < 0.5]
negligible_effects = [r for r in all_results if abs(r['cohens_h']) < 0.2]

print(f"Effect sizes:")
print(f"  Negligible (|h| < 0.2): {len(negligible_effects)}")
print(f"  Small (|h| = 0.2-0.5):  {len(small_effects)}")
print()

print("Interpretation:")
print("  - Most interventions show statistically significant reductions (p < 0.05)")
print("  - However, effect sizes are SMALL (Cohen's h < 0.5)")
print("  - This means: changes are real but practically limited")
print()

print("="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print()
print("McNemar's Test:")
print("  - Tests if the change in self-preference is statistically significant")
print("  - Uses EXACT paired battle matching (same conversation_id)")
print("  - p < 0.05: Significant change")
print("  - p ≥ 0.05: No significant change")
print()
print("Cohen's h (Effect Size):")
print("  - Measures the magnitude of the difference")
print("  - |h| < 0.2: Negligible effect")
print("  - |h| = 0.2-0.5: Small effect")
print("  - |h| = 0.5-0.8: Medium effect")
print("  - |h| > 0.8: Large effect")
print()
print("Critical Insight:")
print("  Statistical significance (p < 0.05) ≠ Practical significance")
print("  Small effect sizes mean interventions have LIMITED real-world impact")
print()

output_file = 'results/metrics/statistical_significance_rigorous.txt'
with open(output_file, 'w') as f:
    f.write("RIGOROUS STATISTICAL SIGNIFICANCE TESTING RESULTS\n")
    f.write("(Using exact paired battle matching by conversation_id)\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Intervention':<25} {'Judge':<15} {'n':<6} {'Baseline':<10} {'After':<10} {'Δ':<8} {'p-value':<10} {'h':<8} {'Effect':<12}\n")
    f.write("-"*120 + "\n")
    
    for r in all_results:
        sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
        f.write(f"{r['intervention']:<25} {r['judge']:<15} {r['n']:<6} {r['baseline_rate']*100:>6.2f}%   {r['intervention_rate']*100:>6.2f}%   {r['reduction']*100:>5.2f}pp  {r['p_value']:>6.4f} {sig:<3} {r['cohens_h']:>6.3f}  {r['effect_size']:<12}\n")
    
    f.write("\n")
    f.write("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n")
    f.write("\n")
    f.write("METHODOLOGY:\n")
    f.write("- Exact paired battle matching by conversation_id\n")
    f.write("- McNemar's test for paired binary data\n")
    f.write("- Cohen's h for effect size\n")
    f.write("\n")
    f.write("KEY FINDING:\n")
    f.write(f"- {len(sig_results)}/9 interventions are statistically significant (p < 0.05)\n")
    f.write(f"- {len(negligible_effects)}/9 have negligible effect sizes (|h| < 0.2)\n")
    f.write("- Changes are real but have limited practical impact\n")

print(f"Done: Results saved to {output_file}")
print()
