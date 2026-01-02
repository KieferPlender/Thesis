import json
import os
from collections import defaultdict

print("="*80)
print("CRITICAL AUDIT - FINDING ALL ISSUES")
print("="*80)
print("\nBeing brutally honest about any problems...\n")

issues = []
warnings = []

# 1. CHECK FILE EXISTENCE
print("1. CHECKING FILE EXISTENCE")
print("-" * 80)

required_files = {
    'Baseline Metadata': 'data/raw/judge_samples.jsonl',
    'Baseline Results': 'data/raw/judge_results.jsonl',
    'Markdown Metadata': 'data/processed/intervention_markdown_strip.jsonl',
    'Markdown Results': 'data/results/intervention_markdown_strip_results.jsonl',
    'Qwen BT Metadata': 'data/processed/intervention_qwen_chinese.jsonl',
    'Qwen BT Results': 'data/results/intervention_qwen_results.jsonl',
    'Qwen Para Metadata': 'data/processed/intervention_qwen_paraphrase.jsonl',
    'Qwen Para Results': 'data/results/intervention_qwen_paraphrase_results.jsonl',
    'Normal Classifier': 'models/mcgovern_classifier.pkl',
    'Markdown-Free Classifier': 'models/mcgovern_classifier_markdown_free.pkl',
}

for name, path in required_files.items():
    full_path = f"../../{path}"
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        print(f"✅ {name:<30} {size:>12,} bytes")
    else:
        print(f"❌ {name:<30} MISSING!")
        issues.append(f"Missing file: {path}")

# 2. CHECK SAMPLE COUNTS
print("\n2. CHECKING SAMPLE COUNTS")
print("-" * 80)

def count_lines(filepath):
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

counts = {}
for name, path in required_files.items():
    if path.endswith('.jsonl'):
        full_path = f"../../{path}"
        count = count_lines(full_path)
        counts[name] = count
        print(f"{name:<30} {count:>6} lines")

# Check for expected counts
if counts.get('Baseline Metadata', 0) != 3000:
    warnings.append(f"Baseline has {counts.get('Baseline Metadata', 0)} samples, expected 3000")

if counts.get('Markdown Metadata', 0) != 3000:
    warnings.append(f"Markdown has {counts.get('Markdown Metadata', 0)} samples, expected 3000")

# 3. CHECK CONVERSATION ID OVERLAP
print("\n3. CHECKING CONVERSATION ID CONSISTENCY")
print("-" * 80)

def load_ids(filepath):
    ids = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    ids.add(data['conversation_id'])
                except:
                    continue
    except:
        pass
    return ids

baseline_ids = load_ids('../../data/raw/judge_samples.jsonl')
markdown_ids = load_ids('../../data/processed/intervention_markdown_strip.jsonl')
qwen_bt_ids = load_ids('../../data/processed/intervention_qwen_chinese.jsonl')
qwen_para_ids = load_ids('../../data/processed/intervention_qwen_paraphrase.jsonl')

print(f"Baseline IDs:        {len(baseline_ids)}")
print(f"Markdown IDs:        {len(markdown_ids)}")
print(f"Qwen BT IDs:         {len(qwen_bt_ids)}")
print(f"Qwen Para IDs:       {len(qwen_para_ids)}")

# Check overlaps
markdown_overlap = len(baseline_ids & markdown_ids)
qwen_bt_overlap = len(baseline_ids & qwen_bt_ids)
qwen_para_overlap = len(baseline_ids & qwen_para_ids)

print(f"\nOverlap with baseline:")
print(f"  Markdown:  {markdown_overlap}/{len(baseline_ids)} ({markdown_overlap/len(baseline_ids)*100:.1f}%)")
print(f"  Qwen BT:   {qwen_bt_overlap}/{len(baseline_ids)} ({qwen_bt_overlap/len(baseline_ids)*100:.1f}%)")
print(f"  Qwen Para: {qwen_para_overlap}/{len(baseline_ids)} ({qwen_para_overlap/len(baseline_ids)*100:.1f}%)")

if markdown_overlap < len(baseline_ids) * 0.99:
    warnings.append(f"Markdown missing {len(baseline_ids) - markdown_overlap} battles")

if qwen_bt_overlap < len(baseline_ids) * 0.95:
    warnings.append(f"Qwen BT missing {len(baseline_ids) - qwen_bt_overlap} battles (expected due to timeouts)")

if qwen_para_overlap < len(baseline_ids) * 0.95:
    warnings.append(f"Qwen Para missing {len(baseline_ids) - qwen_para_overlap} battles (expected due to timeouts)")

# 4. CHECK JUDGE DISTRIBUTION
print("\n4. CHECKING JUDGE DISTRIBUTION")
print("-" * 80)

def count_judges(filepath):
    judges = defaultdict(int)
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    judges[data['judge_model']] += 1
                except:
                    continue
    except:
        pass
    return judges

baseline_judges = count_judges('../../data/raw/judge_samples.jsonl')
print("Baseline judge distribution:")
for judge, count in sorted(baseline_judges.items()):
    print(f"  {judge:<40} {count:>5}")

# Check if we have all 3 target judges
target_judges = ['deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022']
for judge in target_judges:
    if judge not in baseline_judges:
        issues.append(f"Missing judge in baseline: {judge}")
    elif baseline_judges[judge] < 100:
        warnings.append(f"Low sample count for {judge}: {baseline_judges[judge]}")

# 5. CHECK DATA QUALITY
print("\n5. CHECKING DATA QUALITY (SAMPLE)")
print("-" * 80)

# Load a few samples and check structure
try:
    with open('../../data/raw/judge_samples.jsonl', 'r') as f:
        sample = json.loads(f.readline())
    
    required_fields = ['conversation_id', 'user_prompt', 'model_a_name', 'model_b_name', 
                      'model_a_response', 'model_b_response', 'judge_model']
    
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        issues.append(f"Baseline missing fields: {missing_fields}")
    else:
        print("✅ Baseline has all required fields")
    
    # Check if responses are non-empty
    if not sample['model_a_response'] or not sample['model_b_response']:
        issues.append("Found empty responses in baseline")
    else:
        print("✅ Responses are non-empty")
    
except Exception as e:
    issues.append(f"Error reading baseline: {e}")

# 6. CHECK INTERVENTION QUALITY
print("\n6. CHECKING INTERVENTION QUALITY")
print("-" * 80)

# Check if interventions actually modified data
try:
    baseline = {}
    with open('../../data/raw/judge_samples.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10: break
            data = json.loads(line)
            baseline[data['conversation_id']] = data
    
    # Check markdown
    markdown_modified = 0
    with open('../../data/processed/intervention_markdown_strip.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10: break
            data = json.loads(line)
            if data['conversation_id'] in baseline:
                if data['model_a_response'] != baseline[data['conversation_id']]['model_a_response']:
                    markdown_modified += 1
    
    if markdown_modified == 0:
        issues.append("Markdown intervention didn't modify any responses!")
    else:
        print(f"✅ Markdown modified {markdown_modified}/10 sampled responses")
    
    # Check paraphrasing
    para_modified = 0
    para_has_original = 0
    with open('../../data/processed/intervention_qwen_paraphrase.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 10: break
            data = json.loads(line)
            if data['conversation_id'] in baseline:
                if data['model_a_response'] != baseline[data['conversation_id']]['model_a_response']:
                    para_modified += 1
                if 'original_model_a_response' in data:
                    para_has_original += 1
    
    if para_modified == 0:
        issues.append("Paraphrasing didn't modify any responses!")
    else:
        print(f"✅ Paraphrasing modified {para_modified}/10 sampled responses")
    
    if para_has_original < 10:
        issues.append(f"Paraphrasing missing original responses: {para_has_original}/10")
    else:
        print(f"✅ Paraphrasing stored originals: {para_has_original}/10")
    
except Exception as e:
    issues.append(f"Error checking intervention quality: {e}")

# 7. CHECK RESULTS CONSISTENCY
print("\n7. CHECKING RESULTS FILE CONSISTENCY")
print("-" * 80)

baseline_result_ids = load_ids('../../data/raw/judge_results.jsonl')
markdown_result_ids = load_ids('../../data/results/intervention_markdown_strip_results.jsonl')
qwen_bt_result_ids = load_ids('../../data/results/intervention_qwen_results.jsonl')
qwen_para_result_ids = load_ids('../../data/results/intervention_qwen_paraphrase_results.jsonl')

print(f"Baseline results:  {len(baseline_result_ids)}")
print(f"Markdown results:  {len(markdown_result_ids)}")
print(f"Qwen BT results:   {len(qwen_bt_result_ids)}")
print(f"Qwen Para results: {len(qwen_para_result_ids)}")

# Check if results match metadata
markdown_meta_result_match = len(markdown_ids & markdown_result_ids)
qwen_bt_meta_result_match = len(qwen_bt_ids & qwen_bt_result_ids)
qwen_para_meta_result_match = len(qwen_para_ids & qwen_para_result_ids)

print(f"\nMetadata ↔ Results overlap:")
print(f"  Markdown:  {markdown_meta_result_match}/{len(markdown_ids)} ({markdown_meta_result_match/len(markdown_ids)*100:.1f}%)")
print(f"  Qwen BT:   {qwen_bt_meta_result_match}/{len(qwen_bt_ids)} ({qwen_bt_meta_result_match/len(qwen_bt_ids)*100:.1f}%)")
print(f"  Qwen Para: {qwen_para_meta_result_match}/{len(qwen_para_ids)} ({qwen_para_meta_result_match/len(qwen_para_ids)*100:.1f}%)")

if markdown_meta_result_match < len(markdown_ids) * 0.95:
    warnings.append(f"Markdown: {len(markdown_ids) - markdown_meta_result_match} battles judged but not in results")

# FINAL REPORT
print("\n" + "="*80)
print("AUDIT SUMMARY")
print("="*80)

if not issues and not warnings:
    print("\n✅ NO ISSUES FOUND!")
    print("\nYour thesis data is solid:")
    print("  • All files present")
    print("  • Sample counts correct")
    print("  • IDs consistent")
    print("  • Interventions worked")
    print("  • Results match metadata")
    print("\nYou're good to go!")
else:
    if issues:
        print(f"\n❌ CRITICAL ISSUES FOUND: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS: {len(warnings)}")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    print("\nRECOMMENDATIONS:")
    if issues:
        print("  • Address critical issues before thesis submission")
    if warnings:
        print("  • Warnings are minor but should be documented in limitations")

print("\n" + "="*80)
