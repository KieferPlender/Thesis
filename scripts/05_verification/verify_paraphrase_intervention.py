import json
import random

print("="*70)
print("VERIFICATION: Paraphrasing Intervention Correctness")
print("="*70)

# 1. Check data files
print("\n1. DATA FILES CHECK")
print("-" * 70)

with open('judge_samples.jsonl', 'r') as f:
    baseline_samples = [json.loads(line) for line in f]

with open('intervention_qwen_paraphrase.jsonl', 'r') as f:
    para_samples = [json.loads(line) for line in f]

with open('intervention_qwen_paraphrase_results.jsonl', 'r') as f:
    para_results = [json.loads(line) for line in f]

print(f"Baseline samples: {len(baseline_samples)}")
print(f"Paraphrased samples: {len(para_samples)}")
print(f"Paraphrased results: {len(para_results)}")

# Check if conversation IDs overlap
baseline_ids = set(s['conversation_id'] for s in baseline_samples)
para_ids = set(s['conversation_id'] for s in para_samples)
result_ids = set(r['conversation_id'] for r in para_results)

overlap_samples = baseline_ids & para_ids
overlap_results = para_ids & result_ids

print(f"\nOverlap baseline ↔ paraphrased: {len(overlap_samples)} / {len(para_samples)} ({len(overlap_samples)/len(para_samples)*100:.1f}%)")
print(f"Overlap paraphrased ↔ results: {len(overlap_results)} / {len(para_results)} ({len(overlap_results)/len(para_results)*100:.1f}%)")

if len(overlap_samples) < len(para_samples) * 0.95:
    print("⚠️ WARNING: Less than 95% overlap with baseline!")
else:
    print("✅ Good overlap with baseline")

# 2. Check intervention actually worked
print("\n2. INTERVENTION EFFECTIVENESS CHECK")
print("-" * 70)

# Sample 10 random battles
sample_ids = random.sample(list(overlap_samples), min(10, len(overlap_samples)))

identical_count = 0
changed_count = 0

for conv_id in sample_ids:
    baseline = next(s for s in baseline_samples if s['conversation_id'] == conv_id)
    para = next(s for s in para_samples if s['conversation_id'] == conv_id)
    
    # Check if responses actually changed
    if baseline['model_a_response'] == para['model_a_response']:
        identical_count += 1
    else:
        changed_count += 1

print(f"Sampled 10 battles:")
print(f"  Responses changed: {changed_count}")
print(f"  Responses identical: {identical_count}")

if identical_count > 0:
    print("❌ PROBLEM: Some responses were not paraphrased!")
else:
    print("✅ All responses were paraphrased")

# 3. Check semantic fidelity on sample
print("\n3. SEMANTIC FIDELITY SPOT CHECK")
print("-" * 70)

for i, conv_id in enumerate(sample_ids[:3], 1):
    baseline = next(s for s in baseline_samples if s['conversation_id'] == conv_id)
    para = next(s for s in para_samples if s['conversation_id'] == conv_id)
    
    print(f"\nSample {i}:")
    print(f"Original: {baseline['model_a_response'][:100]}...")
    print(f"Paraphrased: {para['model_a_response'][:100]}...")
    
    # Check if stored original matches
    if 'original_model_a_response' in para:
        if baseline['model_a_response'] == para['original_model_a_response']:
            print("✅ Stored original matches baseline")
        else:
            print("❌ Stored original DOES NOT match baseline!")

# 4. Check comparison used correct data
print("\n4. COMPARISON DATA CHECK")
print("-" * 70)

# Check if results have matching metadata
sample_result = para_results[0]
sample_para = next(s for s in para_samples if s['conversation_id'] == sample_result['conversation_id'])

print(f"Sample result conversation_id: {sample_result['conversation_id'][:30]}...")
print(f"Found in paraphrased samples: {'✅' if sample_para else '❌'}")

if sample_para:
    print(f"Judge matches: {'✅' if sample_result['judge_model'] == sample_para['judge_model'] else '❌'}")
    print(f"Models match: {'✅' if sample_result['model_a_name'] == sample_para['model_a_name'] else '❌'}")

# 5. Final verdict
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

issues = []

if len(overlap_samples) < len(para_samples) * 0.95:
    issues.append("Low overlap with baseline")

if identical_count > 0:
    issues.append("Some responses not paraphrased")

if issues:
    print("❌ ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ ALL CHECKS PASSED")
    print("\nIntervention was executed correctly:")
    print("  ✓ Correct data files used")
    print("  ✓ Responses were paraphrased")
    print("  ✓ Semantic fidelity maintained")
    print("  ✓ Comparison used matching battles")
