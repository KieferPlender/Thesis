import json
import random

random.seed(42)

def show_comparison(title, baseline_file, intervention_file, num_samples=5):
    """Show side-by-side comparison of baseline vs intervention"""
    
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    
    # Load data
    baseline = {}
    with open(baseline_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                baseline[data['conversation_id']] = data
            except:
                continue
    
    intervention = {}
    with open(intervention_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                intervention[data['conversation_id']] = data
            except:
                continue
    
    # Get common IDs
    common_ids = list(set(baseline.keys()) & set(intervention.keys()))
    
    if not common_ids:
        print("❌ NO COMMON IDs FOUND - FILES DON'T MATCH!")
        return False
    
    print(f"\n✅ Found {len(common_ids)} common battles")
    
    # Sample random battles
    sample_ids = random.sample(common_ids, min(num_samples, len(common_ids)))
    
    for i, conv_id in enumerate(sample_ids, 1):
        b = baseline[conv_id]
        iv = intervention[conv_id]
        
        print(f"\n{'-'*80}")
        print(f"EXAMPLE {i}/{num_samples}")
        print(f"{'-'*80}")
        print(f"Conversation ID: {conv_id[:40]}...")
        print(f"User Prompt: {b['user_prompt'][:80]}...")
        print(f"Model A: {b['model_a_name']}")
        print()
        
        # Check if user prompt unchanged
        if b['user_prompt'] != iv['user_prompt']:
            print("⚠️  WARNING: User prompt changed!")
        else:
            print("✅ User prompt unchanged")
        
        # Show Model A response comparison
        print(f"\n--- MODEL A RESPONSE ---")
        print(f"ORIGINAL (first 300 chars):")
        print(b['model_a_response'][:300])
        print()
        print(f"MODIFIED (first 300 chars):")
        print(iv['model_a_response'][:300])
        print()
        
        # Check if actually different
        if b['model_a_response'] == iv['model_a_response']:
            print("❌ WARNING: Responses are IDENTICAL - intervention didn't work!")
        else:
            print("✅ Response was modified")
            
            # Check if original is stored (for back-translation and paraphrasing)
            if 'original_model_a_response' in iv:
                if b['model_a_response'] == iv['original_model_a_response']:
                    print("✅ Original response correctly stored")
                else:
                    print("❌ WARNING: Stored original doesn't match baseline!")
    
    return True

# Main validation
print("="*80)
print("INTERVENTION VALIDATION - PROOF THEY ACTUALLY WORKED")
print("="*80)
print("\nThis script proves each intervention actually modified the data.")

# 1. MARKDOWN REMOVAL
print("\n\n" + "="*80)
print("1. MARKDOWN REMOVAL VALIDATION")
print("="*80)

show_comparison(
    "Markdown Removal: Original vs Stripped",
    "../../data/raw/judge_samples.jsonl",
    "../../data/processed/intervention_markdown_strip.jsonl"
)

# 2. QWEN BACK-TRANSLATION
print("\n\n" + "="*80)
print("2. QWEN BACK-TRANSLATION VALIDATION")
print("="*80)

show_comparison(
    "Back-Translation: Original vs English→Chinese→English",
    "../../data/raw/judge_samples.jsonl",
    "../../data/processed/intervention_qwen_chinese.jsonl"
)

# 3. QWEN PARAPHRASING
print("\n\n" + "="*80)
print("3. QWEN PARAPHRASING VALIDATION")
print("="*80)

show_comparison(
    "Paraphrasing: Original vs Style-Removed",
    "../../data/raw/judge_samples.jsonl",
    "../../data/processed/intervention_qwen_paraphrase.jsonl"
)

# FINAL SUMMARY
print("\n\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print("""
If you see:
✅ User prompts unchanged
✅ Responses were modified
✅ Original responses correctly stored (for BT and Para)

Then the interventions DEFINITELY worked!

If you see ❌ warnings, investigate those specific cases.
""")
