import json
import re

def strip_all_markdown(text):
    """Remove ALL markdown formatting characters and patterns"""
    if not text:
        return text
    
    # Remove headers (# ## ### etc.) - including those with leading whitespace
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r'^[\-\*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove bullet points (- or * or + at start of line, with optional whitespace)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists (1. 2. etc)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove blockquotes (>)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove ALL asterisks, underscores, and backticks
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = text.replace('`', '')
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines to double
    text = re.sub(r'  +', ' ', text)  # Multiple spaces to single
    text = text.strip()
    
    return text

print("=" * 70)
print("Complete Markdown Removal")
print("=" * 70)

# Process training data
print("\n1. Processing classifier_training_samples.jsonl...")
with open('classifier_training_samples.jsonl', 'r') as f:
    training_data = [json.loads(line) for line in f]

print(f"   Loaded {len(training_data)} training samples")

for item in training_data:
    item['model_a_response'] = strip_all_markdown(item['model_a_response'])
    item['model_b_response'] = strip_all_markdown(item['model_b_response'])

with open('classifier_training_samples_markdown_stripped.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')

print(f"   Done: Saved to classifier_training_samples_markdown_stripped.jsonl")

# Verify training data
with open('classifier_training_samples_markdown_stripped.jsonl', 'r') as f:
    content = f.read()
    print(f"   Verification:")
    print(f"     * count: {content.count('*')}")
    print(f"     # count: {content.count('#')}")
    print(f"     ### count: {content.count('###')}")

# Process test data
print("\n2. Processing judge_samples.jsonl...")
with open('judge_samples.jsonl', 'r') as f:
    test_data = [json.loads(line) for line in f]

print(f"   Loaded {len(test_data)} test samples")

for item in test_data:
    item['model_a_response'] = strip_all_markdown(item['model_a_response'])
    item['model_b_response'] = strip_all_markdown(item['model_b_response'])

with open('intervention_markdown_strip.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print(f"   Done: Saved to intervention_markdown_strip.jsonl")

# Verify test data
with open('intervention_markdown_strip.jsonl', 'r') as f:
    content = f.read()
    print(f"   Verification:")
    print(f"     * count: {content.count('*')}")
    print(f"     # count: {content.count('#')}")
    print(f"     ### count: {content.count('###')}")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print("\nAll markdown formatting removed:")
print("  - Headers (#, ##, ###)")
print("  - Bold/italic (*, _)")
print("  - Bullet points (-, *, +)")
print("  - Numbered lists (1., 2.)")
print("  - Blockquotes (>)")
print("  - Horizontal rules (---, ***)")
print("  - Code blocks (`)")
