import json
from collections import defaultdict

# Load judge samples
with open('data/raw/judge_samples.jsonl', 'r') as f:
    battles = [json.loads(line) for line in f]

print('='*70)
print('POSITION RANDOMIZATION VERIFICATION')
print('='*70)
print()

# For each judge model, check position distribution
judges = ['deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022']

for judge in judges:
    judge_battles = [b for b in battles if b['judge_model'] == judge]
    
    # Count how often judge is in position A vs B
    position_a = sum(1 for b in judge_battles if b['model_a_name'] == judge)
    position_b = sum(1 for b in judge_battles if b['model_b_name'] == judge)
    total = len(judge_battles)
    
    prob_a = position_a / total if total > 0 else 0
    
    print(f'{judge}:')
    print(f'  Total battles: {total}')
    print(f'  Position A: {position_a} ({prob_a:.3f})')
    print(f'  Position B: {position_b} ({1-prob_a:.3f})')
    
    if abs(prob_a - 0.5) < 0.05:
        print(f'  Randomized? YES (within 5% of 0.5)')
    else:
        print(f'  Randomized? NO (more than 5% from 0.5)')
    print()

print('='*70)
print('CONCLUSION')
print('='*70)
print()
print('Your claim: "position of the models is already randomized"')
print('P(Model_A = Pos_1) should be approximately 0.5 for each judge.')
