
import json
import pandas as pd

def calculate_biased_self_preference(metadata_file, results_file):
    """Calculate biased self-preference for each judge"""
    
    # Load data
    with open(metadata_file, 'r') as f:
        metadata = {json.loads(line)['conversation_id']: json.loads(line) for line in f}
    
    with open(results_file, 'r') as f:
        results = {json.loads(line)['conversation_id']: json.loads(line) for line in f}
    
    # Calculate per judge
    judge_stats = {}
    
    for conv_id, result in results.items():
        if conv_id not in metadata:
            continue
        
        meta = metadata[conv_id]
        judge = result['judge_model']
        human_winner = result.get('original_winner')
        
        if not human_winner or human_winner == 'tie':
            continue
        
        # Extract judge verdict
        judge_response = result['judge_response'].lower()
        if '[[a]]' in judge_response or 'assistant a' in judge_response:
            judge_winner = 'model_a'
        elif '[[b]]' in judge_response or 'assistant b' in judge_response:
            judge_winner = 'model_b'
        else:
            continue
        
        # Check if judge is one of the models
        if judge == meta['model_a_name']:
            judge_position = 'model_a'
        elif judge == meta['model_b_name']:
            judge_position = 'model_b'
        else:
            continue
        
        # Initialize stats
        if judge not in judge_stats:
            judge_stats[judge] = {
                'human_picked_opponent': 0,
                'judge_picked_self': 0
            }
        
        # Count biased self-preference
        if judge_position == 'model_a' and human_winner == 'model_b':
            judge_stats[judge]['human_picked_opponent'] += 1
            if judge_winner == 'model_a':
                judge_stats[judge]['judge_picked_self'] += 1
        elif judge_position == 'model_b' and human_winner == 'model_a':
            judge_stats[judge]['human_picked_opponent'] += 1
            if judge_winner == 'model_b':
                judge_stats[judge]['judge_picked_self'] += 1
    
    return judge_stats

def main():
    print("BASELINE SELF-PREFERENCE BIAS")
    print("-" * 80)
    
    stats = calculate_biased_self_preference(
        'data/raw/judge_samples.jsonl',
        'data/raw/judge_results.jsonl'
    )
    
    # Print results
    total_biased = 0
    total_opportunities = 0
    
    for judge in ['deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022']:
        if judge in stats:
            s = stats[judge]
            rate = (s['judge_picked_self'] / s['human_picked_opponent'] * 100) if s['human_picked_opponent'] > 0 else 0
            
            print(f"\n{judge}:")
            print(f"  Human picked opponent: {s['human_picked_opponent']}")
            print(f"  Judge picked itself: {s['judge_picked_self']}")
            print(f"  Biased self-preference: {rate:.2f}%")
            
            total_biased += s['judge_picked_self']
            total_opportunities += s['human_picked_opponent']
    
    avg_rate = (total_biased / total_opportunities * 100) if total_opportunities > 0 else 0
    print(f"\n{'='*80}")
    print(f"AVERAGE BIASED SELF-PREFERENCE: {avg_rate:.2f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
