

def main():
    print("INTERVENTION COMPARISON - BIASED SELF-PREFERENCE")
    print("-" * 80)
    
    # Data from your results
    results = {
        'Baseline': {
            'deepseek-r1-0528': 78.62,
            'chatgpt-4o-latest-20250326': 52.27,
            'claude-3-5-haiku-20241022': 20.67,
        },
        'Markdown Removal': {
            'deepseek-r1-0528': 75.00,
            'chatgpt-4o-latest-20250326': 48.54,
            'claude-3-5-haiku-20241022': 21.57,
        },
        'Qwen Back-Translation': {
            'deepseek-r1-0528': 73.33,
            'chatgpt-4o-latest-20250326': 45.93,
            'claude-3-5-haiku-20241022': 21.05,
        },
        'Qwen Paraphrasing': {
            'deepseek-r1-0528': 73.91,
            'chatgpt-4o-latest-20250326': 49.63,
            'claude-3-5-haiku-20241022': 22.94,
        }
    }
    
    judges = ['deepseek-r1-0528', 'chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022']
    judge_names = ['DeepSeek-R1', 'ChatGPT-4o', 'Claude-Haiku']
    
    # Print table header
    print(f"\n{'Judge':<20} {'Baseline':>12} {'Markdown':>12} {'Qwen BT':>12} {'Qwen Para':>12}")
    print("-" * 80)
    
    # Print per-judge results
    for judge, name in zip(judges, judge_names):
        baseline = results['Baseline'][judge]
        markdown = results['Markdown Removal'][judge]
        qwen_bt = results['Qwen Back-Translation'][judge]
        qwen_para = results['Qwen Paraphrasing'][judge]
        
        print(f"{name:<20} {baseline:>11.2f}% {markdown:>11.2f}% {qwen_bt:>11.2f}% {qwen_para:>11.2f}%")
    
    print("-" * 80)
    
    # Calculate averages
    avg_baseline = sum(results['Baseline'].values()) / 3
    avg_markdown = sum(results['Markdown Removal'].values()) / 3
    avg_qwen_bt = sum(results['Qwen Back-Translation'].values()) / 3
    avg_qwen_para = sum(results['Qwen Paraphrasing'].values()) / 3
    
    print(f"{'AVERAGE':<20} {avg_baseline:>11.2f}% {avg_markdown:>11.2f}% {avg_qwen_bt:>11.2f}% {avg_qwen_para:>11.2f}%")
    
    # Print reductions
    print("\n" + "="*80)
    print("BIAS REDUCTION (percentage points)")
    print("="*80)
    
    print(f"\n{'Judge':<20} {'Markdown':>12} {'Qwen BT':>12} {'Qwen Para':>12}")
    print("-" * 80)
    
    for judge, name in zip(judges, judge_names):
        baseline = results['Baseline'][judge]
        markdown_red = baseline - results['Markdown Removal'][judge]
        qwen_bt_red = baseline - results['Qwen Back-Translation'][judge]
        qwen_para_red = baseline - results['Qwen Paraphrasing'][judge]
        
        print(f"{name:<20} {markdown_red:>11.2f}pp {qwen_bt_red:>11.2f}pp {qwen_para_red:>11.2f}pp")
    
    print("-" * 80)
    
    avg_markdown_red = avg_baseline - avg_markdown
    avg_qwen_bt_red = avg_baseline - avg_qwen_bt
    avg_qwen_para_red = avg_baseline - avg_qwen_para
    
    print(f"{'AVERAGE':<20} {avg_markdown_red:>11.2f}pp {avg_qwen_bt_red:>11.2f}pp {avg_qwen_para_red:>11.2f}pp")
    
    # Ranking
    print("\n" + "="*80)
    print("RANKING (by average reduction)")
    print("="*80)
    print(f"1. Qwen Back-Translation: -{avg_qwen_bt_red:.2f}pp ⭐")
    print(f"2. Markdown Removal:      -{avg_markdown_red:.2f}pp")
    print(f"3. Qwen Paraphrasing:     -{avg_qwen_para_red:.2f}pp")

if __name__ == "__main__":
    main()
