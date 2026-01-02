import argparse
import json
from bias_metrics import (
    load_judge_results,
    calculate_biased_self_preference,
    compare_interventions,
    get_verdict_changes
)


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs intervention bias metrics')
    parser.add_argument('--baseline', type=str, default='judge_results.jsonl',
                       help='Baseline judge results file')
    parser.add_argument('--intervention', type=str, required=True,
                       help='Intervention judge results file')
    parser.add_argument('--metadata', type=str, default='judge_samples.jsonl',
                       help='Metadata file with human judgments')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (optional)')
    parser.add_argument('--judges', type=str, nargs='+', default=None,
                       help='Specific judges to analyze (default: all)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("INTERVENTION COMPARISON ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading baseline: {args.baseline}")
    baseline_df = load_judge_results(args.baseline, args.metadata)
    
    print(f"Loading intervention: {args.intervention}")
    intervention_df = load_judge_results(args.intervention, args.metadata)
    
    print(f"\nBaseline samples: {len(baseline_df)}")
    print(f"Intervention samples: {len(intervention_df)}")
    
    # Determine which judges to analyze
    if args.judges:
        judges = args.judges
    else:
        judges = baseline_df['judge_model'].unique()
    
    print(f"\nAnalyzing judges: {list(judges)}")
    
    # Run comparison for each judge
    results = {}
    
    for judge in judges:
        print(f"\n{'='*70}")
        print(f"JUDGE: {judge}")
        print(f"{'='*70}")
        
        # Get comparison metrics
        comparison = compare_interventions(baseline_df, intervention_df, judge)
        
        # Get verdict changes
        changes = get_verdict_changes(baseline_df, intervention_df, judge)
        
        # Print results
        base = comparison['baseline']
        inter = comparison['intervention']
        
        print(f"\nBiased Self-Preference:")
        print(f"  Baseline:     {base['bias_rate']:.2f}% ({base['biased_picks']}/{base['opportunities']})")
        print(f"  Intervention: {inter['bias_rate']:.2f}% ({inter['biased_picks']}/{inter['opportunities']})")
        print(f"  Reduction:    {comparison['bias_reduction_pp']:.2f} percentage points")
        print(f"  Relative:     {comparison['bias_reduction_relative']:.1f}%")
        
        print(f"\nVerdict Changes:")
        print(f"  Total battles compared: {changes['total_battles']}")
        print(f"  Verdicts changed:       {changes['verdicts_changed']} ({changes['verdicts_changed']/changes['total_battles']*100:.1f}%)")
        print(f"  Bias reduced (count):   {changes['bias_reduced_count']}")
        print(f"  Bias increased (count): {changes['bias_increased_count']}")
        print(f"  Net improvement:        {changes['net_improvement']}")
        
        # Store for output
        results[judge] = {
            'comparison': comparison,
            'changes': changes
        }
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nInterpretation:")
    print("- Biased Self-Preference: % of times judge picked itself when human picked opponent")
    print("- Reduction: Lower is better (indicates less bias)")
    print("- Net improvement: Positive means intervention reduced bias")


if __name__ == "__main__":
    main()
