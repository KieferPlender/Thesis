
def main():
    print("CLASSIFIER ACCURACY RESULTS")
    print("-" * 80)
    
    results = {
        'Normal Classifier on Baseline': 80.50,
        'Normal Classifier on Markdown-Stripped': 50.47,
        'Markdown-Free Classifier on Markdown-Stripped': 76.57,
    }
    
    print(f"\n{'Test Configuration':<50} {'Accuracy':>10}")
    print("-" * 80)
    
    baseline_acc = results['Normal Classifier on Baseline']
    
    for config, acc in results.items():
        drop = baseline_acc - acc if config != 'Normal Classifier on Baseline' else 0
        drop_str = f"(-{drop:.2f}pp)" if drop > 0 else ""
        print(f"{config:<50} {acc:>9.2f}% {drop_str}")
    
    print("\n" + "="*80)
    print("THE PARADOX")
    print("="*80)
    
    classifier_drop = baseline_acc - results['Normal Classifier on Markdown-Stripped']
    bias_drop = 2.15  # Average bias reduction from markdown intervention
    ratio = classifier_drop / bias_drop
    
    print(f"\nMarkdown removal caused:")
    print(f"  Classifier accuracy drop: {classifier_drop:.2f}pp")
    print(f"  Bias reduction:           {bias_drop:.2f}pp")
    print(f"  Ratio:                    {ratio:.1f}:1")
    print(f"\nInterpretation: Classifier drops {ratio:.1f}x more than bias!")
    print("This shows bias is driven by DEEP patterns, not surface formatting.")

if __name__ == "__main__":
    main()
