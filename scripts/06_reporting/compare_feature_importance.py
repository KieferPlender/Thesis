
import pandas as pd

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("  FEATURE IMPORTANCE COMPARISON")
    print("  Normal Classifier vs Markdown-Free Classifier")
    print("="*80)
    
    # ========================================================================
    # 1. NORMAL CLASSIFIER (trained on baseline with markdown)
    # ========================================================================
    print_section("1. NORMAL CLASSIFIER - Top 20 Features")
    
    try:
        df_normal = pd.read_csv('/Users/kieferplender/Documents/code scriptie/preserved_outputs/feature_importance.csv')
        
        print("Rank  Feature                              Importance  Type")
        print("-" * 75)
        
        for i, row in df_normal.head(20).iterrows():
            feature = row['feature'][:35].ljust(35)
            importance = f"{row['importance']:.6f}"
            category = row['category']
            print(f"{i+1:3d}   {feature}  {importance}  {category}")
        
        print()
        print("Feature Type Distribution:")
        top20_counts = df_normal.head(20)['category'].value_counts()
        for cat, count in top20_counts.items():
            print(f"  {cat}: {count} features ({count/20*100:.0f}%)")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # ========================================================================
    # 2. MARKDOWN-FREE CLASSIFIER (trained on markdown-stripped)
    # ========================================================================
    print_section("2. MARKDOWN-FREE CLASSIFIER - Top 20 Features")
    
    try:
        df_md_free = pd.read_csv('/Users/kieferplender/Documents/code scriptie/preserved_outputs/markdown_free_feature_importance.csv')
        
        print("Rank  Feature                              Importance  Type")
        print("-" * 75)
        
        for i, row in df_md_free.head(20).iterrows():
            feature = row['feature'][:35].ljust(35)
            importance = f"{row['importance']:.6f}"
            category = row['category']
            print(f"{i+1:3d}   {feature}  {importance}  {category}")
        
        print()
        print("Feature Type Distribution:")
        top20_counts = df_md_free.head(20)['category'].value_counts()
        for cat, count in top20_counts.items():
            print(f"  {cat}: {count} features ({count/20*100:.0f}%)")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # ========================================================================
    # 3. KEY DIFFERENCES
    # ========================================================================
    print_section("3. KEY DIFFERENCES")
    
    print("Normal Classifier (with markdown):")
    print("  • Dominated by CHAR features (80%) - markdown artifacts")
    print("  • Top feature: '* **' (bold markdown)")
    print("  • Accuracy: 80.50%")
    print()
    print("Markdown-Free Classifier (without markdown):")
    print("  • Shifts to WORD and POS features")
    print("  • No markdown artifacts in top features")
    print("  • Accuracy: 76.57% (still high!)")
    print()
    print("The Paradox:")
    print("  • Removing markdown breaks normal classifier (50.47%)")
    print("  • But retrained classifier adapts (76.57%)")
    print("  • Proves: Stylistic fingerprints persist beyond markdown")
    print("  • Judges recognize these deeper patterns (bias: -2.43pp only)")
    
    print("\n" + "="*80)
    print("  END OF COMPARISON")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
