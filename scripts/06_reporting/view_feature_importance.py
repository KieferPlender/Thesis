import pandas as pd

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    
    print_section("1. TOP 20 FEATURES (XGBoost Importance)")
    
    try:
        df = pd.read_csv('../../preserved_outputs/feature_importance.csv')
        
        print("Rank  Feature                              Importance  Type")
        print("-" * 75)
        
        for i, row in df.head(20).iterrows():
            feature = row['feature'][:35].ljust(35)
            importance = f"{row['importance']:.6f}"
            category = row['category']
            print(f"{i+1:3d}   {feature}  {importance}  {category}")
        
        print()
        print("Feature Type Distribution (Top 20):")
        top20_counts = df.head(20)['category'].value_counts()
        for cat, count in top20_counts.items():
            print(f"  {cat}: {count} features ({count/20*100:.0f}%)")
            
    except Exception as e:
        print(f"Error loading feature_importance.csv: {e}")
    
    
    print_section("2. PER-MODEL CHARACTERISTIC FEATURES")
    
    try:
        df_model = pd.read_csv('../../preserved_outputs/normal_classifier_per_model_features.csv')
        
        models = df_model['model'].unique()
        
        for model in models:
            model_features = df_model[df_model['model'] == model].head(10)
            
            print(f"\n{model}:")
            print("-" * 75)
            
            for i, row in model_features.iterrows():
                feature = row['feature'][:50]
                importance = row['importance']
                category = row['category']
                print(f"  {i+1:2d}. [{category:4s}] {feature} ({importance:.6f})")
        
    except Exception as e:
        print(f"Error loading per-model features: {e}")
    
    print_section("3. KEY INSIGHTS")
    
    print("DeepSeek-R1 Fingerprints:")
    print("  • Heavy markdown: **, ##, ###")
    print("  • Technical formatting: code blocks, lists")
    print("  • Structured organization")
    print()
    print("ChatGPT-4o Fingerprints:")
    print("  • Social phrases: 'Let me know if you', 'Hope this helps'")
    print("  • Hedging: 'Would you like me to elaborate'")
    print("  • Conversational tone")
    print()
    print("Claude-Haiku Fingerprints:")
    print("  • Formal vocabulary")
    print("  • Complete sentence structure")
    print("  • Minimal markdown")
    print()
    print("Why Interventions Failed:")
    print("  • Markdown removal: Only targets CHAR features")
    print("  • Back-translation: Changes WORD but not POS/structure")
    print("  • Paraphrasing: Removes surface but not deep patterns")
    
    print("\n" + "="*80)
    print("  END OF FEATURE ANALYSIS")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
