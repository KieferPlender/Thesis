
import pandas as pd

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("  PER-MODEL FEATURE IMPORTANCE")
    print("  Characteristic Features for Each Judge")
    print("="*80)
    
   

    print_section("1. NORMAL CLASSIFIER (with markdown)")
    
    try:
        df_normal = pd.read_csv('/Users/kieferplender/Documents/code scriptie/preserved_outputs/normal_classifier_per_model_features.csv')
        
        # Get unique models
        models = df_normal['owner'].unique()
        
        for model in models:
            model_features = df_normal[df_normal['owner'] == model].head(10)
            
            print(f"\n{model}:")
            print("-" * 75)
            
            for idx, (i, row) in enumerate(model_features.iterrows(), 1):
                feature = row['feature'][:45].ljust(45)
                importance = row['importance']
                category = row['type']
                print(f"  {idx:2d}. [{category:4s}] {feature} ({importance:.6f})")
        
    except Exception as e:
        print(f"Error: {e}")
    
    
    print_section("2. MARKDOWN-FREE CLASSIFIER (without markdown)")
    
    try:
        df_md_free = pd.read_csv('/Users/kieferplender/Documents/code scriptie/preserved_outputs/markdown-free_classifier_per_model_features.csv')
        
        # Get unique models
        models = df_md_free['owner'].unique()
        
        for model in models:
            model_features = df_md_free[df_md_free['owner'] == model].head(10)
            
            print(f"\n{model}:")
            print("-" * 75)
            
            for idx, (i, row) in enumerate(model_features.iterrows(), 1):
                feature = row['feature'][:45].ljust(45)
                importance = row['importance']
                category = row['type']
                print(f"  {idx:2d}. [{category:4s}] {feature} ({importance:.6f})")
        
    except Exception as e:
        print(f"Error: {e}")
    

    print_section("3. KEY INSIGHTS - How Features Shift")
    
    print("DeepSeek-R1:")
    print("  WITH markdown:    Heavy **, ###, code blocks")
    print("  WITHOUT markdown: Technical vocabulary, structured patterns")
    print()
    print("ChatGPT-4o:")
    print("  WITH markdown:    'Let me know if', social phrases")
    print("  WITHOUT markdown: Same phrases persist! (conversational tone)")
    print()
    print("Claude-Haiku:")
    print("  WITH markdown:    Formal language, minimal markdown")
    print("  WITHOUT markdown: Same formal patterns (least affected)")
    print()
    print("Why This Matters:")
    print("  • DeepSeek loses its #1 fingerprint (markdown) when stripped")
    print("  • ChatGPT's conversational style PERSISTS (deep pattern)")
    print("  • Claude barely changes (never relied on markdown)")
    print("  • Explains why bias reduction varies by model!")
    
    print("\n" + "="*80)
    print("  END OF PER-MODEL ANALYSIS")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
