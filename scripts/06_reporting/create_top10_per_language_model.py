import pandas as pd

# Load both classifier datasets
normal_df = pd.read_csv('results/feature_data/baseline_classifier_per_model_features.csv')
markdown_free_df = pd.read_csv('results/feature_data/markdown_free_classifier_per_model_features.csv')

# Get unique language models
models = ['chatgpt-4o-latest-20250326', 'claude-3-5-haiku-20241022', 'deepseek-r1-0528']
model_display_names = {
    'chatgpt-4o-latest-20250326': 'ChatGPT-4o',
    'claude-3-5-haiku-20241022': 'Claude-3.5-Haiku',
    'deepseek-r1-0528': 'DeepSeek-R1'
}

# Create a list to hold all results
all_results = []

# Process each classifier type
for classifier_name, df in [('Original Classifier', normal_df), ('Markdown-Free Classifier', markdown_free_df)]:
    # For each language model
    for model_id in models:
        # Filter features owned by this model and get top 10
        model_features = df[df['owner'] == model_id].head(10).copy()
        
        # Add columns for clarity
        model_features['classifier'] = classifier_name
        model_features['language_model'] = model_display_names[model_id]
        
        # Add rank within this model
        model_features['rank_in_model'] = range(1, len(model_features) + 1)
        
        all_results.append(model_features)

# Combine all results
combined_df = pd.concat(all_results, ignore_index=True)

# Select and reorder columns for readability
output_df = combined_df[[
    'classifier',
    'language_model', 
    'rank_in_model',
    'feature',
    'type',
    'importance',
    'owner_value',
    'ratio'
]]

# Rename columns for clarity
output_df = output_df.rename(columns={
    'rank_in_model': 'rank',
    'owner_value': 'model_mean_value',
    'ratio': 'distinctiveness_ratio'
})

# Save to CSV
output_file = 'results/feature_data/feature_importance_top10_per_language_model.csv'
output_df.to_csv(output_file, index=False)

print(f'Done: {output_file}')
print(f'\nTotal rows: {len(output_df)}')
print(f'Models: {len(models)}')
print(f'Classifiers: 2 (Original + Markdown-Free)')
print(f'Features per model per classifier: 10')
print(f'\nBreakdown:')

for classifier_name in ['Original Classifier', 'Markdown-Free Classifier']:
    print(f'\n{classifier_name}:')
    for model_name in model_display_names.values():
        count = len(output_df[(output_df['classifier'] == classifier_name) & 
                              (output_df['language_model'] == model_name)])
        print(f'  {model_name}: {count} features')

print(f'\n\nPreview (first 15 rows):')
print(output_df.head(15).to_string(index=False))
