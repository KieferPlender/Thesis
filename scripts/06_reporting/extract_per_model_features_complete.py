import pickle
import pandas as pd
import numpy as np
import sys
import os

# Add the training scripts directory to path
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts', '01_training'))

# Import the PosTagExtractor class needed for unpickling
from train_mcgovern_classifier import PosTagExtractor

print("="*80)
print("PER-MODEL FEATURE IMPORTANCE ANALYSIS")
print("="*80)

print("\n[1/7] Loading classifiers...")
# Load classifiers
with open('models/mcgovern_classifier.pkl', 'rb') as f:
    original_clf = pickle.load(f)

with open('models/mcgovern_classifier_markdown_free.pkl', 'rb') as f:
    markdown_clf = pickle.load(f)
print("Done: Classifiers loaded")

print("\n[2/7] Loading test data from JSONL...")
import json

# Load classifier training samples to get test data
data = []
with open('data/raw/classifier_training_samples.jsonl', 'r') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except:
            pass

# Extract responses and model labels (same logic as training script)
responses = []
models = []

for row in data:
    # Model A response (only if it's a judge model)
    if row['model_a_name'] == row['judge_model']:
        responses.append(row['model_a_response'])
        models.append(row['model_a_name'])
    
    # Model B response (only if it's a judge model)
    if row['model_b_name'] == row['judge_model']:
        responses.append(row['model_b_response'])
        models.append(row['model_b_name'])

# Create DataFrame
test_df = pd.DataFrame({
    'response': responses,
    'model': models
})

# Load markdown-free versions
data_md_free = []
with open('data/raw/classifier_training_samples_markdown_stripped.jsonl', 'r') as f:
    for line in f:
        try:
            data_md_free.append(json.loads(line))
        except:
            pass

# Extract markdown-free responses
responses_md_free = []
for row in data_md_free:
    if row['model_a_name'] == row['judge_model']:
        responses_md_free.append(row['model_a_response'])
    if row['model_b_name'] == row['judge_model']:
        responses_md_free.append(row['model_b_response'])

test_df['response_markdown_free'] = responses_md_free[:len(test_df)]

print(f"Done: Loaded {len(test_df)} responses")

print("\n[3/7] Extracting feature names...")
feature_union = original_clf.named_steps['features']
feature_names = []

# Word n-grams
word_vectorizer = feature_union.transformer_list[0][1]
feature_names.extend([f'word_{name}' for name in word_vectorizer.get_feature_names_out()])

# Char n-grams
char_vectorizer = feature_union.transformer_list[1][1]
feature_names.extend([f'char_{name}' for name in char_vectorizer.get_feature_names_out()])

# POS n-grams
pos_vectorizer = feature_union.transformer_list[2][1].named_steps['pos_vectorizer']
feature_names.extend([f'pos_{name}' for name in pos_vectorizer.get_feature_names_out()])

print(f"Done: Total features: {len(feature_names)}")

# Get feature importances
original_importances = original_clf.named_steps['classifier'].feature_importances_
markdown_importances = markdown_clf.named_steps['classifier'].feature_importances_

# Transform test data to get feature values
print("\n[4/7] Transforming test data with Original Classifier...")
print("(This will take a few minutes for POS tagging - please be patient)")
X_test_original = original_clf.named_steps['features'].transform(test_df['response'])

print("\n[5/7] Transforming test data with Markdown-Free Classifier...")
X_test_markdown = markdown_clf.named_steps['features'].transform(test_df['response_markdown_free'])

# Convert to dense arrays
print("\n[6/7] Converting to dense arrays...")
X_test_original_dense = X_test_original.toarray()
X_test_markdown_dense = X_test_markdown.toarray()

# Get unique models
unique_models = test_df['model'].unique()
print(f"Done: Models found: {list(unique_models)}")

def analyze_per_model_features(X_features, importances, feature_names, test_df, classifier_name):
    """Analyze which features are most distinctive for each model"""
    
    print(f"\nAnalyzing {classifier_name} classifier features...")
    results = []
    unique_models = test_df['model'].unique()
    
    # For each feature
    for feat_idx, feat_name in enumerate(feature_names):
        if feat_idx % 200 == 0:
            print(f"  Processing feature {feat_idx}/{len(feature_names)}...")
        
        feat_importance = importances[feat_idx]
        
        # Calculate mean usage per model
        model_means = {}
        for model in unique_models:
            model_mask = test_df['model'] == model
            model_mean = X_features[model_mask, feat_idx].mean()
            model_means[model] = model_mean
        
        # Find which model uses this feature most
        max_model = max(model_means, key=model_means.get)
        max_value = model_means[max_model]
        
        # Calculate distinctiveness ratio
        other_models_mean = np.mean([v for k, v in model_means.items() if k != max_model])
        ratio = max_value / other_models_mean if other_models_mean > 0 else 0
        
        # Determine feature type
        if feat_name.startswith('word_'):
            feat_type = 'WORD'
            feat_display = feat_name[5:]
        elif feat_name.startswith('char_'):
            feat_type = 'CHAR'
            feat_display = feat_name[5:]
        else:
            feat_type = 'POS'
            feat_display = feat_name[4:]
        
        results.append({
            'feature': feat_display,
            'type': feat_type,
            'importance': feat_importance,
            'owner': max_model,
            'owner_value': max_value,
            'ratio': ratio,
            **{f'{model}_mean': model_means[model] for model in unique_models}
        })
    
    # Convert to DataFrame and sort by importance
    df = pd.DataFrame(results)
    df = df.sort_values('importance', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    cols = ['rank', 'feature', 'type', 'importance', 'owner', 'owner_value', 'ratio']
    cols.extend([f'{model}_mean' for model in unique_models])
    df = df[cols]
    
    return df

print("\n[7/7] Analyzing per-model features...")
original_results = analyze_per_model_features(
    X_test_original_dense, 
    original_importances, 
    feature_names, 
    test_df,
    'Original'
)

markdown_results = analyze_per_model_features(
    X_test_markdown_dense, 
    markdown_importances, 
    feature_names, 
    test_df,
    'Markdown-Free'
)

# Save complete results
print("\nSaving complete results...")
original_results.to_csv('results/feature_data/baseline_classifier_per_model_features.csv', index=False)
markdown_results.to_csv('results/feature_data/markdown_free_classifier_per_model_features.csv', index=False)

print(f"\nDone: Saved complete feature analysis:")
print(f"  - results/feature_data/baseline_classifier_per_model_features.csv ({len(original_results)} features)")
print(f"  - results/feature_data/markdown_free_classifier_per_model_features.csv ({len(markdown_results)} features)")

# Show summary
print("\n" + "="*80)
print("SUMMARY: Features per Model Owner")
print("="*80)

for name, df in [('Original Classifier', original_results), ('Markdown-Free Classifier', markdown_results)]:
    print(f"\n{name}:")
    owner_counts = df['owner'].value_counts()
    for model, count in owner_counts.items():
        print(f"  {model}: {count} features")

print("\n" + "="*80)
