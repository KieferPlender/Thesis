import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append('scripts/01_training')
from train_mcgovern_classifier import PosTagExtractor

# Load original classifier
print("Loading classifiers...")
with open('models/mcgovern_classifier.pkl', 'rb') as f:
    original_clf = pickle.load(f)

with open('models/mcgovern_classifier_markdown_free.pkl', 'rb') as f:
    markdown_clf = pickle.load(f)

# Get feature names from the pipeline
print("Extracting feature names...")
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

# Get feature importances
print("Getting feature importances...")
original_importances = original_clf.named_steps['classifier'].feature_importances_
markdown_importances = markdown_clf.named_steps['classifier'].feature_importances_

# Create DataFrames
original_df = pd.DataFrame({
    'feature': feature_names,
    'importance': original_importances
}).sort_values('importance', ascending=False)

markdown_df = pd.DataFrame({
    'feature': feature_names,
    'importance': markdown_importances
}).sort_values('importance', ascending=False)

# Save feature importance files
print("Saving results...")

original_df.to_csv('results/feature_data/feature_importance_baseline.csv', index=False)
markdown_df.to_csv('results/feature_data/feature_importance_markdown.csv', index=False)

print(f'\nDone: Feature importance files created successfully')
print(f'Total features: {len(feature_names)}')
print(f'\nOriginal classifier top 5:')
for i in range(5):
    print(f'  {i+1}. {original_df.iloc[i]["feature"]}: {original_df.iloc[i]["importance"]:.4f}')

print(f'\nMarkdown-free classifier top 5:')
for i in range(5):
    print(f'  {i+1}. {markdown_df.iloc[i]["feature"]}: {markdown_df.iloc[i]["importance"]:.4f}')
