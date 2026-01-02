import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../01_training'))

import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from train_mcgovern_classifier import PosTagExtractor

def load_intervention_data(file_path):
    """Load intervention data and extract samples"""
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    df = pd.DataFrame(data)
    
    # Extract samples where model is judge
    texts = []
    labels = []
    for _, row in df.iterrows():
        if row['model_a_name'] == row['judge_model']:
            texts.append(row['model_a_response'])
            labels.append(row['model_a_name'])
        if row['model_b_name'] == row['judge_model']:
            texts.append(row['model_b_response'])
            labels.append(row['model_b_name'])
    
    return texts, labels

def main():
    # Load normal classifier
    print('Loading normal classifier (trained on baseline)...')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '../../models/mcgovern_classifier.pkl')
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    # Test on all interventions
    data_dir = os.path.join(script_dir, '../../data')
    interventions = [
        ('Baseline', os.path.join(data_dir, 'raw/judge_samples.jsonl')),
        ('Markdown Removal', os.path.join(data_dir, 'processed/intervention_markdown_strip.jsonl')),
        ('Qwen Back-Translation', os.path.join(data_dir, 'processed/intervention_qwen_chinese.jsonl')),
        ('Qwen Paraphrasing', os.path.join(data_dir, 'processed/intervention_qwen_paraphrase.jsonl'))
    ]
    
    print('\\n' + '='*70)
    print('NORMAL CLASSIFIER ACCURACY ON ALL INTERVENTIONS')
    print('='*70)
    print('Testing if interventions successfully removed stylistic fingerprints\\n')
    
    results = []
    
    for name, file_path in interventions:
        try:
            print(f'Testing: {name}...')
            texts, labels = load_intervention_data(file_path)
            
            # Predict
            y_pred = clf.predict(texts)
            y_pred_decoded = clf.label_encoder_.inverse_transform(y_pred)
            
            # Calculate accuracy
            accuracy = accuracy_score(labels, y_pred_decoded)
            
            results.append((name, len(texts), accuracy))
            print(f'  Samples: {len(texts)}')
            print(f'  Accuracy: {accuracy*100:.2f}%\\n')
            
        except Exception as e:
            print(f'  ERROR: {e}\\n')
            results.append((name, 0, 0.0))
    
    # Summary
    print('='*70)
    print('SUMMARY')
    print('='*70)
    
    baseline_acc = results[0][2] if results else 0
    
    for name, samples, acc in results:
        if name == 'Baseline':
            print(f'{name:25} {acc*100:6.2f}%  (baseline)')
        else:
            drop = (baseline_acc - acc) * 100
            print(f'{name:25} {acc*100:6.2f}%  ({drop:+.2f}pp)')
    
    print('\\n' + '='*70)
    print('INTERPRETATION')
    print('='*70)
    print('Large drop (>20pp): Style successfully removed')
    print('Small drop (<10pp): Style persists despite intervention')
    print('No drop (0pp): Intervention had no effect on detectability')

if __name__ == '__main__':
    main()
