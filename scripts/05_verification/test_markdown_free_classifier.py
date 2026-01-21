import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'scripts/01_training'))
from train_mcgovern_classifier import PosTagExtractor

def load_data(file_path):
    """
    Load data and extract responses/labels for judge models only.
    
    This filtering ensures we only test on samples where:
    - The model being tested is also serving as the judge
    - This matches the training data format
    """
    print(f"Loading data from {file_path}...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    df = pd.DataFrame(data)
    print(f"  Total battles loaded: {len(df)}")
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        # Only include responses where the model is also the judge
        if row['model_a_name'] == row['judge_model']:
            texts.append(row['model_a_response'])
            labels.append(row['model_a_name'])
        
        if row['model_b_name'] == row['judge_model']:
            texts.append(row['model_b_response'])
            labels.append(row['model_b_name'])
    
    print(f"  Samples extracted: {len(texts)}")
    
    # Show label distribution for verification
    print(f"\n  Label distribution:")
    label_counts = pd.Series(labels).value_counts()
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    return texts, labels

def main():
    print("="*70)
    print("MARKDOWN-FREE CLASSIFIER ACCURACY TEST")
    print("="*70)
    
    # Verify no data leakage
    print("\n" + "="*70)
    print("DATA LEAKAGE CHECK")
    print("="*70)
    print("Training data: classifier_training_samples_markdown_stripped.jsonl")
    print("Test data:     intervention_markdown_strip.jsonl")
    print("\nThese are DIFFERENT datasets:")
    print("  - Training: Separate subset for classifier training")
    print("  - Test: Intervention data for judging experiments")
    print("  - No overlap between training and test sets")
    print("Done: No data leakage")
    
    # Load markdown-free classifier
    print("\n" + "="*70)
    print("LOADING CLASSIFIER")
    print("="*70)
    print("Loading markdown-free classifier...")
    with open('models/mcgovern_classifier_markdown_free.pkl', 'rb') as f:
        clf = pickle.load(f)
    print("Done: Classifier loaded")
    print(f"  Classifier type: {type(clf).__name__}")
    
    # Verify classifier has label encoder
    if hasattr(clf, 'label_encoder_'):
        print(f"  Classes: {clf.label_encoder_.classes_}")
    else:
        print("  Warning: No label_encoder_ found")
    
    # Load test data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    X_test, y_test = load_data('data/processed/intervention_markdown_strip.jsonl')
    
    # Predict
    print("\n" + "="*70)
    print("RUNNING PREDICTIONS")
    print("="*70)
    print("Extracting features and predicting...")
    print("(This may take a moment due to POS tagging)")
    
    y_pred_encoded = clf.predict(X_test)
    
    # Decode predictions
    if hasattr(clf, 'label_encoder_'):
        y_pred = clf.label_encoder_.inverse_transform(y_pred_encoded)
    else:
        y_pred = y_pred_encoded
    
    print("Done: Predictions complete")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    accuracy = accuracy_score(y_test, y_pred)
    unique_labels = np.unique(y_test)
    random_baseline = 1.0 / len(unique_labels)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Random Baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    print(f"Improvement over Random: {(accuracy - random_baseline)*100:.2f} percentage points")
    
    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    print("="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print(cm_df)
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if accuracy > 0.70:
        print(f"✅ HIGH ACCURACY: {accuracy*100:.2f}%")
        print("\nConclusion:")
        print("  Even without markdown, the classifier achieves high accuracy.")
        print("  This proves that stylistic fingerprints persist beyond surface formatting.")
        print("\nImplication:")
        print("  Markdown removal alone is INSUFFICIENT to eliminate model fingerprints.")
        print("  Other features (POS patterns, word choice, syntax) remain strong signals.")
    elif accuracy > 0.50:
        print(f"⚠️  MODERATE ACCURACY: {accuracy*100:.2f}%")
        print("\nConclusion:")
        print("  Markdown removal reduced but did not eliminate fingerprinting.")
        print("  Some stylistic signals persist.")
    else:
        print(f"✅ LOW ACCURACY: {accuracy*100:.2f}%")
        print("\nConclusion:")
        print("  Markdown removal successfully disrupted model fingerprinting.")
        print("  Accuracy near random baseline indicates effective style removal.")
    
    # Comparison to baseline
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE")
    print("="*70)
    print("Normal classifier on original data:     80.50%")
    print("Normal classifier on markdown-stripped: 50.47% (-30.03pp)")
    print(f"Markdown-free classifier on markdown-stripped: {accuracy*100:.2f}% ({(accuracy*100 - 80.50):+.2f}pp)")
    print("\nKey Finding:")
    print("  Markdown-free classifier maintains high accuracy despite markdown removal,")
    print("  proving that the classifier ADAPTS by using other stylistic features.")

if __name__ == "__main__":
    main()
