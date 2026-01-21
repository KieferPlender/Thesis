import pandas as pd
import json
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import nltk
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if not already present
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("Downloading NLTK POS tagger...")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt', quiet=True)

# ============================================================================
# POS Tag Extractor
# ============================================================================

class PosTagExtractor(BaseEstimator, TransformerMixin):
    """Extract Part-of-Speech tags from text using NLTK"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract POS tags using NLTK with parallel processing"""
        from joblib import Parallel, delayed
        import multiprocessing
        
        # Truncate texts to first 5k chars (faster processing, still captures style)
        truncated_texts = [text[:5000] for text in X]
        
        def extract_pos_tags(text):
            """Extract POS tags for a single text"""
            try:
                tokens = nltk.word_tokenize(text)
                pos_tags = nltk.pos_tag(tokens)
                return " ".join([tag for word, tag in pos_tags])
            except Exception as e:
                return ""
        
        # Use all available CPU cores for parallel processing
        n_jobs = multiprocessing.cpu_count()
        print(f"Extracting POS tags using {n_jobs} CPU cores (threading backend)...")
        
        # Process in parallel with threading backend (more stable than multiprocessing)
        pos_texts = Parallel(n_jobs=n_jobs, backend='threading', verbose=10)(
            delayed(extract_pos_tags)(text) for text in truncated_texts
        )
        
        return pos_texts

# ============================================================================
# Feature Pipeline
# ============================================================================

def build_feature_pipeline():
    """
    Build McGovern-style composite feature pipeline
    
    Note: McGovern et al. used 60k samples with 6k features (ratio: 10:1)
    We have 6k samples, so we use 1.5k features (ratio: 4:1) to prevent overfitting
    """
    return FeatureUnion([
        ('word_ngrams', TfidfVectorizer(
            ngram_range=(3, 5),
            max_features=500,  # Reduced from 2000 for our smaller dataset
            analyzer='word',
            lowercase=True
        )),
        ('char_ngrams', TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=500,  # Reduced from 2000 for our smaller dataset
            analyzer='char',
            lowercase=True
        )),
        ('pos_ngrams', Pipeline([
            ('pos_extractor', PosTagExtractor()),
            ('pos_vectorizer', TfidfVectorizer(
                ngram_range=(3, 5),
                max_features=500,  # Reduced from 2000 for our smaller dataset
                analyzer='word'
            ))
        ]))
    ])

# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(file_path='classifier_training_samples.jsonl'):
    """Load training data and extract responses with labels"""
    print(f"Loading training data from {file_path}...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    df = pd.DataFrame(data)
    
    # Extract responses and labels (only for judge models)
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        # Model A response (only if it's a judge model)
        if row['model_a_name'] == row['judge_model']:
            texts.append(row['model_a_response'])
            labels.append(row['model_a_name'])
        
        # Model B response (only if it's a judge model)
        if row['model_b_name'] == row['judge_model']:
            texts.append(row['model_b_response'])
            labels.append(row['model_b_name'])
    
    print(f"Loaded {len(texts)} responses from {len(df)} battles")
    
    # CRITIQUE FIX #1: Verify class balance
    print(f"\nLabel distribution (Class Balance Check):")
    label_counts = pd.Series(labels).value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Check for severe imbalance
    min_count = label_counts.min()
    max_count = label_counts.max()
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 2.0:
        print(f"\n⚠ WARNING: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        print(f"  This may bias the classifier. Consider resampling.")
    else:
        print(f"\nDone: Classes are balanced (ratio: {imbalance_ratio:.2f})")
    
    return texts, labels

# ============================================================================
# Training
# ============================================================================

def train_classifier(X_train, y_train, optimize=True):
    """Train XGBoost classifier with GridSearch optimization"""
    
    print("\nBuilding and extracting features (this will take a few minutes)...")
    print("This is done ONCE before GridSearch to avoid repetition.")
    
    # Encode labels (XGBoost needs numeric labels)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    print(f"Done: Labels encoded: {label_encoder.classes_}", flush=True)
    
    # Build feature pipeline
    feature_pipeline = build_feature_pipeline()
    
    # PRE-COMPUTE features once (avoid re-extraction during GridSearch)
    print("\nExtracting features from training data...", flush=True)
    print("This will extract word n-grams, char n-grams, and POS n-grams...", flush=True)
    X_train_features = feature_pipeline.fit_transform(X_train)
    print(f"Done: Features extracted: {X_train_features.shape}", flush=True)
    
    # Now train classifier on pre-computed features
    classifier = XGBClassifier(
        eval_metric='mlogloss',
        random_state=42
    )
    
    if optimize:
        print("\nPerforming GridSearch for hyperparameter optimization...")
        print("(This adapts McGovern et al.'s parameters to our dataset)")
        
        # CRITIQUE FIX #2: Add colsample_bytree to prevent overfitting
        # Full grid for thorough hyperparameter search (324 combinations)
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 6, 8],  # Full range
            'min_child_weight': [1, 5, 10],  # Full range
            'subsample': [0.8, 1.0],  # Full range
            'colsample_bytree': [0.3, 0.7]
        }
        print(f"Fitting 3 folds for each of {3*3*3*3*2*2} candidates = {3*3*3*3*2*2*3} total fits", flush=True)
        print(f"Estimated time: ~12 minutes (features are pre-computed)", flush=True)
        import sys
        sys.stdout.flush()
        
        grid = GridSearchCV(
            classifier,
            param_grid,
            cv=3,
            verbose=2,
            n_jobs=-1,  # Use all cores for GridSearch (features already extracted)
            scoring='accuracy'
        )
        
        print("Starting GridSearch fit...", flush=True)
        try:
            grid.fit(X_train_features, y_train_encoded)
            print("GridSearch completed successfully!", flush=True)
        except Exception as e:
            print(f"\n✗ GridSearch FAILED with error:", flush=True)
            print(f"Error type: {type(e).__name__}", flush=True)
            print(f"Error message: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        print(f"\nBest parameters found:")
        for param, value in grid.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV accuracy: {grid.best_score_:.4f}")
        
        best_classifier = grid.best_estimator_
    else:
        print("\nTraining with default parameters...")
        classifier.fit(X_train_features, y_train_encoded)
        best_classifier = classifier
    
    # CRITICAL FIX: Create final pipeline with FITTED feature pipeline
    # This ensures we don't re-extract features when using the model
    final_pipeline = Pipeline([
        ('features', feature_pipeline),  # Already fitted above
        ('classifier', best_classifier)  # Already trained on extracted features
    ])
    
    # Store label encoder as an attribute for later use
    final_pipeline.label_encoder_ = label_encoder
    
    return final_pipeline

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_classifier(pipeline, X_test, y_test):
    """Evaluate classifier and print metrics"""
    print("\n" + "="*70)
    print("Evaluation on Test Set")
    print("="*70)
    
    # Encode test labels
    y_test_encoded = pipeline.label_encoder_.transform(y_test)
    
    # Predict (returns encoded labels)
    y_pred_encoded = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Random baseline
    unique_labels = np.unique(y_test_encoded)
    random_baseline = 1.0 / len(unique_labels)
    print(f"Random Baseline: {random_baseline:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if accuracy >= 0.70:
        print("Done: Strong stylistic fingerprints detected")
    elif accuracy >= 0.50:
        print("MODERATE: Some fingerprints detected")
    else:
        print("POOR: Weak fingerprints (may need more data or features)")
    
    # Detailed report (decode labels for readability)
    y_pred_decoded = pipeline.label_encoder_.inverse_transform(y_pred_encoded)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_decoded, zero_division=0))
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_decoded)
    label_names = pipeline.label_encoder_.classes_
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)
    
    return accuracy

# ============================================================================
# Main
# ============================================================================

def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train McGovern-style classifier')
    parser.add_argument('--train_file', type=str, default='classifier_training_samples.jsonl',
                        help='Path to training data JSONL file')
    parser.add_argument('--output', type=str, default='mcgovern_classifier.pkl',
                        help='Output filename for trained model')
    args = parser.parse_args()

    print("="*70)
    print("McGovern-Style Classifier Training")
    print("="*70)
    
    # Load data
    X, y = load_training_data(args.train_file)
    
    # Split
    print(f"\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train
    pipeline = train_classifier(X_train, y_train, optimize=True)
    
    # Evaluate
    accuracy = evaluate_classifier(pipeline, X_test, y_test)
    
    # Save model
    model_path = args.output
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\nDone: Model saved to {model_path}")
    
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("1. Test classifier on judge_samples.jsonl (baseline)")
    print("2. Run interventions (back-translation, paraphrasing)")
    print("3. Test classifier on intervention datasets")
    print("4. Compare accuracy drops to validate style removal")

if __name__ == "__main__":
    main()
