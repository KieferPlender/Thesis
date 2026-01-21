"""
Generates publication-ready figures suitable for IEEE/ACM conference papers.

Consolidates ALL plotting functionality.

All plots use:
- Times New Roman / serif fonts (academic standard)
- Minimum 12pt font size for readability
- Proper figure dimensions for column/page width
- Color-blind friendly, high-contrast palette

Usage:
    python scripts/06_reporting/conference_plots.py              # Generate all plots
    
Output:
    All plots are saved to the results/plots directory as PNG files.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import json
import textwrap
import os

# Set up matplotlib for publication quality
plt.rcParams.update({
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 12,
    
    # Axes settings
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # Tick settings
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    
    # Legend settings
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    
    # Figure settings
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # Grid settings
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
})

# Color palette - color-blind friendly and print-safe
COLORS = {
    'chatgpt': '#2E7D32',      # Dark green
    'claude': '#1565C0',       # Dark blue  
    'deepseek': '#E65100',     # Dark orange
    'primary': '#1565C0',
    'secondary': '#2E7D32',
    'tertiary': '#E65100',
}

# Alternative for grouped charts
COLOR_PALETTE = ['#1565C0', '#2E7D32', '#E65100']  # Blue, Green, Orange

# Figure sizes (inches) - IEEE column width = 3.5", page width = 7.16"
FIG_SINGLE_COL = (3.5, 2.8)
FIG_DOUBLE_COL = (7, 4)
FIG_TALL = (7, 5.5)
FIG_EXTRA_TALL = (7, 7)

# Directories
RESULTS_DIR = 'results'
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'plots')

def ensure_output_dir():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PLOT 1: Baseline Self-Preference Rate
# =============================================================================

def plot_baseline_spr():
    """
    Horizontal bar chart showing baseline self-preference rates for each model.
    """
    # Load data
    data_path = f'{RESULTS_DIR}/metrics/spr_results.json'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping baseline_self_preference: {data_path} not found")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    baseline = data['Baseline']
    models = ['Claude-3.5-Haiku', 'ChatGPT-4o', 'DeepSeek-R1']
    spr_values = [baseline['claude'], baseline['chatgpt'], baseline['deepseek']]
    colors = [COLORS['claude'], COLORS['chatgpt'], COLORS['deepseek']]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    bars = ax.barh(models, spr_values, color=colors, height=0.55, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Baseline Self-Preference Rate (%)')
    ax.set_xlim(0, 100)
    
    # Add value labels
    for bar, value in zip(bars, spr_values):
        ax.text(value + 2, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}%', va='center', ha='left', fontsize=12, fontweight='medium')
    
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/baseline_self_preference.png')
    plt.close()
    print("Done: baseline_self_preference.png")


# =============================================================================
# PLOT 2: Intervention SPR Changes
# =============================================================================

def plot_intervention_spr():
    """
    Grouped horizontal bar chart showing SPR changes for each intervention per model.
    """
    # Load data
    data_path = f'{RESULTS_DIR}/metrics/spr_results.json'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping intervention_spr_changes: {data_path} not found")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Calculate deltas for each model
    baseline = data['Baseline']
    markdown_results = data['Markdown Removal']
    backtrans_results = data['Back-Translation']
    paraphrase_results = data['Paraphrasing']
    
    models = ['Claude-3.5-Haiku', 'ChatGPT-4o', 'DeepSeek-R1']
    keys = ['claude', 'chatgpt', 'deepseek']
    markdown_removal = [markdown_results[k] - baseline[k] for k in keys]
    back_translation = [backtrans_results[k] - baseline[k] for k in keys]
    paraphrasing = [paraphrase_results[k] - baseline[k] for k in keys]
    
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)
    
    y_pos = np.arange(len(models))
    bar_height = 0.22
    
    bars1 = ax.barh(y_pos - bar_height, markdown_removal, bar_height, 
                    label='Markdown Removal', color=COLOR_PALETTE[0], edgecolor='white')
    bars2 = ax.barh(y_pos, back_translation, bar_height, 
                    label='Back-Translation', color=COLOR_PALETTE[1], edgecolor='white')
    bars3 = ax.barh(y_pos + bar_height, paraphrasing, bar_height, 
                    label='Paraphrasing', color=COLOR_PALETTE[2], edgecolor='white')
    
    ax.set_xlabel('Change in Self-Preference Rate (pp)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlim(-7, 3)
    ax.axvline(x=0, color='black', linewidth=1.0, linestyle='-')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            width = bar.get_width()
            label_x = width + 0.2 if width > 0 else width - 0.2
            ha = 'left' if width > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{width:+.2f}', va='center', ha=ha, fontsize=10)
    
    ax.legend(loc='lower left', ncol=1)
    
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/intervention_spr_changes.png')
    plt.close()
    print("Done: intervention_spr_changes.png")


# =============================================================================
# PLOT 3: Classifier Accuracy Drop
# =============================================================================

def plot_classifier_accuracy_drop():
    """
    Horizontal bar chart showing classifier accuracy drops for each intervention.
    """
    # Load data
    data_path = f'{RESULTS_DIR}/metrics/per_model_intervention_data.json'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping classifier_accuracy_drop: {data_path} not found")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    baseline_acc = data['Baseline']['overall']
    
    # Interventions to show (excluding baseline)
    plot_interventions = ['Paraphrasing', 'Markdown Removal', 'Back-Translation']
    accuracy_drops = []
    
    for inter in plot_interventions:
        acc = data[inter]['overall']
        drop = (acc - baseline_acc) * 100
        accuracy_drops.append(drop)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    colors = [COLOR_PALETTE[2], COLOR_PALETTE[0], COLOR_PALETTE[1]]
    bars = ax.barh(plot_interventions, accuracy_drops, color=colors, height=0.55, edgecolor='white')
    
    ax.set_xlabel('Change in Classifier Accuracy (pp)')
    ax.set_xlim(-42, 5)
    ax.axvline(x=0, color='black', linewidth=1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, accuracy_drops):
        ax.text(value - 0.8, bar.get_y() + bar.get_height()/2, 
                f'{value:+.2f}pp', va='center', ha='right', fontsize=11, fontweight='medium')
    
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/classifier_accuracy_drop.png')
    plt.close()
    print("Done: classifier_accuracy_drop.png")


# =============================================================================
# PLOT 4: Per-Model Classifier Metrics
# =============================================================================

def plot_per_model_classifier_metrics():
    """
    Grouped vertical bar chart showing precision, recall, F1 for each model.
    """
    # Load data
    data_path = f'{RESULTS_DIR}/metrics/per_model_intervention_data.json'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping per_model_classifier_metrics: {data_path} not found")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    baseline = data['Baseline']
    # Map model ID to display name and get metrics
    model_mapping = {
        'chatgpt-4o-latest-20250326': 'ChatGPT-4o',
        'claude-3-5-haiku-20241022': 'Claude-3.5-Haiku',
        'deepseek-r1-0528': 'DeepSeek-R1'
    }
    
    models = []
    precision = []
    recall = []
    f1_score = []
    
    for model_id, display_name in model_mapping.items():
        if model_id in baseline['baseline_metrics']:
            models.append(display_name)
            metrics = baseline['baseline_metrics'][model_id]
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_score.append(metrics['f1'])
    
    # Use overall accuracy in title
    overall_acc = baseline['overall']
    
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_COL)
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color=COLOR_PALETTE[0], edgecolor='white')
    bars2 = ax.bar(x, recall, width, label='Recall', color=COLOR_PALETTE[1], edgecolor='white')
    bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color=COLOR_PALETTE[2], edgecolor='white')
    
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.74, 0.86)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    # Title with accuracy info
    ax.set_title(f'Per-Model Classifier Performance (Baseline Accuracy: {overall_acc*100:.2f}%)')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/per_model_classifier_metrics.png')
    plt.close()
    print("Done: per_model_classifier_metrics.png")


# =============================================================================
# PLOT 5: Per-Model Intervention Effects (Classifier Accuracy)
# =============================================================================

def plot_per_model_intervention_effects():
    """
    Grouped vertical bar chart showing classifier accuracy under each intervention per model.
    """
    # Check if data file exists
    data_path = f'{RESULTS_DIR}/metrics/per_model_intervention_data.json'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping per_model_intervention_effects: {data_path} not found")
        return
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    interventions_acc = ['Baseline', 'Markdown Removal', 'Back-Translation', 'Paraphrasing']
    chatgpt_acc = [data[i]['chatgpt'] * 100 for i in interventions_acc]
    claude_acc = [data[i]['claude'] * 100 for i in interventions_acc]
    deepseek_acc = [data[i]['deepseek'] * 100 for i in interventions_acc]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    x = np.arange(len(interventions_acc))
    width = 0.25
    
    bars1 = ax.bar(x - width, chatgpt_acc, width, label='ChatGPT-4o', color=COLORS['chatgpt'], edgecolor='white')
    bars2 = ax.bar(x, claude_acc, width, label='Claude-3.5-Haiku', color=COLORS['claude'], edgecolor='white')
    bars3 = ax.bar(x + width, deepseek_acc, width, label='DeepSeek-R1', color=COLORS['deepseek'], edgecolor='white')
    
    ax.set_xlabel('Intervention')
    ax.set_ylabel('Classifier Accuracy (%)')
    ax.set_title('Per-Model Classifier Accuracy Under Each Intervention')
    ax.set_xticks(x)
    ax.set_xticklabels(interventions_acc)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add baseline reference line
    baseline_overall = (data['Baseline']['chatgpt'] + data['Baseline']['claude'] + data['Baseline']['deepseek']) / 3 * 100
    ax.axhline(y=baseline_overall, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/per_model_intervention_effects.png')
    plt.close()
    print("Done: per_model_intervention_effects.png")


# =============================================================================
# PLOT 6: Top 10 Features Per Model
# =============================================================================

def plot_top10_features_per_model():
    """
    Side-by-side horizontal bar charts showing top 10 features for each model.
    """
    feature_path = f'{RESULTS_DIR}/feature_data/feature_importance_top10_per_language_model.csv'
    if not os.path.exists(feature_path):
        print(f"Skipping top10_features_per_model: {feature_path} not found")
        return
    
    df = pd.read_csv(feature_path)
    df_original = df[df['classifier'] == 'Original Classifier']
    
    models = {
        'ChatGPT-4o': COLORS['chatgpt'],
        'Claude-3.5-Haiku': COLORS['claude'],
        'DeepSeek-R1': COLORS['deepseek']
    }
    
    # Side-by-side layout (1 row, 3 columns) - reduced width for larger text
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.5))
    fig.subplots_adjust(wspace=0.45, left=0.08, right=0.98, top=0.90, bottom=0.08)
    
    for idx, (model_name, color) in enumerate(models.items()):
        model_data = df_original[df_original['language_model'] == model_name].sort_values('rank')
        
        features = model_data['feature'].tolist()
        importance = model_data['importance'].tolist()
        
        axes[idx].barh(features, importance, color=color, height=0.7, edgecolor='white')
        axes[idx].set_xlabel('Feature Importance', fontsize=14)
        axes[idx].set_title(model_name, fontweight='bold', fontsize=18)
        axes[idx].set_xlim(0, max(importance) * 1.25)
        axes[idx].invert_yaxis()
        axes[idx].tick_params(axis='y', labelsize=16)
        
        # Add value labels
        for i, v in enumerate(importance):
            axes[idx].text(v + max(importance) * 0.02, i, f'{v:.4f}', 
                          va='center', fontsize=12)
    
    plt.savefig(f'{OUTPUT_DIR}/top10_features_per_model.png')
    plt.close()
    print("Done: top10_features_per_model.png")




def main():
    """Generate all conference-quality plots."""
    print("\n" + "="*60)
    print("GENERATING CONFERENCE PAPER QUALITY PLOTS")
    print("="*60 + "\n")
    
    ensure_output_dir()
    
    # Generate all plots
    plot_baseline_spr()
    plot_intervention_spr()
    plot_classifier_accuracy_drop()
    plot_per_model_classifier_metrics()
    plot_per_model_intervention_effects()
    plot_top10_features_per_model()
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("Files created:")
    print("  - baseline_self_preference.png")
    print("  - intervention_spr_changes.png")
    print("  - classifier_accuracy_drop.png")
    print("  - per_model_classifier_metrics.png")
    print("  - per_model_intervention_effects.png")
    print("  - top10_features_per_model.png")


if __name__ == '__main__':
    main()
