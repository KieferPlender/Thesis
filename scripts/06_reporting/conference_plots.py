"""
Generates publication-ready figures suitable for IEEE/ACM conference papers.

Consolidates ALL plotting functionality. It replaces
the following individual plot scripts that were previously in the root directory:
- plot_baseline_spr.py
- plot_intervention_spr.py
- plot_classifier_accuracy_drop.py
- plot_per_model_classifier_metrics.py
- plot_per_model_intervention_effects.py
- plots.py

All plots use:
- Times New Roman / serif fonts (academic standard)
- Minimum 12pt font size for readability
- Proper figure dimensions for column/page width
- Color-blind friendly, high-contrast palette

Usage:
    python conference_plots.py              # Generate all plots
    
Output:
    All plots are saved to the results/ directory as PNG files.
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
    models = ['Claude-3.5-Haiku', 'ChatGPT-4o', 'DeepSeek-R1']
    spr_values = [21.38, 54.33, 79.93]
    colors = [COLORS['claude'], COLORS['chatgpt'], COLORS['deepseek']]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    bars = ax.barh(models, spr_values, color=colors, height=0.55, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Baseline Self-Preference Rate (%)')
    ax.set_xlim(0, 100)
    
    # Add value labels
    for bar, value in zip(bars, spr_values):
        ax.text(value + 2, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', ha='left', fontsize=12, fontweight='medium')
    
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
    models = ['Claude-3.5-Haiku', 'ChatGPT-4o', 'DeepSeek-R1']
    
    # Delta SPR values for each intervention
    markdown_removal = [1.02, -4.14, -4.27]
    back_translation = [-1.72, -5.43, -5.15]
    paraphrasing = [1.73, -1.47, -2.51]
    
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
                    f'{width:+.1f}', va='center', ha=ha, fontsize=10)
    
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
    interventions = ['Paraphrasing', 'Markdown Removal', 'Back-Translation']
    accuracy_drops = [-33.47, -30.03, -5.24]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    colors = [COLOR_PALETTE[2], COLOR_PALETTE[0], COLOR_PALETTE[1]]
    bars = ax.barh(interventions, accuracy_drops, color=colors, height=0.55, edgecolor='white')
    
    ax.set_xlabel('Change in Classifier Accuracy (pp)')
    ax.set_xlim(-42, 5)
    ax.axvline(x=0, color='black', linewidth=1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, accuracy_drops):
        ax.text(value - 0.8, bar.get_y() + bar.get_height()/2, 
                f'{value:+.1f}pp', va='center', ha='right', fontsize=11, fontweight='medium')
    
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
    models = ['ChatGPT-4o', 'Claude-3.5-Haiku', 'DeepSeek-R1']
    
    precision = [0.7942, 0.8002, 0.8215]
    recall = [0.7910, 0.8370, 0.7870]
    f1_score = [0.7926, 0.8182, 0.8039]
    
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
    ax.set_title('Per-Model Classifier Performance (Baseline Accuracy: 80.50%)')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
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
    
    interventions = ['Baseline', 'Markdown Removal', 'Back-Translation', 'Paraphrasing']
    chatgpt_acc = [data[i]['chatgpt'] * 100 for i in interventions]
    claude_acc = [data[i]['claude'] * 100 for i in interventions]
    deepseek_acc = [data[i]['deepseek'] * 100 for i in interventions]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    x = np.arange(len(interventions))
    width = 0.25
    
    bars1 = ax.bar(x - width, chatgpt_acc, width, label='ChatGPT-4o', color=COLORS['chatgpt'], edgecolor='white')
    bars2 = ax.bar(x, claude_acc, width, label='Claude-3.5-Haiku', color=COLORS['claude'], edgecolor='white')
    bars3 = ax.bar(x + width, deepseek_acc, width, label='DeepSeek-R1', color=COLORS['deepseek'], edgecolor='white')
    
    ax.set_xlabel('Intervention')
    ax.set_ylabel('Classifier Accuracy (%)')
    ax.set_title('Per-Model Classifier Accuracy Under Each Intervention')
    ax.set_xticks(x)
    ax.set_xticklabels(interventions)
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
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
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
        print(f"⚠ Skipping top10_features_per_model: {feature_path} not found")
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


# =============================================================================
# PLOT 7: Intervention Examples Figure
# =============================================================================

def plot_intervention_examples():
    """
    Compact academic-style text comparison figure.
    Smaller figure = larger text when embedded in paper.
    """
    json_path = f'{RESULTS_DIR}/metrics/intervention_comparison.json'
    if not os.path.exists(json_path):
        print(f"⚠ Skipping intervention_examples: {json_path} not found")
        print("  Run the intervention comparison script first to generate the data.")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Taller figure to fit full text
    fig = plt.figure(figsize=(7, 8))
    
    # Use full text (no truncation)
    samples = [
        ('(a) Baseline', data['baseline'], '#444444'),
        ('(b) Markdown Removal', data['markdown_removed'], COLOR_PALETTE[0]),
        ('(c) Back-Translation', data['back_translated'], COLOR_PALETTE[1]),
        ('(d) Paraphrasing', data['paraphrased'], COLOR_PALETTE[2])
    ]
    
    # Grid spec for 2x2 layout with title space
    gs = fig.add_gridspec(3, 2, height_ratios=[0.08, 1, 1], 
                          hspace=0.12, wspace=0.10,
                          left=0.03, right=0.97, top=0.94, bottom=0.02)
    
    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Intervention Examples', 
                  ha='center', va='center', fontsize=13, fontweight='bold')
    ax_title.text(0.5, -0.3, f'Prompt: "{data["user_prompt"]}"',
                  ha='center', va='center', fontsize=10, style='italic', color='#555')
    
    # Panel positions
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    
    for (title, text, color), (row, col) in zip(samples, positions):
        ax = fig.add_subplot(gs[row, col])
        
        # Clean border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(color)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Title - bold
        ax.text(0.03, 0.97, title, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=color,
                va='top', ha='left')
        
        # Colored underline
        ax.plot([0.03, 0.97], [0.92, 0.92], color=color, linewidth=1.5,
                transform=ax.transAxes, clip_on=False)
        
        # Main text - full text, wrapped appropriately
        wrapped = textwrap.fill(text, width=30)
        ax.text(0.03, 0.89, wrapped, transform=ax.transAxes,
                fontsize=10, color='#1a1a1a', va='top', ha='left',
                family='serif', linespacing=1.25)
        
        ax.set_facecolor('white')
    
    plt.savefig(f'{OUTPUT_DIR}/intervention_examples.png')
    plt.close()
    print("Done: intervention_examples.png")



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
    plot_intervention_examples()
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("Files created:")
    print("  - baseline_self_preference.pdf/png")
    print("  - intervention_spr_changes.pdf/png")
    print("  - classifier_accuracy_drop.pdf/png")
    print("  - per_model_classifier_metrics.pdf/png")
    print("  - per_model_intervention_effects.pdf/png")
    print("  - top10_features_per_model.pdf/png")
    print("  - intervention_examples.pdf/png")


if __name__ == '__main__':
    main()
