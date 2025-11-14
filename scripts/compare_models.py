"""
Compare all team members' model results
Run this after everyone has trained their models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

def load_all_metrics(metrics_dir='../results/metrics'):
    """
    Load all saved model metrics from JSON files
    
    Returns:
        DataFrame with all model results
    """
    metrics_dict = {}
    
    # Find all JSON files in metrics directory
    metrics_path = Path(metrics_dir)
    
    if not metrics_path.exists():
        print(f"❌ Error: {metrics_dir} directory not found!")
        print("   Make sure team members have run their models first.")
        return None
    
    json_files = list(metrics_path.glob('*_metrics.json'))
    
    if len(json_files) == 0:
        print(f"❌ No metrics files found in {metrics_dir}")
        return None
    
    print(f"📊 Found {len(json_files)} model results:")
    
    for json_file in json_files:
        model_key = json_file.stem.replace('_metrics', '')
        
        with open(json_file, 'r') as f:
            metrics_dict[model_key] = json.load(f)
        
        print(f"   ✓ {metrics_dict[model_key]['model_name']}")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    
    # Select relevant columns
    columns_to_show = ['model_name', 'test_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    df = df[columns_to_show]
    
    # Sort by test accuracy
    df = df.sort_values('test_accuracy', ascending=False)
    
    return df


def create_comparison_table(df, save_path='../results/model_comparison.csv'):
    """
    Create and save comparison table
    """
    # Rename columns for clarity
    df_display = df.rename(columns={
        'model_name': 'Model',
        'test_accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'roc_auc': 'ROC-AUC'
    })
    
    # Round to 3 decimals
    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    df_display[numeric_cols] = df_display[numeric_cols].round(3)
    
    # Save to CSV
    df_display.to_csv(save_path, index=False)
    print(f"\n✅ Comparison table saved to {save_path}")
    
    # Print table
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(df_display.to_string(index=False))
    print("="*80)
    
    return df_display


def plot_model_comparison(df, save_path='../results/model_comparison.png'):
    """
    Create bar chart comparing models
    """
    # Prepare data for plotting
    df_plot = df.copy()
    df_plot = df_plot.set_index('model_name')
    
    # Select metrics to plot
    metrics_to_plot = ['test_accuracy', 'precision', 'recall', 'f1_score']
    df_plot = df_plot[metrics_to_plot]
    
    # Rename for display
    df_plot = df_plot.rename(columns={
        'test_accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    })
    
    # Transpose for grouped bar chart
    df_transposed = df_plot.transpose()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_transposed.plot.bar(ax=ax, rot=0, width=0.8)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_title("Model Comparison Across Key Metrics", fontsize=14, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison chart saved to {save_path}")
    plt.show()


def find_best_model(df):
    """
    Determine best model based on different criteria
    """
    print("\n" + "="*80)
    print("BEST MODELS BY METRIC")
    print("="*80)
    
    metrics = {
        'Highest Accuracy': ('test_accuracy', True),
        'Highest Precision': ('precision', True),
        'Highest Recall': ('recall', True),
        'Best F1 Score': ('f1_score', True),
        'Highest ROC-AUC': ('roc_auc', True)
    }
    
    for label, (metric, ascending) in metrics.items():
        if df[metric].notna().any():  # Check if metric exists
            best_idx = df[metric].idxmax() if not ascending else df[metric].idxmax()
            best_model = df.loc[best_idx, 'model_name']
            best_score = df.loc[best_idx, metric]
            print(f"{label:20s}: {best_model:25s} ({best_score:.3f})")
    
    print("="*80)


if __name__ == "__main__":
    print("🔍 Loading all model results...\n")
    
    # Load metrics
    df = load_all_metrics()
    
    if df is not None:
        # Create comparison table
        create_comparison_table(df)
        
        # Plot comparison
        plot_model_comparison(df)
        
        # Find best models
        find_best_model(df)
        
        print("\n✅ Model comparison complete!")
    else:
        print("\n❌ No models to compare yet. Train some models first!")