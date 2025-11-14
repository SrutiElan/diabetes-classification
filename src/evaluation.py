"""
Model evaluation utilities
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model", save_results=True):
    """
    Comprehensive model evaluation with automatic result saving
    
    Args:
        model: Trained model
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        model_name: Name of the model (used for saving)
        save_results: Whether to save metrics to JSON file
    
    Returns:
        Dictionary with all metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities (if available)
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = None
    
    # Calculate metrics
    results = {
        'model_name': model_name,
        'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_test_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba)) if y_test_proba is not None else None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    print(f"Training Accuracy:    {results['train_accuracy']:.3f}")
    print(f"Test Accuracy:        {results['test_accuracy']:.3f}")
    print(f"Precision:            {results['precision']:.3f}")
    print(f"Recall:               {results['recall']:.3f}")
    print(f"F1 Score:             {results['f1_score']:.3f}")
    if results['roc_auc']:
        print(f"ROC-AUC:              {results['roc_auc']:.3f}")
    
    # Check overfitting
    overfit_gap = results['train_accuracy'] - results['test_accuracy']
    if overfit_gap > 0.10:
        print(f"\n⚠️  Warning: Possible overfitting (train-test gap: {overfit_gap:.3f})")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['No Diabetes', 'Diabetes']))
    
    # Save results to JSON
    if save_results:
        save_path = f'results/metrics/{model_name.lower().replace(" ", "_")}_metrics.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Metrics saved to {save_path}")
    
    return results


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path is None:
        save_path = f'results/confusion_matrices/{model_name.lower().replace(" ", "_")}_cm.png'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name="Model", save_path=None):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path is None:
        save_path = f'results/roc_curves/{model_name.lower().replace(" ", "_")}_roc.png'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curve saved to {save_path}")
    
    plt.show()