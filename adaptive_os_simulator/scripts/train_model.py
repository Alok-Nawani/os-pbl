from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    mean_squared_error,
    r2_score
)

from adaptive_os_simulator.backend.ml_model import FEATURE_NAMES, save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and compare multiple ML classifiers with comprehensive visualizations")
    p.add_argument("--csv", required=True, help="Path to dataset CSV (must contain FEATURE_NAMES and 'policy')")
    p.add_argument("--out", required=True, help="Base path to save models (each model will append name before extension)")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--plots_dir", default="plots", help="Directory to save visualization plots")
    return p.parse_args()


def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create comprehensive bar charts comparing algorithm performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy, Precision, Recall, F1-Score
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy Comparison', 'Precision Comparison', 'Recall Comparison', 'F1-Score Comparison']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        if metric in results_df.columns:
            data = results_df[metric].sort_values(ascending=False)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
            bars = ax.bar(range(len(data)), data.values, color=colors)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics comparison plot to {output_dir}/metrics_comparison.png")
    plt.close()
    
    # Plot 2: Training and Prediction Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    if 'train_time' in results_df.columns:
        data = results_df['train_time'].sort_values(ascending=True)
        colors = plt.cm.autumn(np.linspace(0.3, 0.9, len(data)))
        bars = ax1.barh(range(len(data)), data.values, color=colors)
        ax1.set_yticks(range(len(data)))
        ax1.set_yticklabels(data.index)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.4f}s', ha='left', va='center', fontsize=9)
    
    if 'predict_time' in results_df.columns:
        data = results_df['predict_time'].sort_values(ascending=True)
        colors = plt.cm.spring(np.linspace(0.3, 0.9, len(data)))
        bars = ax2.barh(range(len(data)), data.values, color=colors)
        ax2.set_yticks(range(len(data)))
        ax2.set_yticklabels(data.index)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Prediction Time Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.4f}s', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved time comparison plot to {output_dir}/time_comparison.png")
    plt.close()
    
    # Plot 3: Overall Performance Radar Chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Select top 5 models by accuracy
    top_models = results_df.nlargest(5, 'accuracy')
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    for idx, (model_name, row) in enumerate(top_models.iterrows()):
        values = [row['accuracy'], row['precision'], row['recall'], row['f1_score']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Models - Performance Radar Chart', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance radar chart to {output_dir}/performance_radar.png")
    plt.close()


def plot_confusion_matrices(models_dict: Dict, X_test, y_test, labels: List[str], output_dir: str) -> None:
    """Generate confusion matrix heatmaps for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    n_models = len(models_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold')
        
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {output_dir}/confusion_matrices.png")
    plt.close()


def plot_feature_importance(models_dict: Dict, feature_names: List[str], output_dir: str) -> None:
    """Plot feature importance for tree-based models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect models with feature_importances_
    importance_models = {name: model for name, model in models_dict.items() 
                        if hasattr(model, 'feature_importances_')}
    
    if not importance_models:
        print("No models with feature importance available")
        return
    
    n_models = len(importance_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 5 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(importance_models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        ax = axes[idx]
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(feature_names)))
        ax.bar(range(len(feature_names)), importances[indices], color=colors)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Importance')
        ax.set_title(f'{name} - Feature Importance', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(importances[indices]):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance plots to {output_dir}/feature_importance.png")
    plt.close()


def plot_roc_curves(models_dict: Dict, X_test, y_test, labels: List[str], output_dir: str) -> None:
    """Generate ROC curves for models with predict_proba."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Only for models with predict_proba
    prob_models = {name: model for name, model in models_dict.items() 
                   if hasattr(model, 'predict_proba')}
    
    if not prob_models or len(labels) > 10:  # Skip if too many classes
        print("Skipping ROC curves (no probability models or too many classes)")
        return
    
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    
    # Create one plot per model
    for name, model in prob_models.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            y_proba = model.predict_proba(X_test)
            
            # One-vs-Rest ROC for each class
            for i, label in enumerate(labels):
                y_true_binary = (y_test_encoded == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, 
                       label=f'{label} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{name} - ROC Curves (One-vs-Rest)', fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            safe_name = name.replace(' ', '_').lower()
            plt.savefig(os.path.join(output_dir, f'roc_curve_{safe_name}.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved ROC curve for {name} to {output_dir}/roc_curve_{safe_name}.png")
            plt.close()
        except Exception as e:
            print(f"Could not generate ROC curve for {name}: {e}")
            plt.close()


def create_summary_report(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create a text summary report of all results."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'model_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MACHINE LEARNING ALGORITHM COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL RANKINGS:\n")
        f.write("-" * 80 + "\n")
        
        # Rank by accuracy
        f.write("\n1. BY ACCURACY:\n")
        sorted_acc = results_df.sort_values('accuracy', ascending=False)
        for idx, (name, row) in enumerate(sorted_acc.iterrows(), 1):
            f.write(f"   {idx}. {name:25s} : {row['accuracy']:.4f}\n")
        
        # Rank by F1
        f.write("\n2. BY F1-SCORE:\n")
        sorted_f1 = results_df.sort_values('f1_score', ascending=False)
        for idx, (name, row) in enumerate(sorted_f1.iterrows(), 1):
            f.write(f"   {idx}. {name:25s} : {row['f1_score']:.4f}\n")
        
        # Rank by training time (fastest first)
        f.write("\n3. BY TRAINING SPEED (Fastest First):\n")
        sorted_time = results_df.sort_values('train_time', ascending=True)
        for idx, (name, row) in enumerate(sorted_time.iterrows(), 1):
            f.write(f"   {idx}. {name:25s} : {row['train_time']:.4f}s\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED METRICS:\n")
        f.write("=" * 80 + "\n\n")
        f.write(results_df.to_string())
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("WHY RANDOM FOREST OFTEN PERFORMS BEST:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
Random Forest is an ensemble learning method that typically excels for several reasons:

1. ENSEMBLE POWER:
   - Combines predictions from multiple decision trees
   - Reduces variance and overfitting through averaging
   - More robust than single decision trees

2. FEATURE HANDLING:
   - Automatically handles feature interactions
   - Robust to irrelevant features (they get averaged out)
   - Works well with both numerical and categorical data
   - No need for extensive feature scaling

3. OVERFITTING RESISTANCE:
   - Random feature selection at each split
   - Bootstrap aggregating (bagging) of training samples
   - Natural regularization through ensemble averaging

4. INTERPRETABILITY:
   - Provides feature importance scores
   - Helps understand which features drive predictions
   - Can be used for feature selection

5. PRACTICAL ADVANTAGES:
   - Few hyperparameters to tune
   - Works well with default settings
   - Handles missing values reasonably
   - Parallel training capability
   - Good performance across various problem types

PERFORMANCE IN THIS DATASET:
""")
        
        best_model = results_df['accuracy'].idxmax()
        best_acc = results_df.loc[best_model, 'accuracy']
        
        if 'random_forest' in results_df.index:
            rf_acc = results_df.loc['random_forest', 'accuracy']
            f.write(f"\n• Best performing model: {best_model} (Accuracy: {best_acc:.4f})\n")
            f.write(f"• Random Forest accuracy: {rf_acc:.4f}\n")
            
            if best_model == 'random_forest':
                f.write("\n✓ Random Forest achieved the BEST accuracy in this comparison!\n")
                f.write("  This validates its reputation as a highly effective general-purpose algorithm.\n")
            else:
                diff = best_acc - rf_acc
                f.write(f"\n• Performance difference: {diff:.4f}\n")
                if diff < 0.02:
                    f.write("  Random Forest performs nearly as well and may still be preferred for:\n")
                    f.write("  - Better interpretability via feature importances\n")
                    f.write("  - More stable predictions\n")
                    f.write("  - Less risk of overfitting\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Saved detailed report to {report_path}")
    
    # Also print summary to console
    print("\n" + "=" * 80)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 80)
    print("\nTop 5 Models by Accuracy:")
    print(results_df.nlargest(5, 'accuracy')[['accuracy', 'precision', 'recall', 'f1_score']].to_string())
    print("\n" + "=" * 80 + "\n")


def main() -> None:
    args = parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE ML ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"\nLoading dataset from: {args.csv}")
    
    df = pd.read_csv(args.csv)

    # Expect dataset to have precomputed features and 'policy' target
    missing = [c for c in FEATURE_NAMES + ["policy"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = df[FEATURE_NAMES]
    y = df["policy"].astype(str)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {y.nunique()}")
    print(f"Class distribution:\n{y.value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")

    # Prepare comprehensive model suite
    print("Initializing algorithms...")
    models: Dict[str, Any] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            random_state=args.random_state, 
            class_weight="balanced",
            max_depth=15,
            min_samples_split=5
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=args.random_state,
            class_weight="balanced",
            max_depth=15
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=args.random_state,
            max_depth=5,
            learning_rate=0.1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=args.random_state, 
            class_weight="balanced",
            max_depth=15
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=args.random_state,
            learning_rate=1.0
        ),
        "SVM (RBF)": SVC(
            kernel='rbf',
            random_state=args.random_state,
            class_weight="balanced",
            probability=True
        ),
        "Logistic Regression": LogisticRegression(
            random_state=args.random_state,
            class_weight="balanced",
            max_iter=1000,
            multi_class='multinomial'
        ),
        "Naive Bayes": GaussianNB(),
    }

    # Try to add XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            use_label_encoder=False, 
            eval_metric="mlogloss", 
            random_state=args.random_state,
            max_depth=6
        )
        print("✓ XGBoost available")
    except ImportError:
        print("⚠ XGBoost not installed; skipping. Install with: pip install xgboost")

    print(f"\nTotal algorithms to evaluate: {len(models)}\n")
    print("=" * 80)

    labels = sorted(y.unique())
    base_out = args.out
    root, ext = os.path.splitext(base_out)
    if not ext:
        ext = ".joblib"

    # Containers for results
    results: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"Training: {name}")
        print('=' * 80)
        
        # Train
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Predict
        start_pred = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_pred
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # R2 and MSE (encoding labels as proxy metrics)
        le = LabelEncoder()
        y_test_encoded = le.fit_transform(y_test)
        y_pred_encoded = le.transform(y_pred)
        
        try:
            r2 = r2_score(y_test_encoded, y_pred_encoded)
        except:
            r2 = float('nan')
        
        try:
            mse = mean_squared_error(y_test_encoded, y_pred_encoded)
        except:
            mse = float('nan')
        
        # ROC AUC (if model supports predict_proba and not too many classes)
        roc_auc = None
        if hasattr(model, 'predict_proba') and len(labels) <= 10:
            try:
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                pass
        
        print(f"✓ Training completed in {train_time:.4f}s")
        print(f"✓ Prediction completed in {predict_time:.4f}s")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  R²:        {r2:.4f}")
        print(f"  MSE:       {mse:.4f}")
        
        # Store results
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc if roc_auc else np.nan,
            'r2': r2,
            'mse': mse,
            'train_time': train_time,
            'predict_time': predict_time
        }
        
        # Save model
        out_path = f"{root}_{name.replace(' ', '_').lower()}{ext}"
        save_model(model, out_path)
        print(f"✓ Model saved to {out_path}")
        
        trained_models[name] = model

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")

    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    # Generate all plots
    plot_metrics_comparison(results_df, args.plots_dir)
    plot_confusion_matrices(trained_models, X_test, y_test, labels, args.plots_dir)
    plot_feature_importance(trained_models, FEATURE_NAMES, args.plots_dir)
    plot_roc_curves(trained_models, X_test, y_test, labels, args.plots_dir)
    
    # Create summary report
    create_summary_report(results_df, args.plots_dir)
    
    print("\n" + "=" * 80)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {args.plots_dir}/")
    print(f"Models saved to: {os.path.dirname(root)}/")
    print("\nCheck the following files:")
    print(f"  • {args.plots_dir}/model_comparison_report.txt - Detailed text report")
    print(f"  • {args.plots_dir}/metrics_comparison.png - Performance metrics")
    print(f"  • {args.plots_dir}/time_comparison.png - Training/prediction time")
    print(f"  • {args.plots_dir}/performance_radar.png - Radar chart")
    print(f"  • {args.plots_dir}/confusion_matrices.png - Confusion matrices")
    print(f"  • {args.plots_dir}/feature_importance.png - Feature importances")
    print(f"  • {args.plots_dir}/roc_curve_*.png - ROC curves\n")


if __name__ == "__main__":
    main()


