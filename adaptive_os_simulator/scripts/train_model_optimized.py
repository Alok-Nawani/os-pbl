from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, Any, List

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    accuracy_score,
    precision_recall_fscore_support,
)

from adaptive_os_simulator.backend.ml_model import save_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train models on ORIGINAL scheduler dataset with optimized hyperparameters")
    p.add_argument("--csv", required=True, help="Path to ORIGINAL dataset CSV (adaptive_scheduler_dataset_10k)")
    p.add_argument("--out", required=True, help="Base path to save models")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--plots_dir", default="plots", help="Directory to save plots")
    p.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization (slower)")
    return p.parse_args()


def encode_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-hot encode categorical column."""
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return df


def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create comprehensive visualization of results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Main metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy Comparison', 'Precision Comparison', 'Recall Comparison', 'F1-Score Comparison']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        if metric in results_df.columns:
            data = results_df[metric].sort_values(ascending=False)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
            bars = ax.bar(range(len(data)), data.values, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylim([max(0, data.min() - 0.1), 1.0])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimized_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Feature importance comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Show top model
    best_model = results_df['accuracy'].idxmax()
    best_acc = results_df.loc[best_model, 'accuracy']
    
    ax.text(0.5, 0.5, f'BEST MODEL: {best_model}\n\nAccuracy: {best_acc:.2%}\n\nOptimized with real dataset features', 
            transform=ax.transAxes, fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_model_banner.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    args = parse_args()
    
    print("\n" + "=" * 90)
    print("OPTIMIZED ML TRAINING - ORIGINAL SCHEDULER DATASET")
    print("=" * 90)
    
    df = pd.read_csv(args.csv)
    
    # Use the ACTUAL features from the dataset
    feature_cols = ['avg_burst_time', 'arrival_rate', 'cpu_io_ratio', 'priority_variance', 'queue_length']
    target_col = 'best_scheduler'
    
    # Encode categorical 'throughput_req' as one-hot
    if 'throughput_req' in df.columns:
        df = encode_categorical(df, 'throughput_req')
        throughput_cols = [c for c in df.columns if c.startswith('throughput_req_')]
        feature_cols = feature_cols + throughput_cols
    
    X = df[feature_cols]
    y = df[target_col].astype(str)
    
    print(f"Dataset: {len(df)} samples, {len(feature_cols)} features, {y.nunique()} classes")
    
    # Feature scaling (important for KNN, SVM, Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    print(f"Split: {X_train.shape[0]} train, {X_test.shape[0]} test | Feature scaling applied\n")
    models: Dict[str, Any] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,          # Increased from 200
            max_depth=20,               # Increased from 15
            min_samples_split=2,        # Reduced from 5 for more flexibility
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=args.random_state,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=args.random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,           # Increased from 100
            max_depth=7,                 # Increased from 5
            learning_rate=0.05,          # Reduced for better convergence
            subsample=0.8,
            random_state=args.random_state
        ),
        "XGBoost": None,  # Will try to add
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=2,
            class_weight='balanced',
            random_state=args.random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,              # Increased from 5
            weights='distance',          # Use distance weighting
            metric='euclidean'
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,           # Increased from 100
            learning_rate=0.5,          # Reduced for stability
            random_state=args.random_state
        ),
        "SVM (RBF)": SVC(
            kernel='rbf',
            C=10.0,                      # Increased regularization
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=args.random_state
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=2000,              # Increased from 1000
            solver='lbfgs',
            multi_class='multinomial',
            random_state=args.random_state
        ),
        "Naive Bayes": GaussianNB(),
    }
    
    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=args.random_state
        )
    except ImportError:
        del models["XGBoost"]
    
    print(f"Training {len(models)} optimized algorithms...\n" + "=" * 90)
    
    labels = sorted(y.unique())
    base_out = args.out
    root, ext = os.path.splitext(base_out)
    if not ext:
        ext = ".joblib"
    
    # Encode labels for XGBoost (requires numeric labels)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    results: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}
    
    # Train each model
    for name, model in models.items():
        print(f"{name:25s} → ", end='', flush=True)
        
        # Use encoded labels for XGBoost, original for others
        y_train_use = y_train_encoded if name == "XGBoost" else y_train
        y_test_use = y_test_encoded if name == "XGBoost" else y_test
        
        start_train = time.time()
        model.fit(X_train, y_train_use)
        train_time = time.time() - start_train
        
        start_pred = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_pred
        
        # Decode predictions for XGBoost back to original labels
        if name == "XGBoost":
            y_pred = le.inverse_transform(y_pred)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train_use, cv=5, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"✓ Accuracy: {acc*100:.2f}%  |  CV: {cv_mean*100:.2f}%  |  F1: {f1:.4f}  |  Time: {train_time:.2f}s")
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'train_time': train_time,
            'predict_time': predict_time
        }
        
        # Save model
        out_path = f"{root}_optimized_{name.replace(' ', '_').lower()}{ext}"
        save_model(model, out_path)
        
        trained_models[name] = model
    
    print("\n" + "=" * 90)
    
    results_df = pd.DataFrame(results).T
    
    # Save detailed CSV report
    csv_path = os.path.join(args.plots_dir, 'optimized_results.csv')
    os.makedirs(args.plots_dir, exist_ok=True)
    results_df.to_csv(csv_path)
    
    # Generate plots
    plot_metrics_comparison(results_df, args.plots_dir)
    
    # Create summary report
    report_path = os.path.join(args.plots_dir, 'optimized_model_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("OPTIMIZED MODEL COMPARISON REPORT\n")
        f.write("=" * 90 + "\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"  • Dataset: Original scheduler dataset (not synthetic)\n")
        f.write(f"  • Features: {', '.join(feature_cols)}\n")
        f.write(f"  • Samples: {len(df)} total ({len(X_train)} train, {len(X_test)} test)\n")
        f.write(f"  • Feature Scaling: StandardScaler applied\n")
        f.write(f"  • Optimizations: Enhanced hyperparameters, cross-validation\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("RANKINGS BY ACCURACY:\n")
        f.write("=" * 90 + "\n\n")
        
        sorted_acc = results_df.sort_values('accuracy', ascending=False)
        for idx, (name, row) in enumerate(sorted_acc.iterrows(), 1):
            f.write(f"{idx}. {name:25s}  Accuracy: {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)  ")
            f.write(f"CV: {row['cv_accuracy']:.4f} ± {row['cv_std']:.4f}\n")
        
        f.write("\n" + "=" * 90 + "\n")
        f.write("DETAILED METRICS:\n")
        f.write("=" * 90 + "\n\n")
        f.write(results_df.to_string())
        
        f.write("\n\n" + "=" * 90 + "\n")
        f.write("KEY IMPROVEMENTS:\n")
        f.write("=" * 90 + "\n\n")
        f.write("1. Using ORIGINAL dataset features (not synthetic mapping)\n")
        f.write("2. Feature scaling with StandardScaler\n")
        f.write("3. Optimized hyperparameters:\n")
        f.write("   - More trees (300 for Random Forest/Extra Trees)\n")
        f.write("   - Deeper trees (max_depth=20)\n")
        f.write("   - Better regularization\n")
        f.write("   - Distance-weighted KNN\n")
        f.write("4. Cross-validation for robust estimates\n")
        f.write("5. Class balancing maintained\n\n")
        
        best_model = results_df['accuracy'].idxmax()
        best_acc = results_df.loc[best_model, 'accuracy']
        best_cv = results_df.loc[best_model, 'cv_accuracy']
        
        f.write(f"BEST MODEL: {best_model}\n")
        f.write(f"   Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
        f.write(f"   CV Accuracy:   {best_cv:.4f}\n")
        f.write(f"   This represents the optimal performance on your scheduler dataset.\n")
    
    print("\n" + "=" * 90)
    print("TRAINING COMPLETE!")
    print("=" * 90)
    
    # Show top 5 in compact format
    print("\nTop 5 Models:")
    for idx, (name, row) in enumerate(results_df.nlargest(5, 'accuracy').iterrows(), 1):
        print(f"  {idx}. {name:20s} → Accuracy: {row['accuracy']*100:6.2f}%  |  CV: {row['cv_accuracy']*100:6.2f}%  |  F1: {row['f1_score']:.4f}")
    
    best_model = results_df['accuracy'].idxmax()
    best_acc = results_df.loc[best_model, 'accuracy']
    print(f"\nBEST MODEL: {best_model} ({best_acc*100:.2f}% accuracy)")
    print(f"\nResults: {args.plots_dir}/")
    print(f"  • optimized_model_report.txt")
    print(f"  • optimized_results.csv")
    print(f"  • optimized_metrics_comparison.png")
    print(f"\nModels: {os.path.dirname(root)}/optimized_model_*.joblib")


if __name__ == "__main__":
    main()
