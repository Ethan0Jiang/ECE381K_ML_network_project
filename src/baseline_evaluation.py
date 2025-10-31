import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

class BaselineEvaluator:
    def __init__(self, graph_file):
        """Load the saved graph from pickle file"""
        print(f"Loading graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.G = graph_data['graph']
        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def compute_news_trustworthiness_features(self, news_node_id):
        """
        Compute trustworthiness-based features for a news node
        by aggregating from connected user nodes
        
        Returns: dict with avg_trust, min_trust, max_trust, trust_std, high_trust_ratio
        """
        # Get all user neighbors
        neighbors = list(self.G.neighbors(news_node_id))
        user_neighbors = [n for n in neighbors if self.G.nodes[n]['node_type'] == 'user']
        
        if len(user_neighbors) == 0:
            # No user shares - return default values
            return {
                'avg_trust': 0.5,
                'min_trust': 0.5,
                'max_trust': 0.5,
                'trust_std': 0.0,
                'high_trust_ratio': 0.0,
                'num_users': 0
            }
        
        # Get trustworthiness scores from user nodes
        trust_scores = []
        for user_id in user_neighbors:
            user_data = self.G.nodes[user_id]
            trust = user_data.get('trustworthiness', 0.5)
            trust_scores.append(trust)
        
        trust_scores = np.array(trust_scores)
        
        # Compute features
        avg_trust = np.mean(trust_scores)
        min_trust = np.min(trust_scores)
        max_trust = np.max(trust_scores)
        trust_std = np.std(trust_scores)
        high_trust_ratio = np.sum(trust_scores >= 0.6) / len(trust_scores)
        
        return {
            'avg_trust': avg_trust,
            'min_trust': min_trust,
            'max_trust': max_trust,
            'trust_std': trust_std,
            'high_trust_ratio': high_trust_ratio,
            'num_users': len(user_neighbors)
        }
    
    def evaluate_on_split(self, split_df, split_name):
        """
        Evaluate baseline approach on a data split
        Uses three different metrics:
        1. avg_user_trust (average trustworthiness of sharing users)
        2. high_trust_ratio (ratio of high-trust users)
        3. max_user_trust (maximum trustworthiness among sharing users)
        """
        # Filter out news with no shares
        split_df_filtered = split_df[split_df['num_shares'] > 0].copy()
        
        print(f"\n{split_name} set:")
        print(f"  Total news: {len(split_df)}")
        print(f"  News with shares: {len(split_df_filtered)}")
        print(f"  News without shares (ignored): {len(split_df) - len(split_df_filtered)}")
        
        if len(split_df_filtered) == 0:
            print(f"  Warning: No news with shares in {split_name} set!")
            return None
        
        # Get ground truth labels
        y_true = split_df_filtered['label'].values  # 1=real, 0=fake
        
        # Compute trustworthiness features from the graph
        features_list = []
        for node_id in split_df_filtered['node_id']:
            features = self.compute_news_trustworthiness_features(node_id)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Method 1: Use avg_trust as predictor
        # Higher avg_trust -> predict real (1), lower -> predict fake (0)
        avg_trust_scores = features_df['avg_trust'].values
        
        # Method 2: Use high_trust_ratio as predictor
        high_trust_ratio_scores = features_df['high_trust_ratio'].values
        
        # Method 3: Use max_trust as predictor
        max_trust_scores = features_df['max_trust'].values
        
        # Threshold at 0.5 for binary predictions
        y_pred_avg = (avg_trust_scores >= 0.5).astype(int)
        y_pred_ratio = (high_trust_ratio_scores >= 0.5).astype(int)
        y_pred_max = (max_trust_scores >= 0.5).astype(int)
        
        # Compute metrics for all three methods
        results = {}
        
        for method_name, y_pred, scores in [
            ('avg_user_trust', y_pred_avg, avg_trust_scores),
            ('high_trust_ratio', y_pred_ratio, high_trust_ratio_scores),
            ('max_user_trust', y_pred_max, max_trust_scores)
        ]:
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, scores),
                'confusion_matrix': cm
            }
            results[method_name] = {
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred,
                'scores': scores
            }
        
        return results
    
    def format_confusion_matrix(self, cm):
        """Format confusion matrix as a nice text table"""
        lines = []
        lines.append("  Confusion Matrix:")
        lines.append("  " + "-" * 45)
        lines.append("                    Predicted")
        lines.append("                    Fake (0)    Real (1)")
        lines.append("  " + "-" * 45)
        lines.append(f"  Actual  Fake (0)  {cm[0][0]:6d}      {cm[0][1]:6d}")
        lines.append(f"          Real (1)  {cm[1][0]:6d}      {cm[1][1]:6d}")
        lines.append("  " + "-" * 45)
        
        # Add interpretation
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        lines.append(f"  True Negatives  (TN): {tn:6d}  (Correctly identified fake)")
        lines.append(f"  False Positives (FP): {fp:6d}  (Fake predicted as real)")
        lines.append(f"  False Negatives (FN): {fn:6d}  (Real predicted as fake)")
        lines.append(f"  True Positives  (TP): {tp:6d}  (Correctly identified real)")
        
        return "\n".join(lines)
    
    def save_results(self, train_results, val_results, test_results, output_file):
        """Save evaluation results to a text file"""
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BASELINE EVALUATION RESULTS\n")
            f.write("="*70 + "\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Method: Trustworthiness-based prediction (no training)\n")
            f.write("Approach: Predict real news when user trustworthiness >= 0.5\n\n")
            
            f.write("METRIC EXPLANATIONS:\n")
            f.write("-" * 70 + "\n")
            f.write("1. avg_user_trust (Average User Trust):\n")
            f.write("   - Mean trustworthiness across all users who shared the news\n")
            f.write("   - Balanced view of the overall crowd\n\n")
            f.write("2. high_trust_ratio (High Trust Ratio):\n")
            f.write("   - Proportion of users with trustworthiness >= 0.6\n")
            f.write("   - Focuses on the credible majority\n\n")
            f.write("3. max_user_trust (Maximum User Trust):\n")
            f.write("   - Highest trustworthiness among all sharers\n")
            f.write("   - Best case: at least one credible person shared it\n\n")
            
            for split_name, results in [('TRAIN', train_results), 
                                       ('VALIDATION', val_results), 
                                       ('TEST', test_results)]:
                if results is None:
                    continue
                
                f.write("="*70 + "\n")
                f.write(f"{split_name} SET RESULTS\n")
                f.write("="*70 + "\n\n")
                
                for method_name in ['avg_user_trust', 'high_trust_ratio', 'max_user_trust']:
                    method_results = results[method_name]
                    metrics = method_results['metrics']
                    
                    f.write(f"Method: {method_name}\n")
                    f.write("-"*70 + "\n")
                    f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                    f.write(f"  F1 Score:  {metrics['f1']:.4f}\n")
                    f.write(f"  AUC:       {metrics['auc']:.4f}\n\n")
                    
                    # Add confusion matrix
                    f.write(self.format_confusion_matrix(metrics['confusion_matrix']))
                    f.write("\n\n")
        
        print(f"\nResults saved to: {output_file}")
    
    def plot_roc_curves(self, test_results, output_file):
        """Plot ROC curves for all three methods on test set"""
        plt.figure(figsize=(8, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
        method_labels = {
            'avg_user_trust': 'Average User Trust',
            'high_trust_ratio': 'High Trust Ratio',
            'max_user_trust': 'Max User Trust'
        }
        
        for idx, method_name in enumerate(['avg_user_trust', 'high_trust_ratio', 'max_user_trust']):
            method_results = test_results[method_name]
            y_true = method_results['y_true']
            scores = method_results['scores']
            auc = method_results['metrics']['auc']
            
            fpr, tpr, _ = roc_curve(y_true, scores)
            
            plt.plot(fpr, tpr, color=colors[idx], linewidth=2.5, 
                    label=f'{method_labels[method_name]} (AUC = {auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {output_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Baseline evaluation: Predict fake/real news using user trustworthiness'
    )
    parser.add_argument('graph_file', type=str,
                       help='Path to the pickle file containing the graph')
    parser.add_argument('--train-file', type=str, required=True,
                       help='Path to training set CSV')
    parser.add_argument('--val-file', type=str, required=True,
                       help='Path to validation set CSV')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test set CSV')
    parser.add_argument('--output-prefix', '-o', type=str, default='baseline',
                       help='Prefix for output files (default: baseline)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = BaselineEvaluator(args.graph_file)
    
    # Load datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    test_df = pd.read_csv(args.test_file)
    
    print(f"Train: {len(train_df)} news nodes")
    print(f"Val:   {len(val_df)} news nodes")
    print(f"Test:  {len(test_df)} news nodes")
    
    # Evaluate on all splits
    print("\n" + "="*70)
    print("EVALUATING BASELINE APPROACH")
    print("="*70)
    
    train_results = evaluator.evaluate_on_split(train_df, "Train")
    val_results = evaluator.evaluate_on_split(val_df, "Validation")
    test_results = evaluator.evaluate_on_split(test_df, "Test")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.output_prefix}_results_{timestamp}.txt"
    evaluator.save_results(train_results, val_results, test_results, results_file)
    
    # Plot ROC curves for test set
    if test_results is not None:
        roc_file = f"{args.output_prefix}_roc_{timestamp}.pdf"
        evaluator.plot_roc_curves(test_results, roc_file)
    
    print("\n" + "="*70)
    print("BASELINE EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()