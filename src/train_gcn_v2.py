import argparse
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


'''
python train_gcn_v2.py     --graph-file fakenews_graph_202511
03_205558.pkl     --train-file dataset_train.csv     --val-file dataset_val.csv     --test-file dataset_test.csv 

'''

# PyTorch Geometric
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import to_undirected
except ImportError:
    raise SystemExit(
        "Requires PyTorch Geometric.\n"
        "Install: pip install torch-geometric\n"
        "Or follow: https://pytorch-geometric.readthedocs.io/"
    )

# Metrics
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================================
# Utilities
# ============================================================================
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Loading & Preparation
# ============================================================================
class FakeNewsDataset:
    """
    Load graph and edge-based datasets for GCN training.
    
    Dataset format (from create_dataset_splits_NN_v2.py):
    - Each row = one user-news edge (training example)
    - Features (5): gossipcop_fake, gossipcop_real, politifact_fake, 
                    politifact_real, total_shares
    - Label: 0=fake, 1=real (news label)
    """
    
    def __init__(self, graph_file, train_file, val_file, test_file):
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load graph
        print(f"\nLoading graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.G = graph_data['graph']
        print(f"  ✓ Graph loaded: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")
        
        # Load edge datasets
        print(f"\nLoading datasets...")
        self.train_df = pd.read_csv(train_file)
        self.val_df = pd.read_csv(val_file)
        self.test_df = pd.read_csv(test_file)
        
        print(f"  ✓ Train: {len(self.train_df)} edges")
        print(f"  ✓ Val:   {len(self.val_df)} edges")
        print(f"  ✓ Test:  {len(self.test_df)} edges")
        
        # Verify required columns
        required_cols = ['user_id', 'news_id', 'gossipcop_fake', 'gossipcop_real',
                        'politifact_fake', 'politifact_real', 'total_shares', 'label']
        for df, name in [(self.train_df, 'train'), (self.val_df, 'val'), (self.test_df, 'test')]:
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"{name} CSV missing columns: {missing}")
        
        # Get all unique nodes
        self._prepare_node_info()
        
    def _prepare_node_info(self):
        """Prepare node lists and mappings"""
        # Get all nodes from graph
        all_nodes = list(self.G.nodes())
        
        # Separate users and news
        self.user_nodes = []
        self.news_nodes = []
        
        for node in all_nodes:
            node_type = self.G.nodes[node].get('node_type', 'unknown')
            if node_type == 'user':
                self.user_nodes.append(node)
            elif node_type == 'news':
                self.news_nodes.append(node)
        
        # Sort for deterministic ordering
        self.user_nodes.sort(key=str)
        self.news_nodes.sort(key=str)
        
        # Combined node list (users first, then news)
        self.node_list = self.user_nodes + self.news_nodes
        
        # Create node ID to index mapping
        self.node2idx = {node: idx for idx, node in enumerate(self.node_list)}
        
        print(f"\nNode statistics:")
        print(f"  User nodes: {len(self.user_nodes)}")
        print(f"  News nodes: {len(self.news_nodes)}")
        print(f"  Total nodes: {len(self.node_list)}")
        
        # Get news nodes in each split
        self.train_news = set(self.train_df['news_id'].astype(str))
        self.val_news = set(self.val_df['news_id'].astype(str))
        self.test_news = set(self.test_df['news_id'].astype(str))
        
        print(f"\nNews nodes per split:")
        print(f"  Train: {len(self.train_news)} unique news")
        print(f"  Val:   {len(self.val_news)} unique news")
        print(f"  Test:  {len(self.test_news)} unique news")
    
    def build_features(self):
        """
        Build node features using ONLY the 5 user features.
        
        User features (5D): gossipcop_fake, gossipcop_real, politifact_fake,
                           politifact_real, total_shares
        News features (5D): all zeros (news nodes don't have these features)
        
        Returns: torch.Tensor of shape [num_nodes, 5]
        """
        print("\nBuilding node features (5D per node)...")
        
        features = []
        
        for node in self.node_list:
            node_data = self.G.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            
            if node_type == 'user':
                # User: extract 5 features
                feat = [
                    float(node_data.get('gossipcop_fake', 0)),
                    float(node_data.get('gossipcop_real', 0)),
                    float(node_data.get('politifact_fake', 0)),
                    float(node_data.get('politifact_real', 0)),
                    float(node_data.get('total_shares', 0))
                ]
            else:
                # News: all zeros (no user features)
                feat = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            features.append(feat)
        
        x = torch.tensor(features, dtype=torch.float32)
        
        print(f"  ✓ Feature matrix shape: {x.shape}")
        print(f"  ✓ Feature range: [{x.min():.2f}, {x.max():.2f}]")
        
        return x
    
    def build_edge_index(self):
        """Build edge index for PyG"""
        print("\nBuilding edge index...")
        
        src_list = []
        dst_list = []
        
        for u, v in self.G.edges():
            u_idx = self.node2idx.get(u)
            v_idx = self.node2idx.get(v)
            
            if u_idx is None or v_idx is None:
                continue
            
            src_list.append(u_idx)
            dst_list.append(v_idx)
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        # Make undirected (add reverse edges)
        edge_index = to_undirected(edge_index, num_nodes=len(self.node_list))
        
        print(f"  ✓ Edge index shape: {edge_index.shape}")
        print(f"  ✓ Number of edges: {edge_index.shape[1]}")
        
        return edge_index
    
    def build_labels_and_masks(self):
        """
        Build labels and masks for news nodes only.
        
        Returns:
            y: labels for all nodes (-1 for users, 0/1 for news)
            train_mask, val_mask, test_mask: boolean masks
        """
        print("\nBuilding labels and masks...")
        
        num_nodes = len(self.node_list)
        
        # Initialize
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Helper function to set labels from dataframe
        def set_labels_from_df(df, mask):
            # Get unique news and their labels
            news_labels = df.groupby('news_id')['label'].first()
            
            for news_id, label in news_labels.items():
                news_id_str = str(news_id)
                idx = self.node2idx.get(news_id_str)
                
                if idx is None:
                    continue
                
                # Verify it's a news node
                if self.G.nodes[news_id_str].get('node_type') == 'news':
                    y[idx] = int(label)
                    mask[idx] = True
        
        # Set labels and masks for each split
        set_labels_from_df(self.train_df, train_mask)
        set_labels_from_df(self.val_df, val_mask)
        set_labels_from_df(self.test_df, test_mask)
        
        print(f"  Train mask: {train_mask.sum()} news nodes")
        print(f"  Val mask:   {val_mask.sum()} news nodes")
        print(f"  Test mask:  {test_mask.sum()} news nodes")
        
        # Print label distribution
        for name, mask in [('Train', train_mask), ('Val', val_mask), ('Test', test_mask)]:
            labels = y[mask]
            if len(labels) > 0:
                fake = (labels == 0).sum().item()
                real = (labels == 1).sum().item()
                print(f"  {name} labels - Fake: {fake}, Real: {real}")
        
        return y, train_mask, val_mask, test_mask
    
    def to_pyg_data(self):
        """Convert to PyTorch Geometric Data object"""
        print("\n" + "="*70)
        print("CREATING PYTORCH GEOMETRIC DATA")
        print("="*70)
        
        x = self.build_features()
        edge_index = self.build_edge_index()
        y, train_mask, val_mask, test_mask = self.build_labels_and_masks()
        
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        print(f"\n✓ PyG Data created successfully")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.num_node_features}")
        
        return data


# ============================================================================
# GCN Model
# ============================================================================
class GCN(nn.Module):
    """
    Graph Convolutional Network for fake news detection.
    
    Architecture:
    - Input: 5D node features
    - GCN layer 1: 5 -> hidden_dim
    - ReLU + Dropout
    - GCN layer 2: hidden_dim -> hidden_dim
    - ReLU + Dropout
    - GCN layer 3: hidden_dim -> 2 (binary classification)
    """
    
    def __init__(self, in_dim=5, hidden_dim=64, dropout=0.5):
        super().__init__()
        
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 2)  # 2 classes: fake/real
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        print(f"\nGCN Architecture:")
        print(f"  Input:  {in_dim} features")
        print(f"  Layer 1: GCN({in_dim} -> {hidden_dim})")
        print(f"  Layer 2: GCN({hidden_dim} -> {hidden_dim})")
        print(f"  Layer 3: GCN({hidden_dim} -> 2)")
        print(f"  Dropout: {dropout}")
    
    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 3 (output)
        x = self.conv3(x, edge_index)
        
        return x  # logits (no softmax, using CrossEntropyLoss)


# ============================================================================
# Training & Evaluation
# ============================================================================
@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate model on given mask"""
    model.eval()
    
    # Forward pass
    logits = model(data.x.to(device), data.edge_index.to(device))
    logits = logits[mask]
    y_true = data.y[mask].to(device)
    
    # Loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y_true).item()
    
    # Predictions
    probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1 (real)
    preds = (probs >= 0.5).long()
    
    # Convert to numpy
    y_true_np = y_true.cpu().numpy()
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # Compute metrics
    metrics = {
        'loss': loss,
        'accuracy': accuracy_score(y_true_np, preds_np),
        'precision': precision_score(y_true_np, preds_np, zero_division=0),
        'recall': recall_score(y_true_np, preds_np, zero_division=0),
        'f1': f1_score(y_true_np, preds_np, zero_division=0),
    }
    
    # AUC (requires at least 2 classes)
    if len(np.unique(y_true_np)) > 1:
        metrics['auc'] = roc_auc_score(y_true_np, probs_np)
    else:
        metrics['auc'] = float('nan')
    
    return metrics, probs_np, preds_np


def train_epoch(model, data, train_mask, optimizer, device, class_weight=None):
    """Train for one epoch"""
    model.train()
    
    # Forward pass
    logits = model(data.x.to(device), data.edge_index.to(device))
    logits = logits[train_mask]
    y_true = data.y[train_mask].to(device)
    
    # Loss
    if class_weight is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    loss = loss_fn(logits, y_true)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_model(model, data, device, epochs=200, lr=0.01, weight_decay=5e-4, 
                patience=20, class_weight=None):
    """Train the GCN model"""
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    model.to(device)
    data = data.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_auc = -1
    best_epoch = -1
    best_state = None
    patience_counter = 0
    
    train_losses = []
    val_aucs = []
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, data, data.train_mask, optimizer, device, class_weight)
        train_losses.append(train_loss)
        
        # Evaluate
        train_metrics, _, _ = evaluate(model, data, data.train_mask, device)
        val_metrics, _, _ = evaluate(model, data, data.val_mask, device)
        val_aucs.append(val_metrics['auc'])
        
        # Check if best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}] "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch})")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n✓ Loaded best model from epoch {best_epoch} (Val AUC: {best_val_auc:.4f})")
    
    return model, train_losses, val_aucs


# ============================================================================
# Visualization & Results
# ============================================================================
def plot_roc_curve(y_true, y_prob, output_file):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, label=f'GCN (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - GCN Fake News Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ ROC curve saved to: {output_file}")
    plt.close()


def plot_training_curves(train_losses, val_aucs, output_file):
    """Plot training loss and validation AUC curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax1.plot(train_losses, color='#A23B72', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Training Loss over Epochs', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Validation AUC
    ax2.plot(val_aucs, color='#2E86AB', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation AUC', fontsize=11)
    ax2.set_title('Validation AUC over Epochs', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_file}")
    plt.close()


def save_results(train_metrics, val_metrics, test_metrics, output_file):
    """Save results to text file"""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GCN FAKE NEWS DETECTION - RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model: Graph Convolutional Network (GCN)\n")
        f.write("Features: 5D user features (gossipcop_fake, gossipcop_real, "
                "politifact_fake, politifact_real, total_shares)\n")
        f.write("Task: Binary classification (fake=0, real=1)\n\n")
        
        for split_name, metrics in [('TRAIN', train_metrics), 
                                    ('VALIDATION', val_metrics), 
                                    ('TEST', test_metrics)]:
            f.write("="*70 + "\n")
            f.write(f"{split_name} SET\n")
            f.write("="*70 + "\n")
            f.write(f"Loss:      {metrics['loss']:.4f}\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
            f.write(f"AUC:       {metrics['auc']:.4f}\n\n")
    
    print(f"✓ Results saved to: {output_file}")


def print_final_results(train_metrics, val_metrics, test_metrics):
    """Print final results to console"""
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    # Create results table
    splits = ['Train', 'Validation', 'Test']
    metrics_list = [train_metrics, val_metrics, test_metrics]
    
    print(f"\n{'Split':<12} {'Loss':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    print("-" * 70)
    
    for split, metrics in zip(splits, metrics_list):
        print(f"{split:<12} "
              f"{metrics['loss']:<8.4f} "
              f"{metrics['accuracy']:<8.4f} "
              f"{metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} "
              f"{metrics['f1']:<8.4f} "
              f"{metrics['auc']:<8.4f}")
    
    print("\n" + "="*70)
    print(f"TEST AUC: {test_metrics['auc']:.4f}")
    print("="*70)


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GCN for fake news detection using user features'
    )
    
    # Data files
    parser.add_argument('--graph-file', type=str, required=True,
                       help='Path to graph pickle file')
    parser.add_argument('--train-file', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--val-file', type=str, required=True,
                       help='Path to validation CSV file')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test CSV file')
    
    # Model hyperparameters
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension size (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (default: 5e-4)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--output-dir', '-o', type=str, default='gcn_results',
                       help='Output directory (default: gcn_results)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*70)
    print("GCN FAKE NEWS DETECTION")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    dataset = FakeNewsDataset(
        args.graph_file,
        args.train_file,
        args.val_file,
        args.test_file
    )
    
    # Convert to PyG data
    data = dataset.to_pyg_data()
    
    # Compute class weights for imbalanced data
    train_labels = data.y[data.train_mask]
    num_fake = (train_labels == 0).sum().item()
    num_real = (train_labels == 1).sum().item()
    
    if num_fake > 0 and num_real > 0:
        # Weight inversely proportional to class frequency
        weight_real = num_fake / num_real
        class_weight = torch.tensor([1.0, weight_real], dtype=torch.float32)
        print(f"\nClass weights: [1.0 (fake), {weight_real:.2f} (real)]")
    else:
        class_weight = None
    
    # Create model
    model = GCN(
        in_dim=5,  # 5 user features
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    
    # Train
    model, train_losses, val_aucs = train_model(
        model, data, args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weight=class_weight
    )
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    train_metrics, train_probs, train_preds = evaluate(
        model, data, data.train_mask, args.device
    )
    val_metrics, val_probs, val_preds = evaluate(
        model, data, data.val_mask, args.device
    )
    test_metrics, test_probs, test_preds = evaluate(
        model, data, data.test_mask, args.device
    )
    
    # Print results
    print_final_results(train_metrics, val_metrics, test_metrics)
    
    # Save model
    model_file = os.path.join(args.output_dir, f'gcn_model_{timestamp}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'test_auc': test_metrics['auc']
    }, model_file)
    print(f"\n✓ Model saved to: {model_file}")
    
    # Save results
    results_file = os.path.join(args.output_dir, f'results_{timestamp}.txt')
    save_results(train_metrics, val_metrics, test_metrics, results_file)
    
    # Plot ROC curve (test set)
    test_y_true = data.y[data.test_mask].cpu().numpy()
    roc_file = os.path.join(args.output_dir, f'roc_curve_{timestamp}.png')
    plot_roc_curve(test_y_true, test_probs, roc_file)
    
    # Plot training curves
    curves_file = os.path.join(args.output_dir, f'training_curves_{timestamp}.png')
    plot_training_curves(train_losses, val_aucs, curves_file)
    
    # Save test predictions
    test_news_nodes = [dataset.node_list[i] for i in range(len(dataset.node_list)) 
                       if data.test_mask[i]]
    test_df = pd.DataFrame({
        'news_id': test_news_nodes,
        'true_label': test_y_true,
        'predicted_label': test_preds,
        'prob_real': test_probs
    })
    pred_file = os.path.join(args.output_dir, f'test_predictions_{timestamp}.csv')
    test_df.to_csv(pred_file, index=False)
    print(f"✓ Test predictions saved to: {pred_file}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTest AUC: {test_metrics['auc']:.4f}")
    print(f"All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
