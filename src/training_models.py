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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import to_undirected

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

"""
python training_models.py \
  --graph-file fakenews_graph_20251103_205558.pkl \
  --train-file dataset_train.csv \
  --val-file dataset_val.csv \
  --test-file dataset_test.csv \
  --hidden-dim 128 \
  --epochs 500
"""

# ============================================================================
# Utilities
# ============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Loading
# ============================================================================
class FakeNewsDataset:
    def __init__(self, graph_file, train_file, val_file, test_file):
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        self.G = graph_data['graph']

        self.train_df = pd.read_csv(train_file)
        self.val_df = pd.read_csv(val_file)
        self.test_df = pd.read_csv(test_file)

        print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")

        self._prepare_nodes()

    def _prepare_nodes(self):
        all_nodes = list(self.G.nodes())
        self.user_nodes = [n for n in all_nodes if self.G.nodes[n].get('node_type') == 'user']
        self.news_nodes = [n for n in all_nodes if self.G.nodes[n].get('node_type') == 'news']

        self.user_nodes.sort(key=str)
        self.news_nodes.sort(key=str)

        self.node_list = self.user_nodes + self.news_nodes
        self.node2idx = {node: idx for idx, node in enumerate(self.node_list)}

        print(f"Users: {len(self.user_nodes)} | News: {len(self.news_nodes)}")

    def to_pyg_data(self):
        print("\nBuilding PyG Data...")

        # Features
        features = []
        for node in self.node_list:
            node_data = self.G.nodes[node]
            if node_data.get('node_type') == 'user':
                feat = [
                    float(node_data.get('gossipcop_fake', 0)),
                    float(node_data.get('gossipcop_real', 0)),
                    float(node_data.get('politifact_fake', 0)),
                    float(node_data.get('politifact_real', 0)),
                    float(node_data.get('total_shares', 0)),
                ]
            else:
                feat = [0.0, 0.0, 0.0, 0.0, 0.0]
            features.append(feat)

        x = torch.tensor(features, dtype=torch.float32)

        # Edge index
        src_list, dst_list = [], []
        for u, v in self.G.edges():
            u_idx, v_idx = self.node2idx.get(u), self.node2idx.get(v)
            if u_idx is not None and v_idx is not None:
                src_list.append(u_idx)
                dst_list.append(v_idx)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_index = to_undirected(edge_index, num_nodes=len(self.node_list))

        # Labels and masks
        num_nodes = len(self.node_list)
        y = torch.full((num_nodes,), -1, dtype=torch.long)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for df, mask in [(self.train_df, train_mask),
                         (self.val_df, val_mask),
                         (self.test_df, test_mask)]:
            news_labels = df.groupby('news_id')['label'].first()
            for news_id, label in news_labels.items():
                idx = self.node2idx.get(str(news_id))
                if idx is not None and self.G.nodes[str(news_id)].get('node_type') == 'news':
                    y[idx] = int(label)
                    mask[idx] = True

        data = Data(x=x, edge_index=edge_index, y=y,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        print(f"✓ Nodes: {data.num_nodes} | Edges: {data.num_edges} | Features: {data.num_node_features}")
        print(f"✓ Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

        return data


# ============================================================================
# Models: One GCN, One GAT, One GraphSAGE
# ============================================================================
class GCN(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, dropout=0.5, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv5 = GCNConv(hidden_dim // 2, num_classes)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)
        self.model_name = "GCN"

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, dropout=0.2, heads=4, num_classes=2):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"

        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.res1 = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv3 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout)

        self.model_name = "GAT"

    def forward(self, x, edge_index):
        # Block 1
        res = self.res1(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = x + res

        # Block 2
        res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = x + res

        # Block 3
        res = x
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = x + res

        # Output logits
        x = self.out(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, dropout=0.5, num_classes=2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.conv5 = SAGEConv(hidden_dim // 2, num_classes)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)

        self.dropout = nn.Dropout(dropout)
        self.model_name = "GraphSAGE"

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index)
        return x


def create_model(model_type, in_dim=5, hidden_dim=128, dropout=0.5, gat_heads=4):
    if model_type == 'gcn':
        return GCN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == 'gat':
        return GAT(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout, heads=gat_heads)
    elif model_type in ['sage', 'graphsage']:
        return GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_type}")


# ============================================================================
# Training
# ============================================================================
@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    logits = logits[mask]
    y_true = data.y[mask].to(device)

    # Handle empty mask safely
    if y_true.numel() == 0:
        return ({'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0,
                 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}, np.array([]), np.array([]))

    loss = nn.CrossEntropyLoss()(logits, y_true).item()
    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = (probs >= 0.5).long()

    y_true_np = y_true.cpu().numpy()
    probs_np = probs.cpu().numpy()
    preds_np = preds.cpu().numpy()

    auc = roc_auc_score(y_true_np, probs_np) if len(np.unique(y_true_np)) > 1 else 0.0

    metrics = {
        'loss': loss,
        'accuracy': accuracy_score(y_true_np, preds_np),
        'precision': precision_score(y_true_np, preds_np, zero_division=0),
        'recall': recall_score(y_true_np, preds_np, zero_division=0),
        'f1': f1_score(y_true_np, preds_np, zero_division=0),
        'auc': auc
    }

    return metrics, probs_np, preds_np


def train_epoch(model, data, train_mask, optimizer, device, class_weight=None, grad_clip=1.0):
    model.train()
    logits = model(data.x.to(device), data.edge_index.to(device))
    logits = logits[train_mask]
    y_true = data.y[train_mask].to(device)

    if y_true.numel() == 0:
        return 0.0

    if class_weight is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    loss = loss_fn(logits, y_true)

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item()


def train_model(model, data, device, epochs=500, lr=0.001, weight_decay=5e-4,
                patience=50, class_weight=None):
    print(f"\n{'=' * 70}")
    print(f"TRAINING {model.model_name}")
    print(f"{'=' * 70}")

    model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=15, verbose=False, min_lr=1e-6)

    best_val_auc = -1
    best_epoch = -1
    best_state = None
    patience_counter = 0

    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, data, data.train_mask, optimizer, device, class_weight)
        train_metrics, _, _ = evaluate(model, data, data.train_mask, device)
        val_metrics, _, _ = evaluate(model, data, data.val_mask, device)

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])

        scheduler.step(val_metrics['auc'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f} | "
                  f"Train AUC: {train_metrics['auc']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f}")

        # Uncomment to enable early stopping
        # if patience_counter >= patience:
        #     print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
        #     break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"✓ Best model from epoch {best_epoch} (Val AUC: {best_val_auc:.4f})")

    return model, history


# ============================================================================
# Visualization
# ============================================================================
def plot_combined_roc(results_dict, output_file):
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    plt.figure(figsize=(10, 8))

    for idx, (model_name, result) in enumerate(results_dict.items()):
        if len(result['y_true']) == 0:
            continue
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_prob'])
        auc = roc_auc_score(result['y_true'], result['y_prob'])
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], linewidth=2.5,
                 label=f'{model_name} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves - All Models', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC saved: {output_file}")


def plot_combined_training_curves(all_histories, output_file):
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (model_name, history) in enumerate(all_histories.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        color = colors[idx % len(colors)]

        ax1.plot(epochs, history['train_loss'], color=color, linewidth=2, label=model_name)
        ax2.plot(epochs, history['val_loss'], color=color, linewidth=2, label=model_name)
        ax3.plot(epochs, history['train_auc'], color=color, linewidth=2, label=model_name)
        ax4.plot(epochs, history['val_auc'], color=color, linewidth=2, label=model_name)

    for ax, title in [(ax1, 'Training Loss'), (ax2, 'Validation Loss'),
                      (ax3, 'Training AUC'), (ax4, 'Validation AUC')]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title.split()[1], fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved: {output_file}")


def save_results(all_results, output_file):
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"{'Model':<15} {'Loss':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}\n")
        f.write("-" * 70 + "\n")

        for model_name, result in all_results.items():
            m = result['test_metrics']
            f.write(f"{model_name:<15} {m['loss']:<8.4f} {m['accuracy']:<8.4f} "
                    f"{m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1']:<8.4f} {m['auc']:<8.4f}\n")

    print(f"✓ Results saved: {output_file}")


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-file', required=True)
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--val-file', required=True)
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--models', nargs='+', choices=['gcn', 'gat', 'sage'],
                        default=['gcn', 'gat', 'sage'])
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gat-heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', '-o', default='results')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 70}")
    print("MULTI-MODEL TRAINING")
    print(f"{'=' * 70}")
    print(f"Models: {', '.join([m.upper() for m in args.models])}")
    print(f"Device: {args.device} | Hidden: {args.hidden_dim} | Epochs: {args.epochs}")

    # Load data
    dataset = FakeNewsDataset(args.graph_file, args.train_file,
                              args.val_file, args.test_file)
    data = dataset.to_pyg_data()

    # Class weights (avoid division by zero)
    train_labels = data.y[data.train_mask]
    counts = torch.bincount(train_labels[train_labels >= 0].cpu(), minlength=2)
    num_fake = counts[0].item()
    num_real = counts[1].item()
    class_weight = None
    if num_real > 0 and num_fake > 0:
        class_weight = torch.tensor([1.0, num_fake / max(1, num_real)], dtype=torch.float32)

    # Train models
    all_results = {}
    all_histories = {}

    for model_type in args.models:
        model = create_model(model_type, in_dim=5, hidden_dim=args.hidden_dim,
                             dropout=args.dropout, gat_heads=args.gat_heads)
        model, history = train_model(model, data, args.device, args.epochs,
                                     args.lr, args.weight_decay, args.patience, class_weight)

        # Evaluate
        test_metrics, test_probs, test_preds = evaluate(model, data, data.test_mask, args.device)
        test_y_true = data.y[data.test_mask].cpu().numpy()

        all_results[model.model_name] = {
            'test_metrics': test_metrics,
            'y_true': test_y_true,
            'y_prob': test_probs
        }
        all_histories[model.model_name] = history

        print(f"\n{model.model_name} Test AUC: {test_metrics['auc']:.4f}")

    # Save visualizations
    roc_file = os.path.join(args.output_dir, f'combined_roc_{timestamp}.png')
    plot_combined_roc(all_results, roc_file)

    curves_file = os.path.join(args.output_dir, f'training_curves_{timestamp}.png')
    plot_combined_training_curves(all_histories, curves_file)

    results_file = os.path.join(args.output_dir, f'results_{timestamp}.txt')
    save_results(all_results, results_file)

    # Print comparison
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'AUC':<10} {'Accuracy':<10} {'F1':<10}")
    print("-" * 70)
    for name, result in all_results.items():
        m = result['test_metrics']
        print(f"{name:<15} {m['auc']:<10.4f} {m['accuracy']:<10.4f} {m['f1']:<10.4f}")


if __name__ == "__main__":
    main()