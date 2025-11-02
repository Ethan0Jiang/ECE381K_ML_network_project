import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ====== PyG (required) ======
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import to_undirected
except ImportError:
    raise SystemExit(
        "Requires PyTorch Geometric.\n"
        "Install per https://pytorch-geometric.readthedocs.io/ "
        "(match torch/cuda versions)."
    )

# ====== metrics (optional but recommended) ======
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_get(d, key, default=None):
    return d[key] if key in d else default


# ----------------------------
# Feature engineering
# ----------------------------
def build_node_features(G, node_list):
    # Degree normalization for both user & news
    degrees = {n: G.degree(n) for n in node_list}
    max_deg = max(1, max(degrees.values()))

    feats = []
    for n in node_list:
        d = G.nodes[n]
        node_type = d.get("node_type", "user")

        # universal structural flags
        is_user = 1.0 if node_type == "user" else 0.0
        is_news = 1.0 if node_type == "news" else 0.0
        deg_norm = np.log1p(degrees[n]) / np.log1p(max_deg)

        if is_user:
            # user has 5-dimensional behavioral info
            total_shares      = float(d.get("total_shares", 0))
            politifact_fake   = float(d.get("politifact_fake", 0))
            politifact_real   = float(d.get("politifact_real", 0))
            gossipcop_fake    = float(d.get("gossipcop_fake", 0))
            gossipcop_real    = float(d.get("gossipcop_real", 0))

            # users do not have source
            src_gossip = 0.0
            src_politi = 0.0

        else:  # news node
            # news do not have these share stats → set 0
            total_shares = politifact_fake = politifact_real = gossipcop_fake = gossipcop_real = 0.0

            # news does have source
            source = str(d.get("source", "")).lower()
            src_gossip = 1.0 if source == "gossipcop" else 0.0
            src_politi = 1.0 if source == "politifact" else 0.0

        # final 10-dimensional vector
        vec = [
            is_user, is_news, deg_norm,
            total_shares, politifact_fake, politifact_real,
            gossipcop_fake, gossipcop_real,
            src_gossip, src_politi
        ]
        feats.append(vec)

    x = torch.tensor(feats, dtype=torch.float32)
    return x


# ----------------------------
# Splits & labels
# ----------------------------
def load_splits(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        if 'node_id' not in df.columns:
            raise ValueError(f"{name} CSV missing 'node_id'")
        if 'label' not in df.columns:
            raise ValueError(f"{name} CSV missing 'label'")

    splits = {
        'train': set(train_df['node_id'].astype(str).tolist()),
        'val':   set(val_df['node_id'].astype(str).tolist()),
        'test':  set(test_df['node_id'].astype(str).tolist()),
    }
    all_ids = splits['train'] | splits['val'] | splits['test']
    return splits, all_ids


def build_label_and_masks(G, node_list, splits, news_id_set):
    """
    Only NEWS nodes get labels.
    Masks mark only the news nodes that appear in each split CSV.
    """
    N = len(node_list)
    id2idx = {nid: i for i, nid in enumerate(node_list)}

    y = torch.full((N,), fill_value=-1, dtype=torch.long)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    # Use labels from the CSVs (consistent with your split generator)
    # We read them again here so we don't rely on graph attributes.
    def set_labels_from_csv(csv_path, target_mask):
        df = pd.read_csv(csv_path)
        for nid, label in zip(df['node_id'].astype(str), df['label'].astype(int)):
            idx = id2idx.get(nid)
            if idx is None:
                continue
            if G.nodes[nid].get('node_type') == 'news':
                y[idx] = int(label)  # 0 fake, 1 real
                target_mask[idx] = True

    # Build masks & labels
    set_labels_from_csv(args.train_csv, train_mask)
    set_labels_from_csv(args.val_csv,   val_mask)
    set_labels_from_csv(args.test_csv,  test_mask)

    return y, train_mask, val_mask, test_mask


# ----------------------------
# Graph → PyG
# ----------------------------
def build_edge_index(G, node_list):
    id2idx = {n: i for i, n in enumerate(node_list)}
    src, dst = [], []
    for u, v in G.edges():
        ui = id2idx.get(u)
        vi = id2idx.get(v)
        if ui is None or vi is None:
            continue
        src.append(ui); dst.append(vi)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(node_list))
    return edge_index


# ----------------------------
# Model
# ----------------------------
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x  # logits


# ----------------------------
# Training / Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    device = next(model.parameters()).device

    logits = model(data.x.to(device), data.edge_index.to(device))
    logits = logits[mask]
    y_true = data.y[mask].to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y_true)

    y_prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(np.int64)
    y_true_np = y_true.detach().cpu().numpy()

    metrics = {
        'loss': float(loss.item()),
        'acc': float((y_pred == y_true_np).mean()),
    }
    if SKLEARN_AVAILABLE:
        try:
            metrics['auroc'] = float(roc_auc_score(y_true_np, y_prob))
        except Exception:
            metrics['auroc'] = float('nan')
        try:
            metrics['ap'] = float(average_precision_score(y_true_np, y_prob))
        except Exception:
            metrics['ap'] = float('nan')
        try:
            metrics['f1'] = float(f1_score(y_true_np, y_pred))
        except Exception:
            metrics['f1'] = float('nan')
    return metrics, y_prob


def train_loop(model, data, train_mask, val_mask, epochs=200, lr=1e-2, weight_decay=5e-4,
               patience=20, class_weight=None, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=(class_weight.to(device) if class_weight is not None else None))

    best_score = -1e9
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x.to(device), data.edge_index.to(device))
        loss = loss_fn(logits[train_mask], data.y[train_mask].to(device))
        loss.backward()
        optimizer.step()

        val_metrics, _ = evaluate(model, data, val_mask)
        score = val_metrics.get('auroc', None)
        if score is None or np.isnan(score):
            score = -val_metrics['loss']  # fall back

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == 1:
            msg = (f"[Epoch {epoch:03d}] "
                   f"TrainLoss={loss.item():.4f} | "
                   f"ValLoss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.4f}")
            if 'auroc' in val_metrics:
                msg += f", AUROC={val_metrics['auroc']:.4f}, AP={val_metrics.get('ap', float('nan')):.4f}"
            print(msg)

        if epoch - best_epoch >= patience:
            print(f"Early stopping at {epoch}, best epoch {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train GCN on FakeNews graph (news classification)")
    p.add_argument("--graph-pkl", type=str, required=True)
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, required=True)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", type=str, default="runs_gcn")
    return p.parse_args()


def compute_class_weight(y, mask):
    cls = y[mask].numpy()
    pos = (cls == 1).sum()
    neg = (cls == 0).sum()
    if pos > 0 and neg > 0:
        return torch.tensor([1.0, max(1.0, neg / max(1, pos))], dtype=torch.float32)
    return None


def main():
    global args
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load graph
    print(f"Loading graph from {args.graph_pkl} ...")
    with open(args.graph_pkl, "rb") as f:
        graph_data = pickle.load(f)
    G = graph_data['graph']
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2) Splits
    splits, _ = load_splits(args.train_csv, args.val_csv, args.test_csv)

    # 3) Node list (deterministic)
    node_list = list(G.nodes())
    node_list.sort(key=str)

    # 4) Features
    x = build_node_features(G, node_list)

    # 5) Edge index
    edge_index = build_edge_index(G, node_list)

    # 6) Labels + masks (labels from CSVs)
    y, train_mask, val_mask, test_mask = build_label_and_masks(G, node_list, splits, None)

    # 7) Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 8) Class weights (imbalance)
    class_weight = compute_class_weight(data.y, data.train_mask)

    # 9) Train
    model = GCN(in_dim=data.x.size(1),
                hidden_dim=args.hidden_dim,
                out_dim=2,
                dropout=args.dropout)

    model = train_loop(model, data, data.train_mask, data.val_mask,
                       epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                       patience=args.patience, class_weight=class_weight, device=args.device)

    # 10) Final evaluation
    print("\n==== Final Evaluation ====")
    for name, m in [('Train', data.train_mask), ('Val', data.val_mask), ('Test', data.test_mask)]:
        metrics, _ = evaluate(model, data, m)
        line = f"{name}: loss={metrics['loss']:.4f}, acc={metrics['acc']:.4f}"
        if 'auroc' in metrics:
            line += f", auroc={metrics['auroc']:.4f}, ap={metrics.get('ap', float('nan')):.4f}, f1={metrics.get('f1', float('nan')):.4f}"
        print(line)

    # 11) Save model
    model_path = os.path.join(args.outdir, "best_gcn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved best model to: {model_path}")

    # 12) Export test predictions
    model.eval()
    device = args.device
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    node_id_list = np.array(node_list)
    test_idx = data.test_mask.cpu().numpy()
    out_df = pd.DataFrame({
        "node_id": node_id_list[test_idx],
        "label": data.y[test_idx].cpu().numpy(),
        "prob_real": prob[test_idx]
    })
    out_csv = os.path.join(args.outdir, "test_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved test predictions to: {out_csv}")


if __name__ == "__main__":
    main()
