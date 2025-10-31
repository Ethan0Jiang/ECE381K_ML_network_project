import os
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

print("🔹 Step 1: Loading raw graph (.pkl)...")

pkl_path = r"C:\Users\32322\OneDrive\Desktop\MOMZI\transtocsv\transtocsv\fakenews_graph_20251027_130057.pkl"
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

G = data["graph"]
print(f"✅ Graph loaded: nodes={len(G.nodes())}, edges={len(G.edges())}")

# ===========================
# Build node features & labels
# ===========================
features = []
labels = []
node_mapping = {}

for i, (node, attr) in enumerate(G.nodes(data=True)):
    node_mapping[node] = i
    
    if attr.get("node_type") == "user":
        feat = [
            attr.get("trustworthiness", 0.5),
            attr.get("total_shares", 0),
            attr.get("fake_shares", 0),
            attr.get("real_shares", 0),
        ]
        label = -1  # user 不参与分类
    else:
        feat = [
            attr.get("num_shares", 0),
            attr.get("avg_user_trust", 0.5),
            attr.get("min_user_trust", 0.5),
            attr.get("max_user_trust", 0.5),
        ]
        label = 1 if attr.get("label") == "fake" else 0
    
    features.append(feat)
    labels.append(label)

x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

# ===========================
# Build edge_index
# ===========================
edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# ===========================
# Build PyG graph & Save
# ===========================
data_pyg = Data(x=x, edge_index=edge_index, y=y)
torch.save(data_pyg, "fakenews_graph_pyg.pt")

print("📁 Saved processed graph → fakenews_graph_pyg.pt")
print("✅ Step 1 Completed.\n")

# ====================================================
# Step 2: Train GCN using the saved PyG graph
# ====================================================
print("🔹 Step 2: Loading processed PT graph...")
data = torch.load("fakenews_graph_pyg.pt", weights_only=False)

# 用户节点 label = -1 → 设为 1 以不影响分类
data.y[data.y == -1] = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

num_features = data.x.size(1)
num_classes = len(torch.unique(data.y))

print(f"✅ Using device: {device}, features={num_features}, classes={num_classes}")

# ===========================
# GCN Model
# ===========================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.dropout = 0.2
        self.out = torch.nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.out(x)
        return x

model = GCN(num_features, 64, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# ===========================
# Train & Evaluate
# ===========================
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    return (pred == data.y).float().mean().item()

print("🚀 Training GCN...")
losses, accs = [], []
for epoch in range(1, 201):
    loss = train()
    acc = evaluate()
    losses.append(loss)
    accs.append(acc)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss={loss:.4f} | Acc={acc:.4f}")

# Save model
torch.save(model.state_dict(), "GCN_boost_v1.pt")
print("💾 Saved model → GCN_boost_v1.pt")

# Plot
plt.plot(losses, label="Loss", color="red")
plt.plot(accs, label="Accuracy", color="blue")
plt.title("GCN Training Curve")
plt.legend()
plt.savefig("gcn_training_curve.png")
plt.show()

print("\n🎉 All done! Graph saved + Model trained + Curve plotted.")
