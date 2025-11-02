import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from collections import Counter

class DatasetCreator:
    def __init__(self, graph_file):
        print(f"Loading graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)

        self.G = graph_data['graph']
        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def get_news_nodes_data(self):
        news_data = []

        for node in self.G.nodes():
            d = self.G.nodes[node]
            if d.get('node_type') == 'news':
                news_data.append({
                    'node_id': node,
                    'ground_truth': d['ground_truth'],
                    'label': 1 if d['ground_truth'] == 'real' else 0,
                    'source': d.get('source', 'unknown'),
                    'title': d.get('title', ''),
                    'degree': self.G.degree(node)
                })

        return pd.DataFrame(news_data)

    def create_stratified_split(self, news_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):

        print("\n===== DATASET ORIGINAL DISTRIBUTION =====")
        print("Ground Truth:")
        print(news_df['ground_truth'].value_counts())
        print("\nSource:")
        print(news_df['source'].value_counts())

        news_df['stratify_key'] = news_df['ground_truth'] + "_" + news_df['source']

        train_df, temp_df = train_test_split(
            news_df,
            test_size=(val_ratio + test_ratio),
            stratify=news_df['stratify_key'],
            random_state=seed
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio/(val_ratio + test_ratio),
            stratify=temp_df['stratify_key'],
            random_state=seed
        )

        train_df = train_df.drop(columns=['stratify_key'])
        val_df = val_df.drop(columns=['stratify_key'])
        test_df = test_df.drop(columns=['stratify_key'])

        print("\n===== SPLIT STATISTICS =====")
        for name, df in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
            print(f"\n{name}: {len(df)} samples ({len(df)/len(news_df)*100:.1f}%)")
            print(df['ground_truth'].value_counts())
            print(df['source'].value_counts())

        print("\n===== DEGREE DISTRIBUTION =====")
        for name, df in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
            print(f"{name}: mean={df['degree'].mean():.2f}, median={df['degree'].median():.1f}, min={df['degree'].min()}, max={df['degree'].max()}")

        return train_df, val_df, test_df

    def save_outputs(self, train_df, val_df, test_df, prefix="dataset"):

        train_df.to_csv(f"{prefix}_train.csv", index=False)
        val_df.to_csv(f"{prefix}_val.csv", index=False)
        test_df.to_csv(f"{prefix}_test.csv", index=False)

        print("\nSaved:")
        print(f"{prefix}_train.csv")
        print(f"{prefix}_val.csv")
        print(f"{prefix}_test.csv")

        # Construct summary
        summary = pd.concat([
            train_df.assign(split='train'),
            val_df.assign(split='val'),
            test_df.assign(split='test')
        ], ignore_index=True)

        summary[['node_id', 'label', 'degree', 'split']].to_csv(f"{prefix}_summary.csv", index=False)
        print(f"Summary saved: {prefix}_summary.csv")

        # Save metadata
        with open(f"{prefix}_metadata.txt", 'w') as f:
            f.write(f"Created: {datetime.now()}\n")
            f.write(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")

        print(f"Metadata saved: {prefix}_metadata.txt")


def main():
    graph_file = r"C:\Users\32322\OneDrive\Desktop\MOMZI\transtocsv\transtocsv\fakenews_graph_20251101_144330.pkl"

    creator = DatasetCreator(graph_file)

    news_df = creator.get_news_nodes_data()
    print(f"\nExtracted {len(news_df)} news nodes.")

    train_df, val_df, test_df = creator.create_stratified_split(news_df)

    creator.save_outputs(train_df, val_df, test_df, prefix="fakenews_dataset")

if __name__ == "__main__":
    main()
