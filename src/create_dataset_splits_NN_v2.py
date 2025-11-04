import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
from collections import Counter, defaultdict

class DatasetCreatorNN:
    def __init__(self, graph_file):
        """Load the saved graph from pickle file"""
        print(f"Loading graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.G = graph_data['graph']
        self.users_df = graph_data.get('users_df')
        self.user_trustworthiness = graph_data.get('user_trustworthiness', {})
        
        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Verify graph structure
        self._verify_graph_structure()
    
    def _verify_graph_structure(self):
        """Verify that the graph is a proper bipartite user-news graph"""
        print("\nVerifying graph structure...")
        
        user_nodes = set()
        news_nodes = set()
        
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('node_type')
            if node_type == 'user':
                user_nodes.add(node)
            elif node_type == 'news':
                news_nodes.add(node)
            else:
                print(f"Warning: Node {node} has unknown type: {node_type}")
        
        print(f"  User nodes: {len(user_nodes)}")
        print(f"  News nodes: {len(news_nodes)}")
        
        # Check that all edges are between users and news
        non_bipartite_edges = 0
        for u, v in self.G.edges():
            u_type = self.G.nodes[u].get('node_type')
            v_type = self.G.nodes[v].get('node_type')
            
            if not ((u_type == 'user' and v_type == 'news') or 
                    (u_type == 'news' and v_type == 'user')):
                non_bipartite_edges += 1
        
        if non_bipartite_edges > 0:
            print(f"Warning: Found {non_bipartite_edges} non-bipartite edges!")
        else:
            print("  ✓ All edges are bipartite (user-news)")
        
        self.user_nodes = user_nodes
        self.news_nodes = news_nodes
    
    def get_user_news_edges(self):
        """
        Extract all (user, news) edges as training examples.
        Each edge becomes one training example with user features and news label.
        """
        print("\nExtracting user-news edges...")
        edges_data = []
        processed_edges = set()  # Track processed edges to avoid duplicates
        
        # Iterate through all edges
        for node1, node2 in self.G.edges():
            # Create a canonical edge representation (sorted tuple)
            edge_key = tuple(sorted([node1, node2]))
            
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            
            node1_data = self.G.nodes[node1]
            node2_data = self.G.nodes[node2]
            
            # Determine which is user and which is news
            if node1_data.get('node_type') == 'user' and node2_data.get('node_type') == 'news':
                user_id = node1
                news_id = node2
                user_node = node1_data
                news_node = node2_data
            elif node1_data.get('node_type') == 'news' and node2_data.get('node_type') == 'user':
                user_id = node2
                news_id = node1
                user_node = node2_data
                news_node = node1_data
            else:
                # Skip non-bipartite edges
                print(f"Warning: Skipping edge between {node1_data.get('node_type')} and {node2_data.get('node_type')}")
                continue
            
            # Extract user features (5 NN training features)
            edge_record = {
                'user_id': user_id,
                'news_id': news_id,
                'gossipcop_fake': user_node.get('gossipcop_fake', 0),
                'gossipcop_real': user_node.get('gossipcop_real', 0),
                'politifact_fake': user_node.get('politifact_fake', 0),
                'politifact_real': user_node.get('politifact_real', 0),
                'total_shares': user_node.get('total_shares', 0),
                'label': news_node.get('label', 0),  # 0=fake, 1=real
                'news_source': news_node.get('source', 'unknown'),
                'user_trustworthiness': user_node.get('trustworthiness', 0.5)  # Optional, for baseline
            }
            
            edges_data.append(edge_record)
        
        edges_df = pd.DataFrame(edges_data)
        
        print(f"Extracted {len(edges_df)} unique user-news edges")
        
        # Verify no duplicates
        duplicates = edges_df.duplicated(subset=['user_id', 'news_id']).sum()
        if duplicates > 0:
            print(f"Warning: Found {duplicates} duplicate edges after processing")
            edges_df = edges_df.drop_duplicates(subset=['user_id', 'news_id'])
            print(f"After removing duplicates: {len(edges_df)} edges")
        
        # Verify feature consistency
        print("\nVerifying feature consistency...")
        feature_issues = 0
        for _, row in edges_df.iterrows():
            expected_total = (row['gossipcop_fake'] + row['gossipcop_real'] + 
                            row['politifact_fake'] + row['politifact_real'])
            if row['total_shares'] != expected_total:
                feature_issues += 1
        
        if feature_issues > 0:
            print(f"Warning: {feature_issues} edges have inconsistent total_shares")
        else:
            print("  ✓ All edges have consistent features")
        
        return edges_df
    
    def create_stratified_split(self, edges_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
        """
        Create stratified train/validation/test split based on news nodes.
        Strategy: Split news nodes first, then assign edges to splits based on their news node.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        print("\n" + "="*60)
        print("CREATING DATASET SPLITS")
        print("="*60)
        
        # Get unique news nodes with their properties
        unique_news = edges_df[['news_id', 'label', 'news_source']].drop_duplicates()
        
        # Verify we have news from all categories
        print("\nChecking news distribution...")
        label_counts = unique_news['label'].value_counts()
        source_counts = unique_news['news_source'].value_counts()
        
        print(f"Labels: {dict(label_counts)}")
        print(f"Sources: {dict(source_counts)}")
        
        # Create stratification key: combination of label and source
        unique_news['stratify_key'] = unique_news['label'].astype(str) + '_' + unique_news['news_source']
        
        # Check if we have enough samples in each stratification group
        stratify_counts = unique_news['stratify_key'].value_counts()
        min_samples = stratify_counts.min()
        
        print(f"\nStratification groups:")
        for key, count in stratify_counts.items():
            print(f"  {key}: {count} news nodes")
        
        if min_samples < 3:
            print(f"\nWarning: Smallest stratification group has only {min_samples} samples.")
            print("This may cause issues with stratified splitting.")
            print("Consider using non-stratified split or combining categories.")
            
            # Fallback: use label-only stratification
            print("\nFalling back to label-only stratification...")
            unique_news['stratify_key'] = unique_news['label'].astype(str)
        
        # Print distribution before split
        print("\nOriginal News Distribution:")
        print(f"Total unique news nodes: {len(unique_news)}")
        print("\nBy label:")
        print(unique_news['label'].value_counts())
        print("\nBy source:")
        print(unique_news['news_source'].value_counts())
        print("\nBy stratify_key:")
        print(unique_news['stratify_key'].value_counts())
        
        # First split: train vs (val+test) news nodes
        try:
            train_news, temp_news = train_test_split(
                unique_news,
                test_size=(val_ratio + test_ratio),
                stratify=unique_news['stratify_key'],
                random_state=random_state
            )
            
            # Second split: val vs test news nodes
            val_news, test_news = train_test_split(
                temp_news,
                test_size=test_ratio / (val_ratio + test_ratio),
                stratify=temp_news['stratify_key'],
                random_state=random_state
            )
        except ValueError as e:
            print(f"\nStratified split failed: {e}")
            print("Falling back to non-stratified random split...")
            
            train_news, temp_news = train_test_split(
                unique_news,
                test_size=(val_ratio + test_ratio),
                random_state=random_state
            )
            
            val_news, test_news = train_test_split(
                temp_news,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=random_state
            )
        
        # Get news IDs for each split
        train_news_ids = set(train_news['news_id'])
        val_news_ids = set(val_news['news_id'])
        test_news_ids = set(test_news['news_id'])
        
        print(f"\nNews nodes split:")
        print(f"  Train: {len(train_news_ids)} news nodes")
        print(f"  Val:   {len(val_news_ids)} news nodes")
        print(f"  Test:  {len(test_news_ids)} news nodes")
        
        # Assign edges to splits based on their news node
        train_edges = edges_df[edges_df['news_id'].isin(train_news_ids)].copy()
        val_edges = edges_df[edges_df['news_id'].isin(val_news_ids)].copy()
        test_edges = edges_df[edges_df['news_id'].isin(test_news_ids)].copy()
        
        # Verify splits sum to total
        total_split_edges = len(train_edges) + len(val_edges) + len(test_edges)
        if total_split_edges != len(edges_df):
            print(f"\nWarning: Split edges ({total_split_edges}) != original edges ({len(edges_df)})")
            missing_edges = len(edges_df) - total_split_edges
            print(f"Missing edges: {missing_edges}")
        
        # Verify no overlap
        assert len(set(train_edges['news_id']) & set(val_edges['news_id'])) == 0, "Train-Val news overlap!"
        assert len(set(train_edges['news_id']) & set(test_edges['news_id'])) == 0, "Train-Test news overlap!"
        assert len(set(val_edges['news_id']) & set(test_edges['news_id'])) == 0, "Val-Test news overlap!"
        print("\n✓ No news node overlap between splits")
        
        print("\n" + "="*60)
        print("EDGE-BASED SPLIT STATISTICS")
        print("="*60)
        
        total_edges = len(edges_df)
        print(f"\nTotal edges: {total_edges}")
        print(f"\nTrain edges: {len(train_edges)} ({len(train_edges)/total_edges*100:.1f}%)")
        self._print_edge_stats(train_edges)
        
        print(f"\nValidation edges: {len(val_edges)} ({len(val_edges)/total_edges*100:.1f}%)")
        self._print_edge_stats(val_edges)
        
        print(f"\nTest edges: {len(test_edges)} ({len(test_edges)/total_edges*100:.1f}%)")
        self._print_edge_stats(test_edges)
        
        # Check user overlap (this is OK - same user can be in multiple splits)
        train_users = set(train_edges['user_id'])
        val_users = set(val_edges['user_id'])
        test_users = set(test_edges['user_id'])
        
        print("\n" + "="*60)
        print("USER STATISTICS (overlap is expected and OK)")
        print("="*60)
        print(f"Unique users in train: {len(train_users)}")
        print(f"Unique users in val:   {len(val_users)}")
        print(f"Unique users in test:  {len(test_users)}")
        print(f"Users in both train & val:  {len(train_users & val_users)}")
        print(f"Users in both train & test: {len(train_users & test_users)}")
        print(f"Users in both val & test:   {len(val_users & test_users)}")
        print(f"Users in all three splits:  {len(train_users & val_users & test_users)}")
        
        return train_edges, val_edges, test_edges
    
    def _print_edge_stats(self, edges_df):
        """Print statistics for edge-based split"""
        print(f"  Unique users: {edges_df['user_id'].nunique()}")
        print(f"  Unique news:  {edges_df['news_id'].nunique()}")
        
        print(f"  Label distribution:")
        for label, count in sorted(edges_df['label'].value_counts().items()):
            label_name = 'real' if label == 1 else 'fake'
            print(f"    {label_name} ({label}): {count} ({count/len(edges_df)*100:.1f}%)")
        
        print(f"  Source distribution:")
        for src, count in sorted(edges_df['news_source'].value_counts().items()):
            print(f"    {src}: {count} ({count/len(edges_df)*100:.1f}%)")
    
    def save_splits(self, train_edges, val_edges, test_edges, output_prefix='dataset'):
        """Save train/val/test edge splits to CSV files"""
        train_file = f"{output_prefix}_train.csv"
        val_file = f"{output_prefix}_val.csv"
        test_file = f"{output_prefix}_test.csv"
        
        # Select columns for output (user features + label)
        output_cols = ['user_id', 'news_id', 
                      'gossipcop_fake', 'gossipcop_real', 
                      'politifact_fake', 'politifact_real', 
                      'total_shares', 'label',
                      'news_source', 'user_trustworthiness']
        
        # Save to CSV
        train_edges[output_cols].to_csv(train_file, index=False)
        val_edges[output_cols].to_csv(val_file, index=False)
        test_edges[output_cols].to_csv(test_file, index=False)
        
        print("\n" + "="*60)
        print("FILES SAVED")
        print("="*60)
        print(f"Train set: {train_file}")
        print(f"Validation set: {val_file}")
        print(f"Test set: {test_file}")
        
        # Save metadata
        metadata_file = f"{output_prefix}_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write("DATASET SPLIT METADATA (NN v2)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total edges: {len(train_edges) + len(val_edges) + len(test_edges)}\n\n")
            
            f.write("Split sizes (edges):\n")
            total = len(train_edges) + len(val_edges) + len(test_edges)
            f.write(f"  Train: {len(train_edges)} ({len(train_edges)/total*100:.1f}%)\n")
            f.write(f"  Val:   {len(val_edges)} ({len(val_edges)/total*100:.1f}%)\n")
            f.write(f"  Test:  {len(test_edges)} ({len(test_edges)/total*100:.1f}%)\n\n")
            
            f.write("Training paradigm:\n")
            f.write("  Each row = one training example (user-news edge)\n")
            f.write("  Input (X): User features (5 features)\n")
            f.write("    - gossipcop_fake: # of fake news from Gossipcop shared by user\n")
            f.write("    - gossipcop_real: # of real news from Gossipcop shared by user\n")
            f.write("    - politifact_fake: # of fake news from Politifact shared by user\n")
            f.write("    - politifact_real: # of real news from Politifact shared by user\n")
            f.write("    - total_shares: Total # of news shared by user\n")
            f.write("  Target (y): News label (0=fake, 1=real)\n\n")
            
            f.write("Additional columns (not for training):\n")
            f.write("  - user_id: User identifier\n")
            f.write("  - news_id: News identifier\n")
            f.write("  - news_source: News source (gossipcop/politifact)\n")
            f.write("  - user_trustworthiness: For baseline comparison only\n\n")
            
            f.write("Files:\n")
            f.write(f"  {train_file}\n")
            f.write(f"  {val_file}\n")
            f.write(f"  {test_file}\n\n")
            
            f.write("Usage:\n")
            f.write("  import pandas as pd\n")
            f.write("  train_df = pd.read_csv('dataset_train.csv')\n")
            f.write("  \n")
            f.write("  # Extract features and labels\n")
            f.write("  feature_cols = ['gossipcop_fake', 'gossipcop_real', \n")
            f.write("                  'politifact_fake', 'politifact_real', 'total_shares']\n")
            f.write("  X_train = train_df[feature_cols].values\n")
            f.write("  y_train = train_df['label'].values\n")
        
        print(f"Metadata: {metadata_file}")
        
        return train_file, val_file, test_file, metadata_file
    
    def analyze_features(self, train_edges, val_edges, test_edges):
        """Analyze user feature distributions across splits"""
        print("\n" + "="*60)
        print("USER FEATURE STATISTICS (NN Training Features)")
        print("="*60)
        
        feature_cols = ['gossipcop_fake', 'gossipcop_real', 
                       'politifact_fake', 'politifact_real', 'total_shares']
        
        for feature in feature_cols:
            print(f"\n{feature}:")
            print(f"  Train - Mean: {train_edges[feature].mean():.2f}, "
                  f"Std: {train_edges[feature].std():.2f}, "
                  f"Min: {train_edges[feature].min()}, "
                  f"Max: {train_edges[feature].max()}")
            print(f"  Val   - Mean: {val_edges[feature].mean():.2f}, "
                  f"Std: {val_edges[feature].std():.2f}, "
                  f"Min: {val_edges[feature].min()}, "
                  f"Max: {val_edges[feature].max()}")
            print(f"  Test  - Mean: {test_edges[feature].mean():.2f}, "
                  f"Std: {test_edges[feature].std():.2f}, "
                  f"Min: {test_edges[feature].min()}, "
                  f"Max: {test_edges[feature].max()}")
        
        # Also show trustworthiness (baseline feature)
        print(f"\nuser_trustworthiness (baseline only, not for NN training):")
        print(f"  Train - Mean: {train_edges['user_trustworthiness'].mean():.4f}, "
              f"Std: {train_edges['user_trustworthiness'].std():.4f}")
        print(f"  Val   - Mean: {val_edges['user_trustworthiness'].mean():.4f}, "
              f"Std: {val_edges['user_trustworthiness'].std():.4f}")
        print(f"  Test  - Mean: {test_edges['user_trustworthiness'].mean():.4f}, "
              f"Std: {test_edges['user_trustworthiness'].std():.4f}")
    
    def create_summary(self, train_edges, val_edges, test_edges, output_prefix='dataset'):
        """Create summary statistics file"""
        summary_file = f"{output_prefix}_summary.csv"
        
        # Aggregate statistics
        summary_data = []
        
        for split_name, split_df in [('train', train_edges), ('val', val_edges), ('test', test_edges)]:
            summary_data.append({
                'split': split_name,
                'num_edges': len(split_df),
                'num_users': split_df['user_id'].nunique(),
                'num_news': split_df['news_id'].nunique(),
                'num_fake': (split_df['label'] == 0).sum(),
                'num_real': (split_df['label'] == 1).sum(),
                'pct_fake': (split_df['label'] == 0).sum() / len(split_df) * 100,
                'pct_real': (split_df['label'] == 1).sum() / len(split_df) * 100,
                'avg_gossipcop_fake': split_df['gossipcop_fake'].mean(),
                'avg_gossipcop_real': split_df['gossipcop_real'].mean(),
                'avg_politifact_fake': split_df['politifact_fake'].mean(),
                'avg_politifact_real': split_df['politifact_real'].mean(),
                'avg_total_shares': split_df['total_shares'].mean(),
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Summary statistics: {summary_file}")
        
        return summary_file

def main():
    parser = argparse.ArgumentParser(
        description='Create train/validation/test splits for NN training (user features → news labels)'
    )
    parser.add_argument('graph_file', type=str, 
                       help='Path to the pickle file containing the graph')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Ratio of training data (default: 0.6)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of validation data (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Ratio of test data (default: 0.2)')
    parser.add_argument('--output-prefix', '-o', type=str, default='dataset',
                       help='Prefix for output files (default: dataset)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0 (current sum: {total_ratio})")
        return
    
    # Create dataset
    creator = DatasetCreatorNN(args.graph_file)
    
    # Get user-news edges
    edges_df = creator.get_user_news_edges()
    
    if len(edges_df) == 0:
        print("Error: No edges found in graph!")
        return
    
    # Create splits (stratified by news nodes)
    train_edges, val_edges, test_edges = creator.create_stratified_split(
        edges_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )
    
    # Analyze features
    creator.analyze_features(train_edges, val_edges, test_edges)
    
    # Save splits
    train_file, val_file, test_file, metadata_file = creator.save_splits(
        train_edges, val_edges, test_edges, args.output_prefix
    )
    
    # Create summary
    summary_file = creator.create_summary(
        train_edges, val_edges, test_edges, args.output_prefix
    )
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print("\nTraining paradigm:")
    print("  Input (X): User features [gossipcop_fake, gossipcop_real,")
    print("                            politifact_fake, politifact_real, total_shares]")
    print("  Target (y): News label (0=fake, 1=real)")
    print("\nEach row in the CSV = one training example")
    print("\nNext steps:")
    print("1. Load train/val/test CSV files")
    print("2. Extract features and labels:")
    print("   X = df[['gossipcop_fake', 'gossipcop_real', 'politifact_fake',")
    print("           'politifact_real', 'total_shares']].values")
    print("   y = df['label'].values")
    print("3. Train your neural network!")
    print("\nNote: Same user can appear in multiple splits (different news)")
    print("      but no news node appears in multiple splits (no leakage)")

if __name__ == "__main__":
    main()