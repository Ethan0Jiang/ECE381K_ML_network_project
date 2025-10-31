import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
from collections import Counter

class DatasetCreator:
    def __init__(self, graph_file):
        """Load the saved graph from pickle file"""
        print(f"Loading graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.G = graph_data['graph']
        self.users_df = graph_data.get('users_df')
        self.user_trustworthiness = graph_data.get('user_trustworthiness', {})
        
        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def get_news_nodes_data(self):
        """Extract all news nodes with their features and labels"""
        news_data = []
        
        for node in self.G.nodes():
            if self.G.nodes[node]['node_type'] == 'news':
                node_data = self.G.nodes[node]
                
                # Extract features
                record = {
                    'node_id': node,
                    'ground_truth': node_data['ground_truth'],
                    'label': 1 if node_data['ground_truth'] == 'real' else 0,  # Binary: real=1, fake=0
                    'source': node_data.get('source', 'unknown'),
                    'title': node_data.get('title', ''),
                    'degree': self.G.degree(node),
                    'num_shares': node_data.get('num_shares', 0),
                    'avg_user_trust': node_data.get('avg_user_trust', 0.5),
                    'min_user_trust': node_data.get('min_user_trust', 0.5),
                    'max_user_trust': node_data.get('max_user_trust', 0.5),
                    'trust_std': node_data.get('trust_std', 0.0),
                    'high_trust_ratio': node_data.get('high_trust_ratio', 0.0),
                }
                
                news_data.append(record)
        
        return pd.DataFrame(news_data)
    
    def create_stratified_split(self, news_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
        """
        Create stratified train/validation/test split
        Stratify by: ground_truth (fake/real) and source (politifact/gossipcop)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        print("\n" + "="*60)
        print("CREATING DATASET SPLITS")
        print("="*60)
        
        # Create stratification key: combination of ground_truth and source
        news_df['stratify_key'] = news_df['ground_truth'] + '_' + news_df['source']
        
        # Print distribution before split
        print("\nOriginal Dataset Distribution:")
        print(f"Total news nodes: {len(news_df)}")
        print("\nBy ground_truth:")
        print(news_df['ground_truth'].value_counts())
        print("\nBy source:")
        print(news_df['source'].value_counts())
        print("\nBy stratify_key:")
        print(news_df['stratify_key'].value_counts())
        
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            news_df,
            test_size=(val_ratio + test_ratio),
            stratify=news_df['stratify_key'],
            random_state=random_state
        )
        
        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_df['stratify_key'],
            random_state=random_state
        )
        
        # Remove stratify_key column (not needed in final output)
        train_df = train_df.drop(columns=['stratify_key'])
        val_df = val_df.drop(columns=['stratify_key'])
        test_df = test_df.drop(columns=['stratify_key'])
        
        # Print split statistics
        print("\n" + "="*60)
        print("SPLIT STATISTICS")
        print("="*60)
        
        print(f"\nTrain set: {len(train_df)} nodes ({len(train_df)/len(news_df)*100:.1f}%)")
        self._print_split_stats(train_df)
        
        print(f"\nValidation set: {len(val_df)} nodes ({len(val_df)/len(news_df)*100:.1f}%)")
        self._print_split_stats(val_df)
        
        print(f"\nTest set: {len(test_df)} nodes ({len(test_df)/len(news_df)*100:.1f}%)")
        self._print_split_stats(test_df)
        
        # Check degree distribution across splits
        print("\n" + "="*60)
        print("DEGREE DISTRIBUTION ACROSS SPLITS")
        print("="*60)
        print(f"\nTrain - Mean degree: {train_df['degree'].mean():.2f}, "
              f"Median: {train_df['degree'].median():.1f}, "
              f"Min: {train_df['degree'].min()}, "
              f"Max: {train_df['degree'].max()}")
        print(f"Val   - Mean degree: {val_df['degree'].mean():.2f}, "
              f"Median: {val_df['degree'].median():.1f}, "
              f"Min: {val_df['degree'].min()}, "
              f"Max: {val_df['degree'].max()}")
        print(f"Test  - Mean degree: {test_df['degree'].mean():.2f}, "
              f"Median: {test_df['degree'].median():.1f}, "
              f"Min: {test_df['degree'].min()}, "
              f"Max: {test_df['degree'].max()}")
        
        return train_df, val_df, test_df
    
    def _print_split_stats(self, df):
        """Print statistics for a data split"""
        print(f"  Ground truth distribution:")
        for gt, count in df['ground_truth'].value_counts().items():
            print(f"    {gt}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"  Source distribution:")
        for src, count in df['source'].value_counts().items():
            print(f"    {src}: {count} ({count/len(df)*100:.1f}%)")
    
    def save_splits(self, train_df, val_df, test_df, output_prefix='dataset'):
        """Save train/val/test splits to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        train_file = f"{output_prefix}_train.csv"
        val_file = f"{output_prefix}_val.csv"
        test_file = f"{output_prefix}_test.csv"
        
        # Save to CSV
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print("\n" + "="*60)
        print("FILES SAVED")
        print("="*60)
        print(f"Train set: {train_file}")
        print(f"Validation set: {val_file}")
        print(f"Test set: {test_file}")
        
        # Also save a metadata file with split information
        metadata_file = f"{output_prefix}_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write("DATASET SPLIT METADATA\n")
            f.write("="*60 + "\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total news nodes: {len(train_df) + len(val_df) + len(test_df)}\n\n")
            
            f.write("Split sizes:\n")
            f.write(f"  Train: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)\n")
            f.write(f"  Val:   {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)\n")
            f.write(f"  Test:  {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)\n\n")
            
            f.write("Files:\n")
            f.write(f"  {train_file}\n")
            f.write(f"  {val_file}\n")
            f.write(f"  {test_file}\n\n")
            
            f.write("Usage:\n")
            f.write("  1. Load the graph: pickle.load('your_graph.pkl')\n")
            f.write("  2. Load train/val/test CSV files\n")
            f.write("  3. Use 'node_id' column to identify which nodes are in which split\n")
            f.write("  4. Use 'label' column (0=fake, 1=real) as ground truth\n")
        
        print(f"Metadata: {metadata_file}")
        
        return train_file, val_file, test_file, metadata_file
    
    def analyze_features(self, train_df, val_df, test_df):
        """Analyze feature distributions across splits"""
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        
        feature_cols = ['num_shares', 'avg_user_trust', 'min_user_trust', 
                       'max_user_trust', 'trust_std', 'high_trust_ratio']
        
        for feature in feature_cols:
            print(f"\n{feature}:")
            print(f"  Train - Mean: {train_df[feature].mean():.4f}, Std: {train_df[feature].std():.4f}")
            print(f"  Val   - Mean: {val_df[feature].mean():.4f}, Std: {val_df[feature].std():.4f}")
            print(f"  Test  - Mean: {test_df[feature].mean():.4f}, Std: {test_df[feature].std():.4f}")
    
    def create_simple_summary(self, train_df, val_df, test_df, output_prefix='dataset'):
        """Create a simple summary CSV showing node IDs and their split assignment"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{output_prefix}_summary.csv"
        
        # Create summary dataframe
        train_summary = train_df[['node_id', 'ground_truth', 'label', 'degree']].copy()
        train_summary['split'] = 'train'
        
        val_summary = val_df[['node_id', 'ground_truth', 'label', 'degree']].copy()
        val_summary['split'] = 'val'
        
        test_summary = test_df[['node_id', 'ground_truth', 'label', 'degree']].copy()
        test_summary['split'] = 'test'
        
        # Combine
        summary_df = pd.concat([train_summary, val_summary, test_summary], ignore_index=True)
        summary_df = summary_df.sort_values('node_id')
        
        # Save
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary file: {summary_file}")
        print(f"  (Contains all node IDs with their split assignments)")
        
        return summary_file

def main():
    parser = argparse.ArgumentParser(
        description='Create train/validation/test splits for fake news graph dataset'
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
    creator = DatasetCreator(args.graph_file)
    
    # Get news nodes data
    print("\nExtracting news nodes data...")
    news_df = creator.get_news_nodes_data()
    print(f"Extracted {len(news_df)} news nodes")
    
    # Create splits
    train_df, val_df, test_df = creator.create_stratified_split(
        news_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )
    
    # Analyze features
    creator.analyze_features(train_df, val_df, test_df)
    
    # Save splits
    train_file, val_file, test_file, metadata_file = creator.save_splits(
        train_df, val_df, test_df, args.output_prefix
    )
    
    # Create summary file
    summary_file = creator.create_simple_summary(
        train_df, val_df, test_df, args.output_prefix
    )
    
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Load your graph from the .pkl file")
    print("2. Load the train/val/test CSV files to get node IDs for each split")
    print("3. Use the 'node_id' column to identify which nodes belong to which split")
    print("4. When training:")
    print("   - Compute loss only on training node IDs")
    print("   - Evaluate on validation node IDs for hyperparameter tuning")
    print("   - Final evaluation on test node IDs")
    print("\nExample usage in training:")
    print("  train_nodes = set(train_df['node_id'])")
    print("  val_nodes = set(val_df['node_id'])")
    print("  test_nodes = set(test_df['node_id'])")

if __name__ == "__main__":
    main()