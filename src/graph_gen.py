import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import pickle
import json
from datetime import datetime
import os

class FakeNewsNetworkBuilder:
    def __init__(self, dataset_path='../external/FakeNewsNet/dataset'):
        self.dataset_path = dataset_path
        self.G = None
        self.user_trustworthiness = {}
        
    def load_datasets(self):
        """Load all CSV files from FakeNewsNet dataset"""
        datasets = {}
        csv_files = [
            ('politifact_fake', 'politifact_fake.csv'),
            ('politifact_real', 'politifact_real.csv'),
            ('gossipcop_fake', 'gossipcop_fake.csv'),
            ('gossipcop_real', 'gossipcop_real.csv')
        ]
        
        for name, filename in csv_files:
            filepath = os.path.join(self.dataset_path, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    datasets[name] = df
                    print(f"Loaded {name}: {len(df)} articles")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File not found: {filepath}")
        
        return datasets
    
    def build_bipartite_graph(self, datasets):
        """Build bipartite graph with news and user nodes"""
        G = nx.Graph()
        user_sharing_history = defaultdict(lambda: {'fake': 0, 'real': 0})
        
        print("Building bipartite graph...")
        
        for dataset_name, df in datasets.items():
            label = 'fake' if 'fake' in dataset_name else 'real'
            source = dataset_name.split('_')[0]  # politifact or gossipcop
            
            print(f"Processing {dataset_name}...")
            
            for idx, row in df.iterrows():
                news_id = f"{source}_{row['id']}"
                
                # Add news node
                G.add_node(news_id,
                          node_type='news',
                          title=str(row['title'])[:100] + '...' if len(str(row['title'])) > 100 else str(row['title']),
                          url=row['news_url'],
                          ground_truth=label,
                          source=source)
                
                # Process tweet IDs
                if pd.notna(row['tweet_ids']):
                    tweet_ids = str(row['tweet_ids']).split('\t')
                    
                    for tweet_id in tweet_ids:
                        tweet_id = tweet_id.strip()
                        if tweet_id and tweet_id != 'nan' and tweet_id != '':
                            user_id = f"user_{tweet_id}"
                            
                            # Add user node if not exists
                            if not G.has_node(user_id):
                                G.add_node(user_id, node_type='user')
                            
                            # Add edge
                            G.add_edge(user_id, news_id, relationship='shares')
                            
                            # Update sharing history
                            user_sharing_history[user_id][label] += 1
        
        self.G = G
        return G, user_sharing_history
    
    def calculate_trustworthiness(self, user_sharing_history):
        """Calculate user trustworthiness scores"""
        print("Calculating user trustworthiness...")
        
        user_trustworthiness = {}
        
        for user_id, history in user_sharing_history.items():
            total_shares = history['fake'] + history['real']
            
            if total_shares > 0:
                real_ratio = history['real'] / total_shares
                # Confidence weighting based on number of shares
                confidence = min(total_shares / 10, 1.0)
                trustworthiness = real_ratio * confidence + 0.5 * (1 - confidence)
            else:
                trustworthiness = 0.5
            
            user_trustworthiness[user_id] = trustworthiness
            
            # Add to graph
            if self.G.has_node(user_id):
                self.G.nodes[user_id]['trustworthiness'] = trustworthiness
                self.G.nodes[user_id]['total_shares'] = total_shares
                self.G.nodes[user_id]['fake_shares'] = history['fake']
                self.G.nodes[user_id]['real_shares'] = history['real']
        
        self.user_trustworthiness = user_trustworthiness
        return user_trustworthiness
    
    def calculate_news_features(self):
        """Calculate features for all news articles"""
        print("Calculating news features...")
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'news']
        
        for news_node in news_nodes:
            sharing_users = [n for n in self.G.neighbors(news_node) 
                           if self.G.nodes[n]['node_type'] == 'user']
            
            if sharing_users:
                user_scores = [self.G.nodes[user]['trustworthiness'] for user in sharing_users]
                
                features = {
                    'avg_user_trust': np.mean(user_scores),
                    'min_user_trust': np.min(user_scores),
                    'max_user_trust': np.max(user_scores),
                    'trust_std': np.std(user_scores),
                    'num_shares': len(sharing_users),
                    'high_trust_ratio': sum(1 for score in user_scores if score > 0.7) / len(user_scores)
                }
            else:
                features = {
                    'avg_user_trust': 0.5, 'min_user_trust': 0.5, 'max_user_trust': 0.5,
                    'trust_std': 0.0, 'num_shares': 0, 'high_trust_ratio': 0.0
                }
            
            # Add features to node
            for key, value in features.items():
                self.G.nodes[news_node][key] = value
    
    def export_to_gexf(self, filename=None):
        """Export graph to GEXF format for Gephi"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fakenews_network_{timestamp}.gexf"
        
        print(f"Exporting graph to GEXF format: {filename}")
        
        # Create a copy of the graph for export
        G_export = self.G.copy()
        
        # Add visualization attributes for Gephi
        for node in G_export.nodes():
            node_data = G_export.nodes[node]
            
            if node_data['node_type'] == 'news':
                # Color coding for news nodes: red=fake, green=real
                if node_data['ground_truth'] == 'fake':
                    color = {'r': 255, 'g': 0, 'b': 0}
                else:
                    color = {'r': 0, 'g': 255, 'b': 0}
                
                G_export.nodes[node]['viz'] = {
                    'color': color,
                    'size': min(node_data.get('num_shares', 1) * 2, 50)
                }
                
            else:  # user node
                # Color based on trustworthiness (blue gradient)
                trust = node_data.get('trustworthiness', 0.5)
                blue_intensity = int(255 * trust)
                G_export.nodes[node]['viz'] = {
                    'color': {'r': 100, 'g': 100, 'b': blue_intensity},
                    'size': min(node_data.get('total_shares', 1) + 5, 20)
                }
        
        # Write to GEXF
        nx.write_gexf(G_export, filename)
        print(f"Graph exported to {filename}")
        return filename
    
    def save_graph(self, filename=None):
        """Save graph to pickle file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fakenews_graph_{timestamp}.pkl"
        
        graph_data = {
            'graph': self.G,
            'user_trustworthiness': self.user_trustworthiness,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'stats': self.get_graph_stats()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"Graph saved to {filename}")
        return filename
    
    def get_graph_stats(self):
        """Get basic graph statistics"""
        if self.G is None:
            return {}
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'news']
        user_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'user']
        
        fake_news = [n for n in news_nodes if self.G.nodes[n]['ground_truth'] == 'fake']
        real_news = [n for n in news_nodes if self.G.nodes[n]['ground_truth'] == 'real']
        
        stats = {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'news_nodes': len(news_nodes),
            'user_nodes': len(user_nodes),
            'fake_news': len(fake_news),
            'real_news': len(real_news),
            'avg_degree': sum(dict(self.G.degree()).values()) / len(self.G.nodes()) if self.G.nodes() else 0
        }
        
        return stats

def main():
    # Initialize builder
    builder = FakeNewsNetworkBuilder(dataset_path='../external/FakeNewsNet/dataset')
    
    # Load and process data
    datasets = builder.load_datasets()
    if not datasets:
        print("No datasets loaded. Check your dataset path.")
        return
    
    # Build graph
    G, user_sharing_history = builder.build_bipartite_graph(datasets)
    
    # Calculate trustworthiness and features
    user_trustworthiness = builder.calculate_trustworthiness(user_sharing_history)
    builder.calculate_news_features()
    
    # Print statistics
    stats = builder.get_graph_stats()
    print("\n" + "="*50)
    print("GRAPH STATISTICS")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save graph
    saved_file = builder.save_graph()
    
    # Export to GEXF for Gephi
    print("\n" + "="*50)
    print("EXPORTING TO GEPHI")
    print("="*50)
    gexf_file = builder.export_to_gexf()
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Graph saved to: {saved_file}")
    print(f"GEXF file for Gephi: {gexf_file}")
    
    return builder

if __name__ == "__main__":
    builder = main()