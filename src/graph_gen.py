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
        self.users_df = None
        np.random.seed(42)  # For reproducibility
        
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
    
    def create_tweet_pools(self, datasets):
        """Create pools of tweet IDs for fake and real news"""
        fake_tweets = set()
        real_tweets = set()
        news_to_tweets = {}
        
        for dataset_name, df in datasets.items():
            label = 'fake' if 'fake' in dataset_name else 'real'
            source = dataset_name.split('_')[0]
            
            for idx, row in df.iterrows():
                news_id = f"{source}_{row['id']}"
                news_to_tweets[news_id] = {'label': label, 'tweets': []}
                
                if pd.notna(row['tweet_ids']):
                    tweet_ids = str(row['tweet_ids']).split('\t')
                    for tweet_id in tweet_ids:
                        tweet_id = tweet_id.strip()
                        if tweet_id and tweet_id != 'nan' and tweet_id != '':
                            news_to_tweets[news_id]['tweets'].append(tweet_id)
                            if label == 'fake':
                                fake_tweets.add(tweet_id)
                            else:
                                real_tweets.add(tweet_id)
        
        print(f"Total fake tweets: {len(fake_tweets)}")
        print(f"Total real tweets: {len(real_tweets)}")
        
        return fake_tweets, real_tweets, news_to_tweets
    

    def generate_users(self, fake_tweets, real_tweets, num_users=50000):
        """Generate synthetic users with realistic sharing patterns"""
        print(f"Generating {num_users} users...")
        
        all_fake_tweets = list(fake_tweets)
        all_real_tweets = list(real_tweets)
        
        # Calculate total available tweets
        total_available_tweets = len(all_fake_tweets) + len(all_real_tweets)
        
        # Calculate target total tweets to distribute
        avg_tweets_per_user = 7
        target_total_tweets = num_users * avg_tweets_per_user
        
        # Check if we have enough tweets
        if target_total_tweets > total_available_tweets:
            print(f"Warning: Not enough unique tweets ({total_available_tweets}) for {num_users} users")
            print(f"Adjusting to allow tweet reuse or reducing average...")
            # We'll allow some tweet reuse in this case
        
        # Generate tweet counts using normal distribution
        # Mean = 7, we'll use a wide standard deviation (e.g., std = 5)
        # This creates a wide spread while centering around 7
        std_dev = 5
        tweet_counts = np.random.normal(avg_tweets_per_user, std_dev, num_users)
        
        # Round to integers and ensure minimum of 1 tweet
        tweet_counts = np.clip(np.round(tweet_counts), 1, None).astype(int)
        
        # Scale to match our target total (optional - to ensure we use available tweets efficiently)
        current_total = tweet_counts.sum()
        if current_total > 0:
            scaling_factor = min(target_total_tweets / current_total, 1.0)
            if scaling_factor < 1.0:
                tweet_counts = np.clip(np.round(tweet_counts * scaling_factor), 1, None).astype(int)
        
        print(f"Tweet distribution statistics:")
        print(f"  Mean: {tweet_counts.mean():.2f}")
        print(f"  Median: {np.median(tweet_counts):.2f}")
        print(f"  Std Dev: {tweet_counts.std():.2f}")
        print(f"  Min: {tweet_counts.min()}")
        print(f"  Max: {tweet_counts.max()}")
        print(f"  Total tweets to distribute: {tweet_counts.sum()}")
        
        users_data = []
        used_tweets = set()
        
        for user_id in range(1, num_users + 1):
            user_name = f"user_{user_id}"
            num_tweets = tweet_counts[user_id - 1]
            
            # Categorize based on number of tweets for compatibility
            if num_tweets <= 10:
                category = "light"
            elif num_tweets <= 100:
                category = "moderate"
            elif num_tweets <= 1000:
                category = "heavy"
            else:
                category = "super"
            
            # Determine user's preference for real vs fake news (normal distribution)
            # Mean at 0.5 (neutral), std 0.2 to create variety
            if np.random.random() < 0.5:
                # Group 1: Trustworthy users (prefer real news)
                real_preference = np.clip(np.random.normal(0.8, 0.1), 0.6, 0.95)
            else:
                # Group 2: Misinformation spreaders (prefer fake news)
                real_preference = np.clip(np.random.normal(0.2, 0.1), 0.05, 0.4)
            
            # Calculate how many real vs fake tweets this user will share
            num_real = int(num_tweets * real_preference)
            num_fake = num_tweets - num_real
            
            # Select tweets (allowing reuse if necessary)
            if len(used_tweets) < total_available_tweets * 0.95:  # Use unique tweets when possible
                # Try to use unique tweets
                available_real = [t for t in all_real_tweets if t not in used_tweets]
                available_fake = [t for t in all_fake_tweets if t not in used_tweets]
                
                # Fall back to all tweets if not enough unique ones
                if len(available_real) < num_real:
                    available_real = all_real_tweets
                if len(available_fake) < num_fake:
                    available_fake = all_fake_tweets
            else:
                # Allow reuse
                available_real = all_real_tweets
                available_fake = all_fake_tweets
            
            selected_real = np.random.choice(available_real, min(num_real, len(available_real)), 
                                            replace=False) if available_real and num_real > 0 else []
            selected_fake = np.random.choice(available_fake, min(num_fake, len(available_fake)), 
                                            replace=False) if available_fake and num_fake > 0 else []
            
            # Update used tweets (only if we're tracking uniqueness)
            if len(used_tweets) < total_available_tweets * 0.95:
                used_tweets.update(selected_real)
                used_tweets.update(selected_fake)
            
            # Store user data
            user_tweets = list(selected_real) + list(selected_fake)
            users_data.append({
                'user_id': user_name,
                'category': category,
                'real_preference': real_preference,
                'total_tweets': len(user_tweets),
                'real_tweets': len(selected_real),
                'fake_tweets': len(selected_fake),
                'tweet_ids': '\t'.join(user_tweets) if user_tweets else ''
            })
            
            if user_id % 10000 == 0:
                print(f"Generated {user_id} users...")
        
        self.users_df = pd.DataFrame(users_data)
        return self.users_df

    
    def save_users_csv(self, filename='users.csv'):
        """Save users dataframe to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"users_{timestamp}.csv"
        
        if self.users_df is not None:
            self.users_df.to_csv(filename, index=False)
            print(f"Users saved to {filename}")
            
            # Print statistics
            print("\nUser Statistics:")
            print(f"Total users: {len(self.users_df)}")
            print(f"Category distribution:")
            print(self.users_df['category'].value_counts())
            print(f"\nTweet sharing statistics:")
            print(f"Mean tweets per user: {self.users_df['total_tweets'].mean():.2f}")
            print(f"Median tweets per user: {self.users_df['total_tweets'].median():.2f}")
            print(f"Users with >0 tweets: {len(self.users_df[self.users_df['total_tweets'] > 0])}")
            
            return filename
        else:
            print("No users data to save")
            return None
    
    def build_user_news_graph(self, datasets, news_to_tweets):
        """Build bipartite graph with user and news nodes"""
        G = nx.Graph()
        user_sharing_history = defaultdict(lambda: {'fake': 0, 'real': 0})
        
        print("Building user-news bipartite graph...")
        
        # Add news nodes
        for dataset_name, df in datasets.items():
            label = 'fake' if 'fake' in dataset_name else 'real'
            source = dataset_name.split('_')[0]
            
            for idx, row in df.iterrows():
                news_id = f"{source}_{row['id']}"
                G.add_node(news_id,
                          node_type='news',
                          title=str(row['title'])[:100] + '...' if len(str(row['title'])) > 100 else str(row['title']),
                          url=row['news_url'],
                          ground_truth=label,
                          source=source)
        
        # Add user nodes and edges based on generated users
        tweet_to_news = {}
        for news_id, info in news_to_tweets.items():
            for tweet_id in info['tweets']:
                tweet_to_news[tweet_id] = news_id
        
        for _, user_row in self.users_df.iterrows():
            user_id = user_row['user_id']
            
            # Add user node
            G.add_node(user_id, 
                      node_type='user',
                      category=user_row['category'],
                      real_preference=user_row['real_preference'])
            
            # Add edges for each tweet this user shared
            if user_row['tweet_ids']:
                tweet_ids = user_row['tweet_ids'].split('\t')
                
                for tweet_id in tweet_ids:
                    if tweet_id in tweet_to_news:
                        news_id = tweet_to_news[tweet_id]
                        
                        # Add edge between user and news
                        G.add_edge(user_id, news_id, 
                                  relationship='shares',
                                  tweet_id=tweet_id)
                        
                        # Update sharing history
                        news_label = G.nodes[news_id]['ground_truth']
                        user_sharing_history[user_id][news_label] += 1
        
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
            'users_df': self.users_df,
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
    
    # Load datasets
    datasets = builder.load_datasets()
    if not datasets:
        print("No datasets loaded. Check your dataset path.")
        return
    
    # Create tweet pools
    fake_tweets, real_tweets, news_to_tweets = builder.create_tweet_pools(datasets)
    
    # Generate users
    users_df = builder.generate_users(fake_tweets, real_tweets, num_users=18363)
    
    # Save users to CSV
    users_file = builder.save_users_csv()
    
    # Build graph
    G, user_sharing_history = builder.build_user_news_graph(datasets, news_to_tweets)
    
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
    print(f"Users CSV: {users_file}")
    print(f"Graph saved to: {saved_file}")
    print(f"GEXF file for Gephi: {gexf_file}")
    
    return builder

if __name__ == "__main__":
    builder = main()