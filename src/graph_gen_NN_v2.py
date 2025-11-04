import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import pickle
import json
from datetime import datetime
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import math

'''
To run it:
python graph_gen_NN_v2.py --num_partitions 32 --avg_tweets 13 --std_tweets 10 --buffer_factor 1.5

Or with default parameters:
python graph_gen_NN_v2.py

Options:
  --num_partitions: Number of parallel partitions (default: CPU count)
  --avg_tweets: Average tweets per user (default: 13)
  --std_tweets: Standard deviation for tweets per user (default: 10)
  --buffer_factor: User count buffer factor (default: 1.5)
  --skip_gexf: Skip GEXF export (saves time)
'''

class FakeNewsNetworkBuilderNN:
    def __init__(self, dataset_path='../external/FakeNewsNet/dataset', 
                 avg_tweets=13, std_tweets=10, buffer_factor=1.5):
        self.dataset_path = dataset_path
        self.G = None
        self.user_trustworthiness = {}
        self.users_df = None
        self.avg_tweets = avg_tweets
        self.std_tweets = std_tweets
        self.buffer_factor = buffer_factor
        np.random.seed(42)
        
    def load_datasets(self):
        """Load all CSV files from FakeNewsNet dataset and filter out news without tweets"""
        datasets = {}
        csv_files = [
            ('politifact_fake', 'politifact_fake.csv'),
            ('politifact_real', 'politifact_real.csv'),
            ('gossipcop_fake', 'gossipcop_fake.csv'),
            ('gossipcop_real', 'gossipcop_real.csv')
        ]
        
        total_articles = 0
        articles_with_tweets = 0
        articles_without_tweets = 0
        
        for name, filename in csv_files:
            filepath = os.path.join(self.dataset_path, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    total_articles += len(df)
                    
                    # Filter out articles without tweet_ids
                    df_filtered = df[df['tweet_ids'].notna() & 
                                   (df['tweet_ids'] != '') & 
                                   (df['tweet_ids'] != '[]')].copy()
                    
                    articles_with_tweets += len(df_filtered)
                    articles_without_tweets += len(df) - len(df_filtered)
                    
                    datasets[name] = df_filtered
                    print(f"Loaded {name}: {len(df)} total, {len(df_filtered)} with tweets")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File not found: {filepath}")
        
        print(f"\nDataset filtering summary:")
        print(f"  Total articles: {total_articles}")
        print(f"  Articles WITH tweets: {articles_with_tweets} ({articles_with_tweets/total_articles*100:.1f}%)")
        print(f"  Articles WITHOUT tweets: {articles_without_tweets} ({articles_without_tweets/total_articles*100:.1f}%)")
        print(f"  Filtered articles will be included in graph: {articles_with_tweets}")
        
        return datasets
    
    def create_tweet_pools(self, datasets):
        """Create pools of tweet IDs, ensuring each tweet appears only once"""
        news_to_tweets = {}
        total_tweets = 0
        seen_tweets = set()
        duplicate_tweets = 0
        
        for dataset_name, df in datasets.items():
            label = 'fake' if 'fake' in dataset_name else 'real'
            source = dataset_name.split('_')[0]
            
            for idx, row in df.iterrows():
                news_id = f"{source}_{row['id']}"
                news_to_tweets[news_id] = {
                    'label': 1 if label == 'real' else 0,
                    'source': source,
                    'tweets': []
                }
                
                if pd.notna(row['tweet_ids']):
                    tweet_ids = str(row['tweet_ids']).split('\t')
                    for tweet_id in tweet_ids:
                        tweet_id = tweet_id.strip()
                        if tweet_id and tweet_id != 'nan' and tweet_id != '':
                            # Check for duplicates
                            if tweet_id in seen_tweets:
                                duplicate_tweets += 1
                                continue
                            
                            seen_tweets.add(tweet_id)
                            news_to_tweets[news_id]['tweets'].append(tweet_id)
                            total_tweets += 1
        
        print(f"\nTweet pools created:")
        print(f"  Total unique tweets: {total_tweets}")
        print(f"  Duplicate tweets removed: {duplicate_tweets}")
        
        # Count tweets by category
        category_counts = {
            'gossipcop_fake': 0,
            'gossipcop_real': 0,
            'politifact_fake': 0,
            'politifact_real': 0
        }
        
        for news_id, info in news_to_tweets.items():
            source = info['source']
            label = 'fake' if info['label'] == 0 else 'real'
            pool_key = f"{source}_{label}"
            category_counts[pool_key] += len(info['tweets'])
        
        for key, count in category_counts.items():
            print(f"  {key}: {count} tweets")
        
        return news_to_tweets, total_tweets
    
    def estimate_num_users(self, total_tweets):
        """Estimate number of users needed based on tweet distribution"""
        expected_avg_tweets = self.avg_tweets
        base_estimate = total_tweets / expected_avg_tweets
        estimated_users = int(base_estimate * self.buffer_factor)
        
        print(f"\nUser estimation:")
        print(f"  Total tweets to assign: {total_tweets}")
        print(f"  Expected avg tweets per user: {expected_avg_tweets}")
        print(f"  Base estimate: {base_estimate:.0f} users")
        print(f"  Buffer factor: {self.buffer_factor}")
        print(f"  Estimated users needed: {estimated_users}")
        
        return estimated_users
    
    def partition_tweets(self, news_to_tweets, num_partitions):
        """Partition tweets (not news) into N groups to ensure no duplicate assignments"""
        print(f"\nPartitioning tweets into {num_partitions} groups...")
        
        # Collect all tweets with their metadata
        all_tweets = []
        for news_id, news_info in news_to_tweets.items():
            for tweet_id in news_info['tweets']:
                all_tweets.append({
                    'tweet_id': tweet_id,
                    'news_id': news_id,
                    'label': news_info['label'],
                    'source': news_info['source']
                })
        
        # Shuffle for random distribution
        np.random.shuffle(all_tweets)
        
        # Partition tweets
        partition_size = math.ceil(len(all_tweets) / num_partitions)
        partitions = []
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = min((i + 1) * partition_size, len(all_tweets))
            partition_tweets = all_tweets[start_idx:end_idx]
            partitions.append(partition_tweets)
        
        print(f"Partitioned {len(all_tweets)} tweets into {num_partitions} partitions")
        for i, p in enumerate(partitions):
            print(f"  Partition {i}: {len(p)} tweets")
        
        return partitions
    
    def process_partition(self, partition_id, partition_tweets, users_per_partition, avg_tweets, std_tweets):
        """Process one partition: assign tweets to a subset of users"""
        print(f"Processing partition {partition_id} with {len(partition_tweets)} tweets...")
        
        # Generate tweet counts for this partition's users
        tweet_counts = np.random.normal(avg_tweets, std_tweets, users_per_partition)
        tweet_counts = np.clip(np.round(tweet_counts), 1, None).astype(int)
        
        # Create user data for this partition
        user_data = {}
        
        tweet_idx = 0
        for user_idx in range(users_per_partition):
            user_id = f"partition_{partition_id}_user_{user_idx}"
            
            user_tweets = []
            num_tweets_for_user = min(tweet_counts[user_idx], len(partition_tweets) - tweet_idx)
            
            # Assign tweets to this user
            for _ in range(num_tweets_for_user):
                if tweet_idx >= len(partition_tweets):
                    break
                user_tweets.append(partition_tweets[tweet_idx])
                tweet_idx += 1
            
            if user_tweets:  # Only add user if they have tweets
                # Calculate category counts
                gossipcop_fake = sum(1 for t in user_tweets if t['source'] == 'gossipcop' and t['label'] == 0)
                gossipcop_real = sum(1 for t in user_tweets if t['source'] == 'gossipcop' and t['label'] == 1)
                politifact_fake = sum(1 for t in user_tweets if t['source'] == 'politifact' and t['label'] == 0)
                politifact_real = sum(1 for t in user_tweets if t['source'] == 'politifact' and t['label'] == 1)
                
                user_data[user_id] = {
                    'gossipcop_fake': gossipcop_fake,
                    'gossipcop_real': gossipcop_real,
                    'politifact_fake': politifact_fake,
                    'politifact_real': politifact_real,
                    'total_shares': len(user_tweets),
                    'tweets': user_tweets
                }
            
            if tweet_idx >= len(partition_tweets):
                break
        
        # Collect remaining tweets
        remaining_tweets = partition_tweets[tweet_idx:]
        
        print(f"Partition {partition_id}: Created {len(user_data)} users, assigned {tweet_idx}/{len(partition_tweets)} tweets")
        
        return user_data, remaining_tweets
    
    def parallel_generate_users(self, news_to_tweets, num_users, num_partitions):
        """Generate users using parallel processing on tweet partitions"""
        print(f"\nGenerating {num_users} users with {num_partitions} partitions...")
        
        # Partition tweets (not news) to ensure no duplicates
        partitions = self.partition_tweets(news_to_tweets, num_partitions)
        
        # Calculate users per partition
        users_per_partition = math.ceil(num_users / num_partitions)
        
        print(f"\nAssigning ~{users_per_partition} users per partition")
        
        # Process partitions in parallel
        print(f"Processing partitions in parallel using {num_partitions} workers...")
        
        with Pool(processes=num_partitions) as pool:
            process_func = partial(self.process_partition,
                                  users_per_partition=users_per_partition,
                                  avg_tweets=self.avg_tweets,
                                  std_tweets=self.std_tweets)
            
            partition_results = pool.starmap(
                process_func,
                [(i, partition) for i, partition in enumerate(partitions)]
            )
        
        # Merge results from all partitions
        print("\nMerging user data from all partitions...")
        
        all_users = {}
        all_remaining_tweets = []
        user_counter = 0
        
        for user_data, remaining_tweets in partition_results:
            # Rename users to have global IDs
            for old_user_id, data in user_data.items():
                new_user_id = f"user_{user_counter}"
                all_users[new_user_id] = data
                user_counter += 1
            
            all_remaining_tweets.extend(remaining_tweets)
        
        print(f"Merged data for {len(all_users)} users")
        print(f"Total remaining tweets: {len(all_remaining_tweets)}")
        
        # Assign remaining tweets randomly
        if all_remaining_tweets:
            print(f"Assigning {len(all_remaining_tweets)} remaining tweets...")
            user_ids = list(all_users.keys())
            
            for tweet_info in all_remaining_tweets:
                user_id = np.random.choice(user_ids)
                
                # Add tweet
                all_users[user_id]['tweets'].append(tweet_info)
                all_users[user_id]['total_shares'] += 1
                
                # Update category counts
                if tweet_info['source'] == 'gossipcop':
                    if tweet_info['label'] == 0:
                        all_users[user_id]['gossipcop_fake'] += 1
                    else:
                        all_users[user_id]['gossipcop_real'] += 1
                else:  # politifact
                    if tweet_info['label'] == 0:
                        all_users[user_id]['politifact_fake'] += 1
                    else:
                        all_users[user_id]['politifact_real'] += 1
        
        # Create users dataframe
        users_list = []
        for user_id, data in all_users.items():
            users_list.append({
                'user_id': user_id,
                'gossipcop_fake': data['gossipcop_fake'],
                'gossipcop_real': data['gossipcop_real'],
                'politifact_fake': data['politifact_fake'],
                'politifact_real': data['politifact_real'],
                'total_shares': data['total_shares'],
                'tweet_ids': '\t'.join([t['tweet_id'] for t in data['tweets']])
            })
        
        self.users_df = pd.DataFrame(users_list)
        
        # Print actual distribution statistics
        print(f"\nActual tweet distribution statistics:")
        print(f"  Total users created: {len(self.users_df)}")
        print(f"  Mean tweets per user: {self.users_df['total_shares'].mean():.2f}")
        print(f"  Median tweets per user: {self.users_df['total_shares'].median():.2f}")
        print(f"  Std Dev: {self.users_df['total_shares'].std():.2f}")
        print(f"  Min: {self.users_df['total_shares'].min()}")
        print(f"  Max: {self.users_df['total_shares'].max()}")
        
        return self.users_df
    
    def save_users_csv(self, filename='users.csv'):
        """Save users dataframe to CSV"""
        if self.users_df is not None:
            self.users_df.to_csv(filename, index=False)
            print(f"\nUsers saved to {filename}")
            
            # Print statistics
            print("\nUser Statistics:")
            print(f"Total users: {len(self.users_df)}")
            print(f"\nSharing statistics by source and label:")
            print(f"  Gossipcop Fake: {self.users_df['gossipcop_fake'].sum()}")
            print(f"  Gossipcop Real: {self.users_df['gossipcop_real'].sum()}")
            print(f"  Politifact Fake: {self.users_df['politifact_fake'].sum()}")
            print(f"  Politifact Real: {self.users_df['politifact_real'].sum()}")
            print(f"  Total tweets assigned: {self.users_df['total_shares'].sum()}")
            print(f"\nTweet sharing statistics:")
            print(f"  Mean tweets per user: {self.users_df['total_shares'].mean():.2f}")
            print(f"  Median tweets per user: {self.users_df['total_shares'].median():.2f}")
            print(f"  Users with >0 tweets: {len(self.users_df[self.users_df['total_shares'] > 0])}")
            
            return filename
        else:
            print("No users data to save")
            return None
    
    def build_user_news_graph(self, datasets, news_to_tweets):
        """Build bipartite graph with user and news nodes"""
        G = nx.Graph()
        user_sharing_history = defaultdict(lambda: {
            'gossipcop_fake': 0,
            'gossipcop_real': 0,
            'politifact_fake': 0,
            'politifact_real': 0,
            'total': 0
        })
        
        print("\nBuilding user-news bipartite graph...")
        
        # Add news nodes (only those with tweets)
        news_share_counts = defaultdict(int)  # Track actual shares per news
        
        for dataset_name, df in datasets.items():
            label = 1 if 'real' in dataset_name else 0
            source = dataset_name.split('_')[0]
            
            for idx, row in df.iterrows():
                news_id = f"{source}_{row['id']}"
                G.add_node(news_id,
                          node_type='news',
                          title=str(row['title'])[:100] + '...' if len(str(row['title'])) > 100 else str(row['title']),
                          url=row['news_url'],
                          label=label,
                          source=source,
                          ground_truth='fake' if label == 0 else 'real',
                          num_shares=0)  # Will be updated when adding edges
        
        # Build tweet to news mapping
        tweet_to_news = {}
        for news_id, info in news_to_tweets.items():
            for tweet_id in info['tweets']:
                tweet_to_news[tweet_id] = news_id
        
        # Add user nodes and edges
        print("Adding user nodes and edges...")
        edge_count = 0
        
        for _, user_row in self.users_df.iterrows():
            user_id = user_row['user_id']
            
            # Add user node
            G.add_node(user_id, 
                      node_type='user',
                      gossipcop_fake=int(user_row['gossipcop_fake']),
                      gossipcop_real=int(user_row['gossipcop_real']),
                      politifact_fake=int(user_row['politifact_fake']),
                      politifact_real=int(user_row['politifact_real']),
                      total_shares=int(user_row['total_shares']))
            
            # Add edges
            if user_row['tweet_ids']:
                tweet_ids = user_row['tweet_ids'].split('\t')
                
                for tweet_id in tweet_ids:
                    if tweet_id in tweet_to_news:
                        news_id = tweet_to_news[tweet_id]
                        
                        # Only add edge if news node exists
                        if G.has_node(news_id):
                            # Add edge
                            G.add_edge(user_id, news_id, 
                                      relationship='shares',
                                      tweet_id=tweet_id)
                            
                            # Increment share count
                            news_share_counts[news_id] += 1
                            edge_count += 1
                            
                            # Update user sharing history
                            news_label = G.nodes[news_id]['label']
                            news_source = G.nodes[news_id]['source']
                            
                            if news_label == 0:
                                user_sharing_history[user_id][f'{news_source}_fake'] += 1
                            else:
                                user_sharing_history[user_id][f'{news_source}_real'] += 1
                            user_sharing_history[user_id]['total'] += 1
        
        print(f"Added {edge_count} edges")
        
        # Update num_shares for all news nodes based on actual edges
        print("Updating news node share counts...")
        for news_id in [n for n in G.nodes() if G.nodes[n]['node_type'] == 'news']:
            actual_degree = G.degree(news_id)
            G.nodes[news_id]['num_shares'] = actual_degree
        
        # Verify num_shares == degree for all news nodes
        mismatches = 0
        for news_id in [n for n in G.nodes() if G.nodes[n]['node_type'] == 'news']:
            if G.nodes[news_id]['num_shares'] != G.degree(news_id):
                mismatches += 1
        
        if mismatches > 0:
            print(f"WARNING: Found {mismatches} news nodes with num_shares != degree")
        else:
            print("✓ All news nodes have num_shares == degree")
        
        # Remove isolated nodes
        print("\nRemoving isolated nodes (degree 0)...")
        isolated_nodes = list(nx.isolates(G))
        print(f"Found {len(isolated_nodes)} isolated nodes")
        G.remove_nodes_from(isolated_nodes)
        print(f"Graph after cleanup: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        self.G = G
        return G, user_sharing_history
    
    def calculate_trustworthiness(self, user_sharing_history):
        """Calculate user trustworthiness scores"""
        print("\nCalculating user trustworthiness...")
        
        user_trustworthiness = {}
        
        for user_id, history in user_sharing_history.items():
            total_shares = history['total']
            
            if total_shares > 0:
                real_shares = history['gossipcop_real'] + history['politifact_real']
                real_ratio = real_shares / total_shares
                
                confidence = min(total_shares / 10, 1.0)
                trustworthiness = real_ratio * confidence + 0.5 * (1 - confidence)
            else:
                trustworthiness = 0.5
            
            user_trustworthiness[user_id] = trustworthiness
            
            if self.G.has_node(user_id):
                self.G.nodes[user_id]['trustworthiness'] = trustworthiness
        
        self.user_trustworthiness = user_trustworthiness
        return user_trustworthiness
    
    def save_graph(self, filename=None):
        """Save graph to pickle file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fakenews_graph_{timestamp}.pkl'
        
        graph_data = {
            'graph': self.G,
            'users_df': self.users_df,
            'user_trustworthiness': self.user_trustworthiness,
            'stats': self.get_graph_stats()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"\nGraph saved to {filename}")
        return filename
    
    def get_graph_stats(self):
        """Get basic graph statistics"""
        if self.G is None:
            return {}
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'news']
        user_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'user']
        
        fake_news = [n for n in news_nodes if self.G.nodes[n]['label'] == 0]
        real_news = [n for n in news_nodes if self.G.nodes[n]['label'] == 1]
        
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
    
    def export_to_gexf(self, filename='fakenews_network.gexf'):
        """Export graph to GEXF format for Gephi"""
        print(f"\nExporting graph to GEXF format: {filename}")
        
        G_export = self.G.copy()
        
        for node in G_export.nodes():
            node_data = G_export.nodes[node]
            
            if node_data['node_type'] == 'news':
                if node_data['label'] == 0:
                    color = {'r': 255, 'g': 0, 'b': 0}
                else:
                    color = {'r': 0, 'g': 255, 'b': 0}
                
                G_export.nodes[node]['viz'] = {
                    'color': color,
                    'size': min(node_data.get('num_shares', 1) * 2, 50)
                }
            else:
                trust = node_data.get('trustworthiness', 0.5)
                blue_intensity = int(255 * trust)
                G_export.nodes[node]['viz'] = {
                    'color': {'r': 100, 'g': 100, 'b': blue_intensity},
                    'size': min(node_data.get('total_shares', 1) + 5, 20)
                }
        
        nx.write_gexf(G_export, filename)
        print(f"Graph exported to {filename}")
        return filename

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate FakeNews Network Graph')
    parser.add_argument('--num_partitions', type=int, default=cpu_count(),
                       help='Number of parallel partitions (default: CPU count)')
    parser.add_argument('--avg_tweets', type=int, default=13,
                       help='Average tweets per user (default: 13)')
    parser.add_argument('--std_tweets', type=int, default=10,
                       help='Standard deviation for tweets per user (default: 10)')
    parser.add_argument('--buffer_factor', type=float, default=1.5,
                       help='User count buffer factor (default: 1.5)')
    parser.add_argument('--skip_gexf', action='store_true',
                       help='Skip GEXF export (saves time)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FAKENEWS NETWORK BUILDER - NEURAL NETWORK VERSION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Num partitions: {args.num_partitions}")
    print(f"  Avg tweets per user: {args.avg_tweets}")
    print(f"  Std tweets per user: {args.std_tweets}")
    print(f"  Buffer factor: {args.buffer_factor}")
    print(f"  Skip GEXF: {args.skip_gexf}")
    print("="*60)
    
    # Initialize builder
    builder = FakeNewsNetworkBuilderNN(
        dataset_path='../external/FakeNewsNet/dataset',
        avg_tweets=args.avg_tweets,
        std_tweets=args.std_tweets,
        buffer_factor=args.buffer_factor
    )
    
    # Load datasets (automatically filters out news without tweets)
    datasets = builder.load_datasets()
    if not datasets:
        print("No datasets loaded. Check your dataset path.")
        return
    
    # Create tweet pools (now ensures no duplicates)
    news_to_tweets, total_tweets = builder.create_tweet_pools(datasets)
    
    # Estimate number of users
    num_users = builder.estimate_num_users(total_tweets)
    
    # Generate users in parallel (partitions tweets, not news)
    users_df = builder.parallel_generate_users(news_to_tweets, num_users, args.num_partitions)
    
    # Save users to CSV
    users_file = builder.save_users_csv()
    
    # Build graph
    G, user_sharing_history = builder.build_user_news_graph(datasets, news_to_tweets)
    
    # Calculate trustworthiness
    user_trustworthiness = builder.calculate_trustworthiness(user_sharing_history)
    
    # Print statistics
    stats = builder.get_graph_stats()
    print("\n" + "="*60)
    print("GRAPH STATISTICS")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Print NN features
    print("\n" + "="*60)
    print("NN TRAINING FEATURES (per user node)")
    print("="*60)
    print("1. gossipcop_fake")
    print("2. gossipcop_real")
    print("3. politifact_fake")
    print("4. politifact_real")
    print("5. total_shares")
    print("\nTarget: News label (0=fake, 1=real)")
    
    # Save graph
    saved_file = builder.save_graph()
    
    # Export to GEXF
    if not args.skip_gexf:
        print("\n" + "="*60)
        print("EXPORTING TO GEPHI")
        print("="*60)
        gexf_file = builder.export_to_gexf()
    else:
        print("\nSkipping GEXF export (--skip_gexf flag set)")
        gexf_file = None
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Users CSV: {users_file}")
    print(f"Graph saved to: {saved_file}")
    if gexf_file:
        print(f"GEXF file for Gephi: {gexf_file}")
    
    return builder

if __name__ == "__main__":
    builder = main()