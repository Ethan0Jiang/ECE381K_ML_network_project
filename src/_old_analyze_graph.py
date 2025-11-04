import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import argparse
import random
from datetime import datetime
"""
Get 5 subgraphs from nodes with degree 1-10, distance 3, max 10 users per news
python analyze_graph.py fakenews_graph_20251025_153015.pkl \
    --min-degree 1 \
    --max-degree 10 \
    --top-n 5 \
    --distance 3 \
    --max-users-per-news 10 \
    --output-prefix small_cluster
"""

class GraphAnalyzer:
    def __init__(self, graph_file):
        """Load the saved graph from pickle file"""
        print(f"Loading graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.G = graph_data['graph']
        self.users_df = graph_data.get('users_df')
        self.user_trustworthiness = graph_data.get('user_trustworthiness', {})
        
        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def get_news_nodes(self):
        """Get all news nodes from the graph"""
        return [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'news']
    
    def get_user_nodes(self):
        """Get all user nodes from the graph"""
        return [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'user']
    
    def get_news_by_degree_range(self, min_degree, max_degree, top_n=1):
        """Get top_n news nodes within a specific degree range"""
        news_nodes = self.get_news_nodes()
        
        # Get degrees for all news nodes
        news_degrees = [(n, self.G.degree(n)) for n in news_nodes]
        
        # Filter by degree range
        filtered = [(n, d) for n, d in news_degrees if min_degree <= d <= max_degree]
        
        if not filtered:
            print(f"Warning: No news nodes found in degree range [{min_degree}, {max_degree}]")
            return []
        
        # Sort by degree (descending) and take top_n
        filtered.sort(key=lambda x: x[1], reverse=True)
        selected = [n for n, d in filtered[:top_n]]
        
        print(f"\nSelected {len(selected)} news node(s) from degree range [{min_degree}, {max_degree}]:")
        for node in selected:
            degree = self.G.degree(node)
            node_data = self.G.nodes[node]
            print(f"  Node: {node}")
            print(f"    Degree: {degree}")
            print(f"    Ground truth: {node_data['ground_truth']}")
            print(f"    Title: {node_data.get('title', 'N/A')[:80]}...")
        
        return selected
    
    def extract_subgraph(self, center_node, distance=5, max_users_per_news=10):
        """Extract subgraph within distance n from center node with user limit"""
        print(f"\nExtracting subgraph around node: {center_node}")
        print(f"Distance: {distance}")
        print(f"Max users per news node: {max_users_per_news}")
        
        # Get all nodes within distance n using BFS with user limitation
        nodes_in_range = set([center_node])
        current_level = {center_node}
        
        for d in range(distance):
            next_level = set()
            for node in current_level:
                neighbors = list(self.G.neighbors(node))
                
                # If current node is a news node, limit user connections
                if self.G.nodes[node]['node_type'] == 'news':
                    user_neighbors = [n for n in neighbors if self.G.nodes[n]['node_type'] == 'user']
                    news_neighbors = [n for n in neighbors if self.G.nodes[n]['node_type'] == 'news']
                    
                    # Limit user neighbors
                    if len(user_neighbors) > max_users_per_news:
                        user_neighbors = random.sample(user_neighbors, max_users_per_news)
                    
                    neighbors = user_neighbors + news_neighbors
                
                next_level.update(neighbors)
            
            nodes_in_range.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        # Create subgraph
        subgraph = self.G.subgraph(nodes_in_range).copy()
        
        print(f"Subgraph extracted: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        # Print node type distribution
        news_count = sum(1 for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'news')
        user_count = sum(1 for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'user')
        print(f"  News nodes: {news_count}")
        print(f"  User nodes: {user_count}")
        
        return subgraph, nodes_in_range
    
    def plot_subgraph(self, subgraph, center_node, distance, output_file=None):
        """Plot the subgraph with different colors for node types"""
        # Set font sizes optimized for IEEE 2-column format
        title_fontsize = 14
        label_fontsize = 13
        tick_fontsize = 12
        
        # Configure matplotlib for IEEE paper quality
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans'],
            'font.size': tick_fontsize,
            'axes.labelsize': label_fontsize,
            'axes.titlesize': title_fontsize,
            'xtick.labelsize': tick_fontsize,
            'ytick.labelsize': tick_fontsize,
            'figure.titlesize': title_fontsize,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.0,
        })
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Separate nodes by type
        news_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'news']
        user_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'user']
        
        # Create layout
        print("Computing layout...")
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=ax)
        
        # Draw news nodes (no special color for center node)
        news_colors = []
        for node in news_nodes:
            if subgraph.nodes[node]['ground_truth'] == 'fake':
                news_colors.append('red')
            else:
                news_colors.append('green')
        
        nx.draw_networkx_nodes(subgraph, pos, nodelist=news_nodes,
                               node_color=news_colors, node_size=100,
                               node_shape='s', alpha=0.8, ax=ax, label='News')
        
        # Draw user nodes (colored by trustworthiness)
        user_trust_scores = [subgraph.nodes[n].get('trustworthiness', 0.5) for n in user_nodes]
        
        nx.draw_networkx_nodes(subgraph, pos, nodelist=user_nodes,
                               node_color=user_trust_scores, cmap='Blues',
                               node_size=50, alpha=0.7, ax=ax,
                               vmin=0, vmax=1, label='Users')
        
        # Add colorbar for user trustworthiness
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('User Trustworthiness', fontsize=label_fontsize)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Fake News'),
            Patch(facecolor='green', alpha=0.8, label='Real News'),
            Patch(facecolor='lightblue', alpha=0.7, label='Users')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=tick_fontsize)
        
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Subgraph plot saved to {output_file}")
        plt.close()
        
        return output_file
    
    def plot_news_degree_distribution(self, output_file=None):
        """Plot degree distribution for news nodes with log scale on x-axis"""
        # Set font sizes optimized for IEEE 2-column format
        title_fontsize = 14
        label_fontsize = 13
        tick_fontsize = 12
        
        # Configure matplotlib for IEEE paper quality
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans'],
            'font.size': tick_fontsize,
            'axes.labelsize': label_fontsize,
            'axes.titlesize': title_fontsize,
            'xtick.labelsize': tick_fontsize,
            'ytick.labelsize': tick_fontsize,
            'figure.titlesize': title_fontsize,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.0,
        })
        
        # Get degrees for news nodes
        news_nodes = self.get_news_nodes()
        news_degrees = [self.G.degree(n) for n in news_nodes]
        
        # Count degree frequencies
        news_degree_count = Counter(news_degrees)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Sort degrees for plotting
        degrees = sorted(news_degree_count.keys())
        counts = [news_degree_count[d] for d in degrees]
        
        # Plot with log scale on x-axis
        ax.bar(degrees, counts, alpha=0.7, color='green', edgecolor='black', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Degree (k)', fontsize=label_fontsize)
        ax.set_ylabel('Frequency', fontsize=label_fontsize)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.text(0.95, 0.95, 'News Nodes', transform=ax.transAxes,
                fontsize=label_fontsize, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\nNews degree distribution plot saved to {output_file}")
        plt.close()
        
        # Print statistics
        print("\nNews Degree Distribution Statistics:")
        print(f"  Mean degree: {np.mean(news_degrees):.2f}")
        print(f"  Median degree: {np.median(news_degrees):.2f}")
        print(f"  Max degree: {max(news_degrees)}")
        print(f"  Min degree: {min(news_degrees)}")
        
        return output_file
    
    def plot_user_degree_distribution(self, output_file=None):
        """Plot degree distribution for user nodes"""
        # Set font sizes optimized for IEEE 2-column format
        title_fontsize = 14
        label_fontsize = 13
        tick_fontsize = 12
        
        # Configure matplotlib for IEEE paper quality
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans'],
            'font.size': tick_fontsize,
            'axes.labelsize': label_fontsize,
            'axes.titlesize': title_fontsize,
            'xtick.labelsize': tick_fontsize,
            'ytick.labelsize': tick_fontsize,
            'figure.titlesize': title_fontsize,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.0,
        })
        
        # Get degrees for user nodes
        user_nodes = self.get_user_nodes()
        user_degrees = [self.G.degree(n) for n in user_nodes]
        
        # Count degree frequencies
        user_degree_count = Counter(user_degrees)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Sort degrees for plotting
        degrees = sorted(user_degree_count.keys())
        counts = [user_degree_count[d] for d in degrees]
        
        # Plot regular scale
        ax.bar(degrees, counts, alpha=0.7, color='blue', edgecolor='black', linewidth=0.8)
        ax.set_xlabel('Degree (k)', fontsize=label_fontsize)
        ax.set_ylabel('Frequency', fontsize=label_fontsize)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.text(0.95, 0.95, 'User Nodes', transform=ax.transAxes,
                fontsize=label_fontsize, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        print(f"\nUser degree distribution plot saved to {output_file}")
        plt.close()
        
        # Print statistics
        print("\nUser Degree Distribution Statistics:")
        print(f"  Mean degree: {np.mean(user_degrees):.2f}")
        print(f"  Median degree: {np.median(user_degrees):.2f}")
        print(f"  Max degree: {max(user_degrees)}")
        print(f"  Min degree: {min(user_degrees)}")
        
        return output_file
    
    def print_degree_statistics(self):
        """Print degree range statistics for news nodes"""
        news_nodes = self.get_news_nodes()
        news_degrees = [self.G.degree(n) for n in news_nodes]
        
        print("\n" + "="*60)
        print("NEWS DEGREE RANGE STATISTICS")
        print("="*60)
        print(f"Total news nodes: {len(news_nodes)}")
        print(f"Min degree: {min(news_degrees)}")
        print(f"Max degree: {max(news_degrees)}")
        print(f"Mean degree: {np.mean(news_degrees):.2f}")
        print(f"Median degree: {np.median(news_degrees):.2f}")
        
        # Print quartiles
        quartiles = np.percentile(news_degrees, [25, 50, 75])
        print(f"\nQuartiles:")
        print(f"  Q1 (25%): {quartiles[0]:.0f}")
        print(f"  Q2 (50%): {quartiles[1]:.0f}")
        print(f"  Q3 (75%): {quartiles[2]:.0f}")
        
        # Print distribution by ranges
        ranges = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, max(news_degrees))]
        print(f"\nDistribution by degree ranges:")
        for min_d, max_d in ranges:
            count = sum(1 for d in news_degrees if min_d <= d <= max_d)
            percentage = (count / len(news_degrees)) * 100
            print(f"  [{min_d:3d}, {max_d:3d}]: {count:5d} nodes ({percentage:5.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize fake news network graph')
    parser.add_argument('graph_file', type=str, help='Path to the pickle file containing the graph')
    parser.add_argument('--distance', '-d', type=int, default=5,
                       help='Distance for subgraph extraction (default: 5)')
    parser.add_argument('--max-users-per-news', '-u', type=int, default=10,
                       help='Maximum users per news node in subgraph (default: 10)')
    parser.add_argument('--min-degree', type=int, default=1,
                       help='Minimum degree for news node selection (default: 1)')
    parser.add_argument('--max-degree', type=int, default=10,
                       help='Maximum degree for news node selection (default: 10)')
    parser.add_argument('--top-n', '-n', type=int, default=1,
                       help='Number of top nodes to select from degree range (default: 1)')
    parser.add_argument('--output-prefix', '-o', type=str, default='subgraph',
                       help='Prefix for output files (default: subgraph)')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show degree statistics to help choose degree range')
    
    args = parser.parse_args()
    
    # Load graph
    analyzer = GraphAnalyzer(args.graph_file)
    
    # Show degree statistics if requested
    if args.show_stats:
        analyzer.print_degree_statistics()
        return
    
    # Get news nodes in specified degree range
    selected_nodes = analyzer.get_news_by_degree_range(
        args.min_degree, 
        args.max_degree, 
        args.top_n
    )
    
    if not selected_nodes:
        print("No nodes found in specified degree range. Exiting.")
        return
    
    # Process each selected node
    for i, center_node in enumerate(selected_nodes):
        # Extract subgraph
        subgraph, nodes_in_range = analyzer.extract_subgraph(
            center_node, 
            args.distance,
            args.max_users_per_news
        )
        
        # Generate output filename with parameters
        degree = analyzer.G.degree(center_node)
        subgraph_file = f"{args.output_prefix}_k{args.min_degree}-{args.max_degree}_d{args.distance}_u{args.max_users_per_news}_node{i+1}_deg{degree}.pdf"
        
        # Plot subgraph
        analyzer.plot_subgraph(subgraph, center_node, args.distance, subgraph_file)
    
    # Plot degree distributions (separate PDFs)
    news_degree_file = f"{args.output_prefix}_news_degree_distribution.pdf"
    user_degree_file = f"{args.output_prefix}_user_degree_distribution.pdf"
    
    analyzer.plot_news_degree_distribution(news_degree_file)
    analyzer.plot_user_degree_distribution(user_degree_file)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"News degree distribution: {news_degree_file}")
    print(f"User degree distribution: {user_degree_file}")

if __name__ == "__main__":
    main()