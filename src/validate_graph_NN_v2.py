import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import argparse
from datetime import datetime

"""
Validate the FakeNews Network Graph Dataset

Usage:
python validate_graph.py fakenews_graph.pkl --output validation_report

This script validates:
- Node attributes and types
- Edge properties
- Data consistency
- Statistical distributions
- Missing or invalid values
"""

class GraphValidator:
    def __init__(self, graph_file):
        """Load the saved graph from pickle file"""
        print(f"Loading graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        self.G = graph_data['graph']
        self.users_df = graph_data.get('users_df')
        self.user_trustworthiness = graph_data.get('user_trustworthiness', {})
        self.stats = graph_data.get('stats', {})
        
        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        self.validation_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'errors': [],
            'warnings': [],
            'info': []
        }
    
    def validate_node_attributes(self):
        """Validate that all nodes have required attributes"""
        print("\n" + "="*60)
        print("VALIDATING NODE ATTRIBUTES")
        print("="*60)
        
        required_news_attrs = ['node_type', 'title', 'url', 'label', 'source', 'num_shares']
        required_user_attrs = ['node_type', 'gossipcop_fake', 'gossipcop_real', 
                              'politifact_fake', 'politifact_real', 'total_shares']
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'news']
        user_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'user']
        
        # Validate news nodes
        print(f"\nValidating {len(news_nodes)} news nodes...")
        news_missing_attrs = defaultdict(list)
        news_invalid_values = defaultdict(list)
        
        for node in news_nodes:
            node_data = self.G.nodes[node]
            
            # Check required attributes
            for attr in required_news_attrs:
                if attr not in node_data:
                    news_missing_attrs[attr].append(node)
            
            # Validate label values
            if 'label' in node_data:
                if node_data['label'] not in [0, 1]:
                    news_invalid_values['label'].append((node, node_data['label']))
            
            # Validate source values
            if 'source' in node_data:
                if node_data['source'] not in ['gossipcop', 'politifact']:
                    news_invalid_values['source'].append((node, node_data['source']))
            
            # Validate num_shares
            if 'num_shares' in node_data:
                if not isinstance(node_data['num_shares'], (int, np.integer)):
                    news_invalid_values['num_shares'].append((node, node_data['num_shares']))
                elif node_data['num_shares'] < 0:
                    news_invalid_values['num_shares'].append((node, node_data['num_shares']))
        
        # Validate user nodes
        print(f"Validating {len(user_nodes)} user nodes...")
        user_missing_attrs = defaultdict(list)
        user_invalid_values = defaultdict(list)
        
        for node in user_nodes:
            node_data = self.G.nodes[node]
            
            # Check required attributes
            for attr in required_user_attrs:
                if attr not in node_data:
                    user_missing_attrs[attr].append(node)
            
            # Validate numeric values
            for attr in ['gossipcop_fake', 'gossipcop_real', 'politifact_fake', 
                        'politifact_real', 'total_shares']:
                if attr in node_data:
                    if not isinstance(node_data[attr], (int, np.integer)):
                        user_invalid_values[attr].append((node, node_data[attr]))
                    elif node_data[attr] < 0:
                        user_invalid_values[attr].append((node, node_data[attr]))
            
            # Validate total_shares consistency
            if all(attr in node_data for attr in required_user_attrs[1:]):
                calculated_total = (node_data['gossipcop_fake'] + 
                                  node_data['gossipcop_real'] + 
                                  node_data['politifact_fake'] + 
                                  node_data['politifact_real'])
                if calculated_total != node_data['total_shares']:
                    user_invalid_values['total_shares_mismatch'].append(
                        (node, calculated_total, node_data['total_shares'])
                    )
        
        # Report results
        if news_missing_attrs:
            for attr, nodes in news_missing_attrs.items():
                error_msg = f"News nodes missing '{attr}': {len(nodes)} nodes"
                print(f"  ERROR: {error_msg}")
                self.validation_results['errors'].append(error_msg)
        
        if news_invalid_values:
            for attr, values in news_invalid_values.items():
                error_msg = f"News nodes with invalid '{attr}': {len(values)} nodes"
                print(f"  ERROR: {error_msg}")
                self.validation_results['errors'].append(error_msg)
        
        if user_missing_attrs:
            for attr, nodes in user_missing_attrs.items():
                error_msg = f"User nodes missing '{attr}': {len(nodes)} nodes"
                print(f"  ERROR: {error_msg}")
                self.validation_results['errors'].append(error_msg)
        
        if user_invalid_values:
            for attr, values in user_invalid_values.items():
                error_msg = f"User nodes with invalid '{attr}': {len(values)} nodes"
                print(f"  ERROR: {error_msg}")
                self.validation_results['errors'].append(error_msg)
        
        if not (news_missing_attrs or news_invalid_values or 
                user_missing_attrs or user_invalid_values):
            print("  ✓ All node attributes are valid")
            self.validation_results['info'].append("All node attributes are valid")
        
        return {
            'news_missing_attrs': dict(news_missing_attrs),
            'news_invalid_values': dict(news_invalid_values),
            'user_missing_attrs': dict(user_missing_attrs),
            'user_invalid_values': dict(user_invalid_values)
        }
    
    def validate_edge_properties(self):
        """Validate edge properties and bipartite structure"""
        print("\n" + "="*60)
        print("VALIDATING EDGE PROPERTIES")
        print("="*60)
        
        invalid_edges = []
        
        for u, v in self.G.edges():
            u_type = self.G.nodes[u].get('node_type')
            v_type = self.G.nodes[v].get('node_type')
            
            # Check bipartite structure (user-news only)
            if u_type == v_type:
                invalid_edges.append((u, v, u_type, v_type))
        
        if invalid_edges:
            error_msg = f"Found {len(invalid_edges)} edges between same node types (should be bipartite)"
            print(f"  ERROR: {error_msg}")
            self.validation_results['errors'].append(error_msg)
        else:
            print("  ✓ Graph is properly bipartite")
            self.validation_results['info'].append("Graph is properly bipartite")
        
        return {'invalid_edges': invalid_edges}
    
    def validate_degree_consistency(self):
        """Validate that node degrees match edge counts"""
        print("\n" + "="*60)
        print("VALIDATING DEGREE CONSISTENCY")
        print("="*60)
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'news']
        
        degree_mismatches = []
        
        for node in news_nodes:
            num_shares = self.G.nodes[node].get('num_shares', 0)
            actual_degree = self.G.degree(node)
            
            if num_shares != actual_degree:
                degree_mismatches.append((node, num_shares, actual_degree))
        
        if degree_mismatches:
            error_msg = f"Found {len(degree_mismatches)} news nodes with num_shares != degree"
            print(f"  ERROR: {error_msg}")
            self.validation_results['errors'].append(error_msg)
        else:
            print("  ✓ All news node degrees match num_shares")
            self.validation_results['info'].append("All news node degrees match num_shares")
        
        return {'degree_mismatches': degree_mismatches}
    
    def validate_isolated_nodes(self):
        """Check for isolated nodes (degree 0)"""
        print("\n" + "="*60)
        print("VALIDATING ISOLATED NODES")
        print("="*60)
        
        isolated = list(nx.isolates(self.G))
        
        if isolated:
            warning_msg = f"Found {len(isolated)} isolated nodes (degree 0)"
            print(f"  WARNING: {warning_msg}")
            self.validation_results['warnings'].append(warning_msg)
        else:
            print("  ✓ No isolated nodes found")
            self.validation_results['info'].append("No isolated nodes found")
        
        return {'isolated_nodes': isolated}
    
    def validate_label_distribution(self):
        """Validate news label distribution"""
        print("\n" + "="*60)
        print("VALIDATING LABEL DISTRIBUTION")
        print("="*60)
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'news']
        
        labels = [self.G.nodes[n].get('label') for n in news_nodes]
        label_counts = Counter(labels)
        
        print(f"  Label 0 (fake): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(news_nodes)*100:.2f}%)")
        print(f"  Label 1 (real): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(news_nodes)*100:.2f}%)")
        
        # Check for imbalance
        if label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
            error_msg = "Severe class imbalance: one class has 0 samples"
            print(f"  ERROR: {error_msg}")
            self.validation_results['errors'].append(error_msg)
        elif abs(label_counts.get(0, 0) - label_counts.get(1, 0)) / len(news_nodes) > 0.8:
            warning_msg = "High class imbalance detected"
            print(f"  WARNING: {warning_msg}")
            self.validation_results['warnings'].append(warning_msg)
        else:
            print("  ✓ Label distribution is reasonable")
            self.validation_results['info'].append("Label distribution is reasonable")
        
        return {'label_distribution': dict(label_counts)}
    
    def validate_source_distribution(self):
        """Validate source distribution"""
        print("\n" + "="*60)
        print("VALIDATING SOURCE DISTRIBUTION")
        print("="*60)
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'news']
        
        sources = [self.G.nodes[n].get('source') for n in news_nodes]
        source_counts = Counter(sources)
        
        for source, count in source_counts.items():
            print(f"  {source}: {count} ({count/len(news_nodes)*100:.2f}%)")
        
        if None in source_counts:
            error_msg = f"Found {source_counts[None]} news nodes with missing source"
            print(f"  ERROR: {error_msg}")
            self.validation_results['errors'].append(error_msg)
        
        return {'source_distribution': dict(source_counts)}
    
    def compute_statistics(self):
        """Compute comprehensive graph statistics"""
        print("\n" + "="*60)
        print("COMPUTING GRAPH STATISTICS")
        print("="*60)
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'news']
        user_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'user']
        
        news_degrees = [self.G.degree(n) for n in news_nodes]
        user_degrees = [self.G.degree(n) for n in user_nodes]
        
        stats = {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'news_nodes': len(news_nodes),
            'user_nodes': len(user_nodes),
            'news_degree_mean': np.mean(news_degrees) if news_degrees else 0,
            'news_degree_median': np.median(news_degrees) if news_degrees else 0,
            'news_degree_std': np.std(news_degrees) if news_degrees else 0,
            'news_degree_min': min(news_degrees) if news_degrees else 0,
            'news_degree_max': max(news_degrees) if news_degrees else 0,
            'user_degree_mean': np.mean(user_degrees) if user_degrees else 0,
            'user_degree_median': np.median(user_degrees) if user_degrees else 0,
            'user_degree_std': np.std(user_degrees) if user_degrees else 0,
            'user_degree_min': min(user_degrees) if user_degrees else 0,
            'user_degree_max': max(user_degrees) if user_degrees else 0,
        }
        
        # Label distribution
        fake_news = [n for n in news_nodes if self.G.nodes[n].get('label') == 0]
        real_news = [n for n in news_nodes if self.G.nodes[n].get('label') == 1]
        stats['fake_news'] = len(fake_news)
        stats['real_news'] = len(real_news)
        
        # Source distribution
        gossipcop_news = [n for n in news_nodes if self.G.nodes[n].get('source') == 'gossipcop']
        politifact_news = [n for n in news_nodes if self.G.nodes[n].get('source') == 'politifact']
        stats['gossipcop_news'] = len(gossipcop_news)
        stats['politifact_news'] = len(politifact_news)
        
        # User sharing statistics
        if user_nodes:
            total_shares = [self.G.nodes[n].get('total_shares', 0) for n in user_nodes]
            stats['user_shares_mean'] = np.mean(total_shares)
            stats['user_shares_median'] = np.median(total_shares)
            stats['user_shares_std'] = np.std(total_shares)
        
        return stats
    
    def plot_validation_figures(self, output_prefix='validation'):
        """Generate validation plots"""
        print("\n" + "="*60)
        print("GENERATING VALIDATION FIGURES")
        print("="*60)
        
        # Font configuration
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans'],
            'font.size': 12,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
        })
        
        news_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'news']
        user_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'user']
        
        # 1. News degree distribution
        news_degrees = [self.G.degree(n) for n in news_nodes]
        degree_counts = Counter(news_degrees)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        degrees = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees]
        ax.bar(degrees, counts, alpha=0.7, color='green', edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel('Degree (k)')
        ax.set_ylabel('Frequency')
        ax.set_title('News Node Degree Distribution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        news_deg_file = f"{output_prefix}_news_degree.pdf"
        plt.savefig(news_deg_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {news_deg_file}")
        
        # 2. User degree distribution
        user_degrees = [self.G.degree(n) for n in user_nodes]
        degree_counts = Counter(user_degrees)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        degrees = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees]
        ax.bar(degrees, counts, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Degree (k)')
        ax.set_ylabel('Frequency')
        ax.set_title('User Node Degree Distribution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        user_deg_file = f"{output_prefix}_user_degree.pdf"
        plt.savefig(user_deg_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {user_deg_file}")
        
        # 3. Label distribution
        labels = [self.G.nodes[n].get('label') for n in news_nodes]
        label_counts = Counter(labels)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        labels_list = ['Fake (0)', 'Real (1)']
        counts_list = [label_counts.get(0, 0), label_counts.get(1, 0)]
        colors = ['red', 'green']
        ax.bar(labels_list, counts_list, alpha=0.7, color=colors, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('News Label Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        label_dist_file = f"{output_prefix}_label_distribution.pdf"
        plt.savefig(label_dist_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {label_dist_file}")
        
        # 4. Source distribution
        sources = [self.G.nodes[n].get('source') for n in news_nodes]
        source_counts = Counter(sources)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        source_names = list(source_counts.keys())
        source_values = list(source_counts.values())
        ax.bar(source_names, source_values, alpha=0.7, color='orange', edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('News Source Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        source_dist_file = f"{output_prefix}_source_distribution.pdf"
        plt.savefig(source_dist_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {source_dist_file}")
        
        # 5. User sharing statistics (4 categories)
        gossipcop_fake = [self.G.nodes[n].get('gossipcop_fake', 0) for n in user_nodes]
        gossipcop_real = [self.G.nodes[n].get('gossipcop_real', 0) for n in user_nodes]
        politifact_fake = [self.G.nodes[n].get('politifact_fake', 0) for n in user_nodes]
        politifact_real = [self.G.nodes[n].get('politifact_real', 0) for n in user_nodes]
        
        fig, ax = plt.subplots(figsize=(7, 4))
        categories = ['Gossipcop\nFake', 'Gossipcop\nReal', 'Politifact\nFake', 'Politifact\nReal']
        means = [np.mean(gossipcop_fake), np.mean(gossipcop_real), 
                np.mean(politifact_fake), np.mean(politifact_real)]
        colors_cat = ['salmon', 'lightgreen', 'coral', 'darkseagreen']
        ax.bar(categories, means, alpha=0.7, color=colors_cat, edgecolor='black')
        ax.set_ylabel('Mean Shares per User')
        ax.set_title('User Sharing Behavior by Category')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        user_sharing_file = f"{output_prefix}_user_sharing.pdf"
        plt.savefig(user_sharing_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {user_sharing_file}")
        
        return {
            'news_degree': news_deg_file,
            'user_degree': user_deg_file,
            'label_distribution': label_dist_file,
            'source_distribution': source_dist_file,
            'user_sharing': user_sharing_file
        }
    
    def write_validation_report(self, output_file='validation_report.txt'):
        """Write comprehensive validation report to text file"""
        print("\n" + "="*60)
        print("WRITING VALIDATION REPORT")
        print("="*60)
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FAKE NEWS NETWORK GRAPH VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {self.validation_results['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            f.write("VALIDATION SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Errors: {len(self.validation_results['errors'])}\n")
            f.write(f"Total Warnings: {len(self.validation_results['warnings'])}\n")
            f.write(f"Total Info: {len(self.validation_results['info'])}\n")
            f.write("\n")
            
            # Errors
            if self.validation_results['errors']:
                f.write("ERRORS\n")
                f.write("-"*80 + "\n")
                for i, error in enumerate(self.validation_results['errors'], 1):
                    f.write(f"{i}. {error}\n")
                f.write("\n")
            
            # Warnings
            if self.validation_results['warnings']:
                f.write("WARNINGS\n")
                f.write("-"*80 + "\n")
                for i, warning in enumerate(self.validation_results['warnings'], 1):
                    f.write(f"{i}. {warning}\n")
                f.write("\n")
            
            # Graph Statistics
            stats = self.compute_statistics()
            f.write("GRAPH STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Nodes: {stats['total_nodes']}\n")
            f.write(f"Total Edges: {stats['total_edges']}\n")
            f.write(f"News Nodes: {stats['news_nodes']}\n")
            f.write(f"User Nodes: {stats['user_nodes']}\n")
            f.write(f"Fake News: {stats['fake_news']}\n")
            f.write(f"Real News: {stats['real_news']}\n")
            f.write(f"Gossipcop News: {stats['gossipcop_news']}\n")
            f.write(f"Politifact News: {stats['politifact_news']}\n")
            f.write("\n")
            
            # News Degree Statistics
            f.write("NEWS NODE DEGREE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean: {stats['news_degree_mean']:.2f}\n")
            f.write(f"Median: {stats['news_degree_median']:.2f}\n")
            f.write(f"Std Dev: {stats['news_degree_std']:.2f}\n")
            f.write(f"Min: {stats['news_degree_min']}\n")
            f.write(f"Max: {stats['news_degree_max']}\n")
            f.write("\n")
            
            # User Degree Statistics
            f.write("USER NODE DEGREE STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean: {stats['user_degree_mean']:.2f}\n")
            f.write(f"Median: {stats['user_degree_median']:.2f}\n")
            f.write(f"Std Dev: {stats['user_degree_std']:.2f}\n")
            f.write(f"Min: {stats['user_degree_min']}\n")
            f.write(f"Max: {stats['user_degree_max']}\n")
            f.write("\n")
            
            # User Sharing Statistics
            if 'user_shares_mean' in stats:
                f.write("USER SHARING STATISTICS\n")
                f.write("-"*80 + "\n")
                f.write(f"Mean Shares per User: {stats['user_shares_mean']:.2f}\n")
                f.write(f"Median Shares per User: {stats['user_shares_median']:.2f}\n")
                f.write(f"Std Dev: {stats['user_shares_std']:.2f}\n")
                f.write("\n")
            
            # Node Attributes Sample
            f.write("SAMPLE NODE ATTRIBUTES\n")
            f.write("-"*80 + "\n")
            
            news_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'news']
            if news_nodes:
                sample_news = news_nodes[0]
                f.write(f"Sample News Node: {sample_news}\n")
                for key, value in self.G.nodes[sample_news].items():
                    if key == 'title':
                        f.write(f"  {key}: {str(value)[:100]}...\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            user_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('node_type') == 'user']
            if user_nodes:
                sample_user = user_nodes[0]
                f.write(f"Sample User Node: {sample_user}\n")
                for key, value in self.G.nodes[sample_user].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Final Status
            f.write("="*80 + "\n")
            if not self.validation_results['errors']:
                f.write("VALIDATION STATUS: PASSED ✓\n")
            else:
                f.write("VALIDATION STATUS: FAILED ✗\n")
            f.write("="*80 + "\n")
        
        print(f"  Validation report saved to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description='Validate FakeNews Network Graph Dataset')
    parser.add_argument('graph_file', type=str, help='Path to the pickle file containing the graph')
    parser.add_argument('--output', '-o', type=str, default='validation_report',
                       help='Output prefix for validation files (default: validation_report)')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = GraphValidator(args.graph_file)
    
    # Run validations
    validator.validate_node_attributes()
    validator.validate_edge_properties()
    validator.validate_degree_consistency()
    validator.validate_isolated_nodes()
    validator.validate_label_distribution()
    validator.validate_source_distribution()
    
    # Generate plots
    plot_files = validator.plot_validation_figures(output_prefix=args.output)
    
    # Write report
    report_file = validator.write_validation_report(output_file=f"{args.output}.txt")
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"Report: {report_file}")
    print("Figures:")
    for name, path in plot_files.items():
        print(f"  {name}: {path}")
    
    print("\n" + "="*60)
    if not validator.validation_results['errors']:
        print("VALIDATION STATUS: PASSED ✓")
    else:
        print("VALIDATION STATUS: FAILED ✗")
        print(f"Errors found: {len(validator.validation_results['errors'])}")
    print("="*60)

if __name__ == "__main__":
    main()
