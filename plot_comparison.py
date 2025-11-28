"""
Comparison Plotting for Federated Learning Experiments
Visualizes metrics across all experiments
"""

import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_experiment_metrics(metrics_dir):
    """Load all experiment metrics from JSON files"""
    metrics_dir = Path(metrics_dir)
    experiments = {}
    
    for exp_dir in sorted(metrics_dir.iterdir()):
        if exp_dir.is_dir():
            json_files = list(exp_dir.glob("*.json"))
            if json_files:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    exp_name = exp_dir.name
                    experiments[exp_name] = data
    
    return experiments


def load_training_history(results_dir):
    """Load training history for all experiments"""
    results_dir = Path(results_dir)
    histories = {}
    
    for exp_dir in sorted(results_dir.iterdir()):
        if exp_dir.is_dir():
            history_file = exp_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    exp_name = exp_dir.name
                    histories[exp_name] = data
    
    return histories


def plot_fid_comparison(experiments, output_path):
    """Plot FID scores across experiments"""
    exp_names = []
    fid_scores = []
    
    # Sort experiments
    for name in sorted(experiments.keys()):
        exp_names.append(name.replace('_', '\n'))
        fid_scores.append(experiments[name]['fid_score'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(exp_names)))
    bars = ax.bar(range(len(exp_names)), fid_scores, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('FID Score Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, fid_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_inception_comparison(experiments, output_path):
    """Plot Inception scores across experiments"""
    exp_names = []
    is_means = []
    is_stds = []
    
    for name in sorted(experiments.keys()):
        exp_names.append(name.replace('_', '\n'))
        is_means.append(experiments[name]['inception_score_mean'])
        is_stds.append(experiments[name]['inception_score_std'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(exp_names)))
    x = np.arange(len(exp_names))
    
    bars = ax.bar(x, is_means, yerr=is_stds, color=colors, alpha=0.8, 
                   edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inception Score (higher is better)', fontsize=12, fontweight='bold')
    ax.set_title('Inception Score Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, is_means, is_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_metrics(experiments, output_path):
    """Plot FID and IS together for comparison"""
    exp_names = []
    fid_scores = []
    is_means = []
    
    for name in sorted(experiments.keys()):
        exp_names.append(name.replace('_', '\n'))
        fid_scores.append(experiments[name]['fid_score'])
        is_means.append(experiments[name]['inception_score_mean'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # FID scores
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(exp_names)))
    bars1 = ax1.bar(range(len(exp_names)), fid_scores, color=colors1, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('FID Score Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels([])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, score in zip(bars1, fid_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Inception scores
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(exp_names)))
    bars2 = ax2.bar(range(len(exp_names)), is_means, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Inception Score (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_title('Inception Score Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(exp_names)))
    ax2.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, score in zip(bars2, is_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_curves(histories, output_path):
    """Plot training loss curves for all experiments"""
    if not histories:
        print("No training histories found")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for (name, history), color in zip(sorted(histories.items()), colors):
        rounds = history['rounds']
        g_loss = history['avg_g_loss']
        d_loss = history['avg_d_loss']
        
        display_name = name.replace('_', ' ').title()
        
        ax1.plot(rounds, g_loss, marker='o', label=display_name, color=color, linewidth=2, markersize=4)
        ax2.plot(rounds, d_loss, marker='s', label=display_name, color=color, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Generator Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Generator Loss over Training', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=8, loc='best')
    
    ax2.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Discriminator Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Discriminator Loss over Training', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_heterogeneity_study(experiments, output_path):
    """Plot how data heterogeneity affects performance"""
    heterogeneity_exps = {
        'IID': None,
        'α=1.0': None,
        'α=0.5': None,
        'α=0.1': None
    }
    
    for name, data in experiments.items():
        if 'iid' in name.lower() and 'noniid' not in name.lower():
            heterogeneity_exps['IID'] = data
        elif 'alpha1.0' in name or 'alpha_1.0' in name:
            heterogeneity_exps['α=1.0'] = data
        elif 'alpha0.5' in name or 'alpha_0.5' in name:
            heterogeneity_exps['α=0.5'] = data
        elif 'alpha0.1' in name or 'alpha_0.1' in name:
            heterogeneity_exps['α=0.1'] = data
    
    # Filter out None values
    heterogeneity_exps = {k: v for k, v in heterogeneity_exps.items() if v is not None}
    
    if len(heterogeneity_exps) < 2:
        print("Not enough heterogeneity experiments found")
        return
    
    labels = list(heterogeneity_exps.keys())
    fid_scores = [heterogeneity_exps[k]['fid_score'] for k in labels]
    is_scores = [heterogeneity_exps[k]['inception_score_mean'] for k in labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # FID vs Heterogeneity
    ax1.plot(labels, fid_scores, marker='o', markersize=10, linewidth=2.5, 
             color='steelblue', markerfacecolor='orange', markeredgewidth=2)
    ax1.set_xlabel('Data Heterogeneity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Data Heterogeneity on FID', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    for i, (x, y) in enumerate(zip(labels, fid_scores)):
        ax1.annotate(f'{y:.2f}', (i, y), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    # IS vs Heterogeneity
    ax2.plot(labels, is_scores, marker='s', markersize=10, linewidth=2.5,
             color='forestgreen', markerfacecolor='yellow', markeredgewidth=2)
    ax2.set_xlabel('Data Heterogeneity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Inception Score (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_title('Effect of Data Heterogeneity on IS', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    for i, (x, y) in enumerate(zip(labels, is_scores)):
        ax2.annotate(f'{y:.2f}', (i, y), textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_client_scaling(experiments, output_path):
    """Plot how number of clients affects performance"""
    client_exps = {}
    
    for name, data in experiments.items():
        if 'clients_3' in name or 'client_3' in name:
            client_exps['3 Clients'] = data
        elif 'clients_5' in name or 'client_5' in name or ('noniid_alpha0.5' in name and 'clients' not in name):
            client_exps['5 Clients'] = data
        elif 'clients_10' in name or 'client_10' in name:
            client_exps['10 Clients'] = data
    
    if len(client_exps) < 2:
        print("Not enough client scaling experiments found")
        return
    
    # Sort by number of clients
    sorted_exps = sorted(client_exps.items(), key=lambda x: int(x[0].split()[0]))
    labels = [k for k, v in sorted_exps]
    fid_scores = [v['fid_score'] for k, v in sorted_exps]
    is_scores = [v['inception_score_mean'] for k, v in sorted_exps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # FID vs Number of Clients
    ax1.plot(labels, fid_scores, marker='D', markersize=10, linewidth=2.5,
             color='purple', markerfacecolor='cyan', markeredgewidth=2)
    ax1.set_xlabel('Number of Clients', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FID Score (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Effect of Client Count on FID', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    for i, (x, y) in enumerate(zip(labels, fid_scores)):
        ax1.annotate(f'{y:.2f}', (i, y), textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    # IS vs Number of Clients
    ax2.plot(labels, is_scores, marker='D', markersize=10, linewidth=2.5,
             color='darkred', markerfacecolor='pink', markeredgewidth=2)
    ax2.set_xlabel('Number of Clients', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Inception Score (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_title('Effect of Client Count on IS', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    for i, (x, y) in enumerate(zip(labels, is_scores)):
        ax2.annotate(f'{y:.2f}', (i, y), textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_markdown_report(experiments, histories, output_path):
    """Generate a markdown report with all results"""
    with open(output_path, 'w') as f:
        f.write("# Federated Learning GAN Experiments - Results Report\n\n")
        f.write("## Experiment Summary\n\n")
        
        # Metrics table
        f.write("### Performance Metrics\n\n")
        f.write("| Experiment | FID Score ↓ | Inception Score ↑ | Samples |\n")
        f.write("|------------|-------------|-------------------|----------|\n")
        
        for name in sorted(experiments.keys()):
            data = experiments[name]
            fid = data['fid_score']
            is_mean = data['inception_score_mean']
            is_std = data['inception_score_std']
            samples = data['num_samples']
            
            f.write(f"| {name} | {fid:.4f} | {is_mean:.4f} ± {is_std:.4f} | {samples} |\n")
        
        f.write("\n")
        
        # Best performing models
        f.write("## Best Performing Models\n\n")
        
        best_fid_name = min(experiments.items(), key=lambda x: x[1]['fid_score'])[0]
        best_fid_score = experiments[best_fid_name]['fid_score']
        f.write(f"**Best FID Score:** {best_fid_name} ({best_fid_score:.4f})\n\n")
        
        best_is_name = max(experiments.items(), key=lambda x: x[1]['inception_score_mean'])[0]
        best_is_score = experiments[best_is_name]['inception_score_mean']
        f.write(f"**Best Inception Score:** {best_is_name} ({best_is_score:.4f})\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        f.write("### 1. Centralized vs Federated\n\n")
        
        if 'e1_centralized' in experiments:
            centralized_fid = experiments['e1_centralized']['fid_score']
            f.write(f"- Centralized baseline FID: {centralized_fid:.4f}\n")
            
            fed_exps = {k: v for k, v in experiments.items() if k != 'e1_centralized'}
            if fed_exps:
                avg_fed_fid = np.mean([v['fid_score'] for v in fed_exps.values()])
                gap = avg_fed_fid - centralized_fid
                gap_pct = (gap / centralized_fid) * 100
                f.write(f"- Average federated FID: {avg_fed_fid:.4f}\n")
                f.write(f"- Performance gap: {gap:.4f} ({gap_pct:.2f}%)\n\n")
        
        f.write("### 2. Data Heterogeneity Impact\n\n")
        f.write("Impact of non-IID data distribution on model performance.\n\n")
        
        f.write("### 3. Client Scalability\n\n")
        f.write("How the number of participating clients affects convergence and quality.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This report summarizes the results of federated learning experiments ")
        f.write("for medical imaging GANs. The experiments demonstrate the trade-offs ")
        f.write("between privacy-preserving distributed training and model performance.\n\n")
        
        f.write("---\n\n")
        f.write(f"*Generated automatically from experiment results*\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot comparison of federated learning experiments")
    parser.add_argument('--metrics_dir', type=str, default='results/metrics',
                        help='Directory containing metric JSON files')
    parser.add_argument('--results_dir', type=str, default='results/experiments',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("Loading experiment data...")
    print(f"{'='*80}\n")
    
    # Load data
    experiments = load_experiment_metrics(args.metrics_dir)
    histories = load_training_history(args.results_dir)
    
    if not experiments:
        print(f"No experiments found in {args.metrics_dir}")
        return
    
    print(f"Found {len(experiments)} experiments with metrics")
    print(f"Found {len(histories)} experiments with training history")
    print()
    
    # Generate plots
    print(f"{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}\n")
    
    plot_fid_comparison(experiments, output_dir / 'fid_comparison.png')
    plot_inception_comparison(experiments, output_dir / 'inception_comparison.png')
    plot_combined_metrics(experiments, output_dir / 'combined_metrics.png')
    
    if histories:
        plot_training_curves(histories, output_dir / 'training_curves_all.png')
    
    plot_heterogeneity_study(experiments, output_dir / 'heterogeneity_study.png')
    plot_client_scaling(experiments, output_dir / 'client_scaling.png')
    
    # Generate report
    generate_markdown_report(experiments, histories, output_dir / 'results_report.md')
    
    print(f"\n{'='*80}")
    print("All plots generated successfully!")
    print(f"{'='*80}\n")
    print(f"Plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - fid_comparison.png")
    print(f"  - inception_comparison.png")
    print(f"  - combined_metrics.png")
    print(f"  - training_curves_all.png")
    print(f"  - heterogeneity_study.png")
    print(f"  - client_scaling.png")
    print(f"  - results_report.md")
    print()


if __name__ == "__main__":
    main()

