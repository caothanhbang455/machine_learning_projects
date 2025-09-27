"""
Evaluation Utilities

This module contains functions for evaluating clustering performance
and generating comprehensive reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances
import warnings
warnings.filterwarnings('ignore')


def calculate_clustering_metrics(data, labels, verbose=True):
    """
    Calculate comprehensive clustering evaluation metrics.

    Parameters:
    -----------
    data : array-like
        Original data
    labels : array-like
        Cluster labels
    verbose : bool
        Whether to print metrics

    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    if verbose:
        print("📊 Calculating Clustering Evaluation Metrics")
        print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    metrics = {}

    # Internal evaluation metrics
    try:
        silhouette_avg = silhouette_score(data_dense, labels)
        metrics['silhouette_score'] = silhouette_avg
    except Exception as e:
        metrics['silhouette_score'] = None
        if verbose:
            print(f"⚠️  Could not calculate silhouette score: {e}")

    # Cluster statistics
    n_clusters = len(np.unique(labels))
    cluster_sizes = np.bincount(labels)

    metrics['n_clusters'] = n_clusters
    metrics['cluster_sizes'] = cluster_sizes
    metrics['min_cluster_size'] = np.min(cluster_sizes)
    metrics['max_cluster_size'] = np.max(cluster_sizes)
    metrics['mean_cluster_size'] = np.mean(cluster_sizes)
    metrics['std_cluster_size'] = np.std(cluster_sizes)

    # Cluster balance coefficient
    metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)

    # Intra-cluster cohesion
    intra_cluster_distances = []
    cluster_cohesions = []

    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]

        if len(cluster_data) > 1:
            # Calculate pairwise distances within cluster
            distances = pairwise_distances(cluster_data)
            # Get upper triangle (excluding diagonal)
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            intra_cluster_distances.extend(upper_tri)
            cluster_cohesions.append(np.mean(upper_tri))
        else:
            cluster_cohesions.append(0)

    metrics['mean_intra_cluster_distance'] = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
    metrics['std_intra_cluster_distance'] = np.std(intra_cluster_distances) if intra_cluster_distances else 0
    metrics['cluster_cohesions'] = cluster_cohesions

    # Inter-cluster separation
    cluster_centers = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]
        cluster_center = np.mean(cluster_data, axis=0)
        cluster_centers.append(cluster_center)

    cluster_centers = np.array(cluster_centers)

    if len(cluster_centers) > 1:
        inter_distances = pairwise_distances(cluster_centers)
        upper_tri = inter_distances[np.triu_indices_from(inter_distances, k=1)]
        metrics['mean_inter_cluster_distance'] = np.mean(upper_tri)
        metrics['std_inter_cluster_distance'] = np.std(upper_tri)
    else:
        metrics['mean_inter_cluster_distance'] = 0
        metrics['std_inter_cluster_distance'] = 0

    # Separation ratio (Davies-Bouldin like)
    if metrics['mean_intra_cluster_distance'] > 0:
        metrics['separation_ratio'] = (metrics['mean_inter_cluster_distance']
                                       / metrics['mean_intra_cluster_distance'])
    else:
        metrics['separation_ratio'] = float('inf')

    # Calinski-Harabasz Index (approximation)
    if n_clusters > 1:
        overall_center = np.mean(data_dense, axis=0)

        # Between-cluster sum of squares
        between_cluster_ss = 0
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = data_dense[cluster_mask]
            cluster_center = np.mean(cluster_data, axis=0)
            cluster_size = len(cluster_data)

            center_distance = np.linalg.norm(cluster_center - overall_center)
            between_cluster_ss += cluster_size * (center_distance ** 2)

        # Within-cluster sum of squares
        within_cluster_ss = 0
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = data_dense[cluster_mask]
            cluster_center = np.mean(cluster_data, axis=0)

            for point in cluster_data:
                within_cluster_ss += np.linalg.norm(point - cluster_center) ** 2

        if within_cluster_ss > 0:
            metrics['calinski_harabasz_index'] = (between_cluster_ss / (n_clusters - 1)) / \
                (within_cluster_ss / (len(data_dense) - n_clusters))
        else:
            metrics['calinski_harabasz_index'] = float('inf')
    else:
        metrics['calinski_harabasz_index'] = 0

    if verbose:
        print(f"📊 Clustering Evaluation Results:")
        print(f"   🎯 Silhouette Score: {metrics['silhouette_score']:.3f}" if metrics['silhouette_score']
              is not None else "   🎯 Silhouette Score: N/A")
        print(f"   📊 Number of Clusters: {metrics['n_clusters']}")
        print(f"   📊 Cluster Sizes: {metrics['cluster_sizes']}")
        print(f"   📊 Cluster Balance: {metrics['cluster_balance']:.3f}")
        print(f"   📊 Mean Intra-cluster Distance: {metrics['mean_intra_cluster_distance']:.3f}")
        print(f"   📊 Mean Inter-cluster Distance: {metrics['mean_inter_cluster_distance']:.3f}")
        print(f"   📊 Separation Ratio: {metrics['separation_ratio']:.3f}")
        print(f"   📊 Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.3f}")

    return metrics


def evaluate_cluster_quality(data, labels, method_name="Clustering", verbose=True):
    """
    Evaluate the quality of clustering results with detailed analysis.

    Parameters:
    -----------
    data : array-like
        Original data
    labels : array-like
        Cluster labels
    method_name : str
        Name of the clustering method
    verbose : bool
        Whether to print evaluation

    Returns:
    --------
    dict
        Dictionary containing quality assessment
    """
    if verbose:
        print(f"🔍 Evaluating {method_name} Quality")
        print("=" * 50)

    # Get comprehensive metrics
    metrics = calculate_clustering_metrics(data, labels, verbose=False)

    # Quality assessment
    quality_assessment = {}

    # Silhouette score interpretation
    silhouette_score = metrics['silhouette_score']
    if silhouette_score is not None:
        if silhouette_score > 0.7:
            silhouette_quality = "Excellent"
        elif silhouette_score > 0.5:
            silhouette_quality = "Good"
        elif silhouette_score > 0.25:
            silhouette_quality = "Fair"
        else:
            silhouette_quality = "Poor"

        quality_assessment['silhouette_quality'] = silhouette_quality
    else:
        quality_assessment['silhouette_quality'] = "Cannot evaluate"

    # Cluster balance assessment
    balance_score = metrics['cluster_balance']
    if balance_score < 0.5:
        balance_quality = "Well-balanced"
    elif balance_score < 1.0:
        balance_quality = "Moderately balanced"
    else:
        balance_quality = "Imbalanced"

    quality_assessment['balance_quality'] = balance_quality

    # Separation assessment
    separation_ratio = metrics['separation_ratio']
    if separation_ratio > 2.0:
        separation_quality = "Well-separated"
    elif separation_ratio > 1.0:
        separation_quality = "Moderately separated"
    else:
        separation_quality = "Poorly separated"

    quality_assessment['separation_quality'] = separation_quality

    # Overall quality score (0-100)
    overall_score = 0

    if silhouette_score is not None:
        overall_score += min(silhouette_score * 50, 50)  # Max 50 points

    balance_component = max(0, 25 - balance_score * 12.5)  # Max 25 points
    overall_score += balance_component

    separation_component = min(separation_ratio * 12.5, 25)  # Max 25 points
    overall_score += separation_component

    quality_assessment['overall_score'] = overall_score

    # Overall quality grade
    if overall_score >= 80:
        overall_grade = "A (Excellent)"
    elif overall_score >= 70:
        overall_grade = "B (Good)"
    elif overall_score >= 60:
        overall_grade = "C (Fair)"
    elif overall_score >= 50:
        overall_grade = "D (Poor)"
    else:
        overall_grade = "F (Very Poor)"

    quality_assessment['overall_grade'] = overall_grade

    if verbose:
        print(f"📊 {method_name} Quality Assessment:")
        print(f"   🎯 Silhouette Quality: {quality_assessment['silhouette_quality']}")
        print(f"   ⚖️  Balance Quality: {quality_assessment['balance_quality']}")
        print(f"   📏 Separation Quality: {quality_assessment['separation_quality']}")
        print(f"   🎖️  Overall Score: {quality_assessment['overall_score']:.1f}/100")
        print(f"   🏆 Overall Grade: {quality_assessment['overall_grade']}")

    return quality_assessment


def create_evaluation_report(results_dict, data, verbose=True):
    """
    Create comprehensive evaluation report for multiple clustering results.

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results from different methods/k values
    data : array-like
        Original data
    verbose : bool
        Whether to print report

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing evaluation report
    """
    if verbose:
        print("📋 Creating Comprehensive Evaluation Report")
        print("=" * 50)

    report_data = []

    for method_name, method_results in results_dict.items():
        if isinstance(method_results, dict):
            # Multiple k values
            for k, result in method_results.items():
                if 'labels' in result:
                    metrics = calculate_clustering_metrics(
                        data, result['labels'], verbose=False
                    )
                    quality = evaluate_cluster_quality(
                        data, result['labels'], f"{method_name}_k{k}", verbose=False
                    )

                    report_row = {
                        'Method': method_name,
                        'K': k,
                        'Silhouette_Score': metrics['silhouette_score'],
                        'N_Clusters': metrics['n_clusters'],
                        'Cluster_Balance': metrics['cluster_balance'],
                        'Separation_Ratio': metrics['separation_ratio'],
                        'Calinski_Harabasz': metrics['calinski_harabasz_index'],
                        'Overall_Score': quality['overall_score'],
                        'Overall_Grade': quality['overall_grade']
                    }

                    if 'inertia' in result:
                        report_row['Inertia'] = result['inertia']

                    report_data.append(report_row)
        else:
            # Single result
            if 'labels' in method_results:
                metrics = calculate_clustering_metrics(
                    data, method_results['labels'], verbose=False
                )
                quality = evaluate_cluster_quality(
                    data, method_results['labels'], method_name, verbose=False
                )

                report_row = {
                    'Method': method_name,
                    'K': len(np.unique(method_results['labels'])),
                    'Silhouette_Score': metrics['silhouette_score'],
                    'N_Clusters': metrics['n_clusters'],
                    'Cluster_Balance': metrics['cluster_balance'],
                    'Separation_Ratio': metrics['separation_ratio'],
                    'Calinski_Harabasz': metrics['calinski_harabasz_index'],
                    'Overall_Score': quality['overall_score'],
                    'Overall_Grade': quality['overall_grade']
                }

                if 'inertia' in method_results:
                    report_row['Inertia'] = method_results['inertia']

                report_data.append(report_row)

    # Create DataFrame
    report_df = pd.DataFrame(report_data)

    if len(report_df) > 0:
        # Sort by overall score (descending)
        report_df = report_df.sort_values('Overall_Score', ascending=False)

        if verbose:
            print("\n📊 Evaluation Report Summary:")
            print(report_df.to_string(index=False, float_format='%.3f'))

            # Highlight best results
            best_result = report_df.iloc[0]
            print(f"\n🏆 Best Clustering Result:")
            print(f"   Method: {best_result['Method']}")
            print(f"   K: {best_result['K']}")
            print(f"   Overall Score: {best_result['Overall_Score']:.1f}/100")
            print(f"   Grade: {best_result['Overall_Grade']}")

    return report_df


def plot_evaluation_metrics(results_dict, data, figsize=(16, 12)):
    """
    Create comprehensive plots for evaluation metrics.

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results
    data : array-like
        Original data
    figsize : tuple
        Figure size
    """
    print("📊 Creating Evaluation Metrics Plots")
    print("=" * 50)

    # Create evaluation report
    report_df = create_evaluation_report(results_dict, data, verbose=False)

    if len(report_df) == 0:
        print("⚠️  No valid results to plot")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('📊 Clustering Evaluation Metrics Dashboard', fontsize=16, fontweight='bold')

    # 1. Silhouette Score vs K
    ax1 = axes[0, 0]
    methods = report_df['Method'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        method_data = report_df[report_df['Method'] == method]
        # Sort by K values to ensure proper line plotting
        method_data = method_data.sort_values('K')
        ax1.plot(method_data['K'], method_data['Silhouette_Score'],
                 'o-', label=method, color=colors[i], linewidth=2, markersize=8)

    ax1.set_title('🎯 Silhouette Score vs Number of Clusters')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Silhouette Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Overall Score vs K
    ax2 = axes[0, 1]
    for i, method in enumerate(methods):
        method_data = report_df[report_df['Method'] == method]
        # Sort by K values to ensure proper line plotting
        method_data = method_data.sort_values('K')
        ax2.plot(method_data['K'], method_data['Overall_Score'],
                 'o-', label=method, color=colors[i], linewidth=2, markersize=8)

    ax2.set_title('🏆 Overall Quality Score vs Number of Clusters')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Overall Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cluster Balance vs K
    ax3 = axes[0, 2]
    for i, method in enumerate(methods):
        method_data = report_df[report_df['Method'] == method]
        # Sort by K values to ensure proper line plotting
        method_data = method_data.sort_values('K')
        ax3.plot(method_data['K'], method_data['Cluster_Balance'],
                 'o-', label=method, color=colors[i], linewidth=2, markersize=8)

    ax3.set_title('⚖️  Cluster Balance vs Number of Clusters')
    ax3.set_xlabel('Number of Clusters (K)')
    ax3.set_ylabel('Cluster Balance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Separation Ratio vs K
    ax4 = axes[1, 0]
    for i, method in enumerate(methods):
        method_data = report_df[report_df['Method'] == method]
        # Sort by K values to ensure proper line plotting
        method_data = method_data.sort_values('K')
        ax4.plot(method_data['K'], method_data['Separation_Ratio'],
                 'o-', label=method, color=colors[i], linewidth=2, markersize=8)

    ax4.set_title('📏 Separation Ratio vs Number of Clusters')
    ax4.set_xlabel('Number of Clusters (K)')
    ax4.set_ylabel('Separation Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Method Comparison (Best K for each method)
    ax5 = axes[1, 1]
    best_results = report_df.groupby('Method')['Overall_Score'].max()

    bars = ax5.bar(best_results.index, best_results.values,
                   color=colors[:len(best_results)], alpha=0.7)
    ax5.set_title('🏆 Best Overall Score by Method')
    ax5.set_xlabel('Method')
    ax5.set_ylabel('Best Overall Score')
    ax5.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, best_results.values):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # 6. Inertia vs K (if available)
    ax6 = axes[1, 2]
    if 'Inertia' in report_df.columns:
        for i, method in enumerate(methods):
            method_data = report_df[report_df['Method'] == method]
            if not method_data['Inertia'].isna().all():
                # Sort by K values to ensure proper line plotting
                method_data = method_data.sort_values('K')
                ax6.plot(method_data['K'], method_data['Inertia'],
                         'o-', label=method, color=colors[i], linewidth=2, markersize=8)

        ax6.set_title('📊 Inertia vs Number of Clusters')
        ax6.set_xlabel('Number of Clusters (K)')
        ax6.set_ylabel('Inertia')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Inertia data\nnot available',
                 ha='center', va='center', transform=ax6.transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax6.set_title('📊 Inertia vs Number of Clusters')

    plt.tight_layout()
    plt.show()

    print("✅ Evaluation metrics plots completed!")


def compare_clustering_results(results_dict, data, metric='silhouette_score',
                               figsize=(12, 8), verbose=True):
    """
    Compare clustering results across different methods and parameters.

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results
    data : array-like
        Original data
    metric : str
        Metric to use for comparison
    figsize : tuple
        Figure size
    verbose : bool
        Whether to print comparison

    Returns:
    --------
    pandas.DataFrame
        DataFrame with comparison results
    """
    if verbose:
        print(f"🔍 Comparing Clustering Results by {metric}")
        print("=" * 50)

    # Create evaluation report
    report_df = create_evaluation_report(results_dict, data, verbose=False)

    if len(report_df) == 0:
        print("⚠️  No valid results to compare")
        return pd.DataFrame()

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Metric comparison by method
    if metric in report_df.columns:
        methods = report_df['Method'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

        for i, method in enumerate(methods):
            method_data = report_df[report_df['Method'] == method]
            # Sort by K values to ensure proper line plotting
            method_data = method_data.sort_values('K')
            ax1.plot(method_data['K'], method_data[metric],
                     'o-', label=method, color=colors[i], linewidth=2, markersize=8)

        ax1.set_title(f'📊 {metric} Comparison')
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel(metric)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Best results by method
    best_results = report_df.loc[report_df.groupby('Method')[metric].idxmax()]

    bars = ax2.bar(best_results['Method'], best_results[metric],
                   color=colors[:len(best_results)], alpha=0.7)
    ax2.set_title(f'🏆 Best {metric} by Method')
    ax2.set_xlabel('Method')
    ax2.set_ylabel(f'Best {metric}')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels and K values
    for bar, value, k in zip(bars, best_results[metric], best_results['K']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}\n(K={k})', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"\n📊 Best {metric} Results:")
        print(best_results[['Method', 'K', metric]].to_string(index=False))

    return report_df
