"""
Visualization Utilities

This module contains functions for creating beautiful visualizations
for clustering analysis in the text preprocessing demo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def plot_beautiful_clusters(data, labels, method='PCA', figsize=(16, 12),
                            n_components=3, title=None):
    """
    Create beautiful cluster visualizations with dimensionality reduction.

    Parameters:
    -----------
    data : array-like
        Data to visualize
    labels : array-like
        Cluster labels
    method : str
        Dimensionality reduction method
    figsize : tuple
        Figure size
    n_components : int
        Number of components for visualization
    title : str
        Plot title
    """
    if title is None:
        title = f"✨ Beautiful Cluster Visualization ({method})"

    print(f"🎨 Creating {title}")
    print("=" * 50)

    # Apply dimensionality reduction
    if method == 'PCA':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=min(n_components, 3))
    elif method == 'SVD':
        reducer = TruncatedSVD(n_components=min(n_components, 3))
    else:
        raise ValueError("Method must be 'PCA' or 'SVD'")

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    reduced_data = reducer.fit_transform(data_dense)

    # Create color palette
    n_clusters = len(np.unique(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    if reduced_data.shape[1] >= 3:
        # 3D visualization
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        for i, cluster_id in enumerate(np.unique(labels)):
            cluster_mask = labels == cluster_id
            ax.scatter(reduced_data[cluster_mask, 0],
                       reduced_data[cluster_mask, 1],
                       reduced_data[cluster_mask, 2],
                       c=[colors[i]], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=50)

        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')
        ax.set_zlabel(f'{method} Component 3')
        ax.set_title(title)
        ax.legend()

    else:
        # 2D visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, cluster_id in enumerate(np.unique(labels)):
            cluster_mask = labels == cluster_id
            ax.scatter(reduced_data[cluster_mask, 0],
                       reduced_data[cluster_mask, 1],
                       c=[colors[i]], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=50)

        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✅ Beautiful cluster visualization completed!")


def create_cluster_dashboard(data, labels, cluster_names=None, figsize=(20, 16)):
    """
    Create a comprehensive clustering dashboard.

    Parameters:
    -----------
    data : array-like
        Original data
    labels : array-like
        Cluster labels
    cluster_names : list, optional
        Names for clusters
    figsize : tuple
        Figure size
    """
    print("📊 Creating Comprehensive Clustering Dashboard")
    print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Color palette
    n_clusters = len(np.unique(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    if cluster_names is None:
        cluster_names = [f'Cluster {i}' for i in range(n_clusters)]

    # 1. Cluster size distribution
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    bars = ax1.bar(range(len(cluster_counts)), cluster_counts.values,
                   color=colors[:len(cluster_counts)], alpha=0.7)
    ax1.set_title('📊 Cluster Size Distribution')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Documents')
    ax1.set_xticks(range(len(cluster_counts)))
    ax1.set_xticklabels([f'C{i}' for i in cluster_counts.index])

    # Add value labels on bars
    for bar, value in zip(bars, cluster_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value}', ha='center', va='bottom')

    # 2. 2D PCA visualization
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_dense)

    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_mask = labels == cluster_id
        ax2.scatter(pca_result[cluster_mask, 0], pca_result[cluster_mask, 1],
                    c=[colors[i]], label=cluster_names[i], alpha=0.6, s=30)

    ax2.set_title('📊 PCA Visualization')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. Silhouette score
    ax3 = fig.add_subplot(gs[0, 2])
    silhouette_avg = silhouette_score(data_dense, labels)
    sample_silhouette_values = silhouette_samples(data_dense, labels)

    y_lower = 10
    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_silhouette_values = sample_silhouette_values[labels == cluster_id]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax3.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_values,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

        ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
        y_lower = y_upper + 10

    ax3.set_title('📊 Silhouette Analysis')
    ax3.set_xlabel('Silhouette Coefficient')
    ax3.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f'Average Score: {silhouette_avg:.3f}')
    ax3.legend()

    # 4. Cluster statistics
    ax4 = fig.add_subplot(gs[0, 3])
    cluster_stats = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]

        stats = {
            'size': np.sum(cluster_mask),
            'mean_feature_activity': np.mean(np.sum(cluster_data > 0, axis=1)),
            'mean_magnitude': np.mean(np.linalg.norm(cluster_data, axis=1))
        }
        cluster_stats.append(stats)

    stats_df = pd.DataFrame(cluster_stats)
    stats_df.index = [f'C{i}' for i in range(len(stats_df))]

    im = ax4.imshow(stats_df.values.T, cmap='YlOrRd', aspect='auto')
    ax4.set_title('📊 Cluster Statistics Heatmap')
    ax4.set_xticks(range(len(stats_df)))
    ax4.set_xticklabels(stats_df.index)
    ax4.set_yticks(range(len(stats_df.columns)))
    ax4.set_yticklabels(stats_df.columns)

    # Add text annotations
    for i in range(len(stats_df.columns)):
        for j in range(len(stats_df)):
            text = ax4.text(j, i, f'{stats_df.iloc[j, i]:.1f}',
                            ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax4)

    # 5. Document distribution by cluster (if we have more info)
    ax5 = fig.add_subplot(gs[1, :2])

    # Create a stacked bar chart showing cluster composition
    cluster_composition = pd.Series(labels).value_counts().sort_index()

    wedges, texts, autotexts = ax5.pie(cluster_composition.values,
                                       labels=[f'Cluster {i}' for i in cluster_composition.index],
                                       colors=colors[:len(cluster_composition)],
                                       autopct='%1.1f%%', startangle=90)
    ax5.set_title('📊 Cluster Distribution')

    # 6. Feature activity heatmap
    ax6 = fig.add_subplot(gs[1, 2:])

    # Show top features for each cluster
    top_features_per_cluster = {}
    n_top_features = 10

    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]

        # Calculate mean feature values for this cluster
        mean_features = np.mean(cluster_data, axis=0)
        top_feature_indices = np.argsort(mean_features)[-n_top_features:]
        top_features_per_cluster[cluster_id] = top_feature_indices

    # Create heatmap data
    all_top_features = set()
    for features in top_features_per_cluster.values():
        all_top_features.update(features)

    heatmap_data = np.zeros((len(all_top_features), len(np.unique(labels))))
    feature_list = list(all_top_features)

    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]

        for j, feature_idx in enumerate(feature_list):
            heatmap_data[j, i] = np.mean(cluster_data[:, feature_idx])

    im = ax6.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax6.set_title('📊 Top Features by Cluster')
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Feature Index')
    ax6.set_xticks(range(len(np.unique(labels))))
    ax6.set_xticklabels([f'C{i}' for i in np.unique(labels)])

    plt.colorbar(im, ax=ax6)

    # 7. Cluster separation visualization
    ax7 = fig.add_subplot(gs[2, :])

    # Use t-SNE for better separation visualization
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(data_dense)

        for i, cluster_id in enumerate(np.unique(labels)):
            cluster_mask = labels == cluster_id
            ax7.scatter(tsne_result[cluster_mask, 0], tsne_result[cluster_mask, 1],
                        c=[colors[i]], label=cluster_names[i], alpha=0.6, s=30)

        ax7.set_title('📊 t-SNE Cluster Separation')
        ax7.set_xlabel('t-SNE Component 1')
        ax7.set_ylabel('t-SNE Component 2')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)

    except ImportError:
        # Fall back to PCA if t-SNE not available
        for i, cluster_id in enumerate(np.unique(labels)):
            cluster_mask = labels == cluster_id
            ax7.scatter(pca_result[cluster_mask, 0], pca_result[cluster_mask, 1],
                        c=[colors[i]], label=cluster_names[i], alpha=0.6, s=30)

        ax7.set_title('📊 PCA Cluster Separation (Fallback)')
        ax7.set_xlabel('PC1')
        ax7.set_ylabel('PC2')
        ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✅ Comprehensive clustering dashboard completed!")


def plot_silhouette_analysis(data, labels, figsize=(12, 8)):
    """
    Create detailed silhouette analysis plot.

    Parameters:
    -----------
    data : array-like
        Data used for clustering
    labels : array-like
        Cluster labels
    figsize : tuple
        Figure size
    """
    print("📊 Creating Detailed Silhouette Analysis")
    print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    n_clusters = len(np.unique(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # First subplot: Silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data_dense) + (n_clusters + 1) * 10])

    # Calculate silhouette scores
    silhouette_avg = silhouette_score(data_dense, labels)
    sample_silhouette_values = silhouette_samples(data_dense, labels)

    print(f"📊 Average Silhouette Score: {silhouette_avg:.3f}")

    y_lower = 10
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_silhouette_values = sample_silhouette_values[labels == cluster_id]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_values,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
        y_lower = y_upper + 10

    ax1.set_title('📊 Silhouette Plot')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_ylabel('Cluster Label')

    # Add vertical line for average silhouette score
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f'Average Score: {silhouette_avg:.3f}')
    ax1.legend()

    # Second subplot: Cluster visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_dense)

    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_mask = labels == cluster_id
        ax2.scatter(pca_result[cluster_mask, 0], pca_result[cluster_mask, 1],
                    c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=30)

    ax2.set_title('📊 Cluster Visualization')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print cluster-wise silhouette scores
    print("\n📊 Cluster-wise Silhouette Scores:")
    for cluster_id in np.unique(labels):
        cluster_silhouette_values = sample_silhouette_values[labels == cluster_id]
        print(f"   Cluster {cluster_id}: {cluster_silhouette_values.mean():.3f} "
              f"(±{cluster_silhouette_values.std():.3f})")

    print("✅ Detailed silhouette analysis completed!")


def plot_silhouette_analysis_multiple_k(data, clustering_results, method_name='Clustering', figsize=(20, 12)):
    """
    Create beautiful silhouette analysis plots for multiple k values.

    Parameters:
    -----------
    data : array-like
        Data used for clustering
    clustering_results : dict
        Dictionary containing clustering results for different k values
    method_name : str
        Name of the clustering method for title
    figsize : tuple
        Figure size
    """
    print(f"📊 Creating Multiple K Silhouette Analysis for {method_name}")
    print("=" * 60)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    # Get k values and sort them
    k_values = sorted(clustering_results.keys())
    n_k = len(k_values)
    
    # Calculate grid dimensions
    n_cols = min(3, n_k)  # Maximum 3 columns
    n_rows = (n_k + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Track best k (highest silhouette score)
    best_k = None
    best_score = -1
    
    # Store all scores for comparison plot
    all_scores = []

    for idx, k in enumerate(k_values):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get clustering results for this k
        result = clustering_results[k]
        labels = result['labels']
        silhouette_avg = result['silhouette_score']
        
        # Track best k
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k
        
        all_scores.append(silhouette_avg)

        # Calculate silhouette samples
        sample_silhouette_values = silhouette_samples(data_dense, labels)

        # Set up silhouette plot
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(data_dense) + (k + 1) * 10])

        y_lower = 10
        colors = plt.cm.Set3(np.linspace(0, 1, k))

        for i, cluster_id in enumerate(np.unique(labels)):
            cluster_silhouette_values = sample_silhouette_values[labels == cluster_id]
            cluster_silhouette_values.sort()

            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

            # Add cluster label
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id), 
                    fontsize=8, ha='center')
            y_lower = y_upper + 10

        # Add vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
        
        # Styling
        ax.set_title(f'K={k}\nSilhouette Score: {silhouette_avg:.3f}', 
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Silhouette Coefficient', fontsize=9)
        if col == 0:
            ax.set_ylabel('Cluster Label', fontsize=9)
        
        # Highlight best k
        if k == best_k:
            ax.set_facecolor('#fff2cc')  # Light yellow background
            for spine in ax.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(3)

    # Hide unused subplots
    for idx in range(n_k, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'🔍 {method_name} Silhouette Analysis - All K Values\n'
                 f'🏆 Best K: {best_k} (Score: {best_score:.3f})', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # Create summary comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Silhouette scores trend
    ax1.plot(k_values, all_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.axhline(y=max(all_scores), color='red', linestyle='--', alpha=0.7, 
                label=f'Best Score: {max(all_scores):.3f}')
    ax1.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Best K: {best_k}')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Average Silhouette Score')
    ax1.set_title(f'📊 {method_name} Silhouette Score Trend')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Annotate each point
    for i, (k, score) in enumerate(zip(k_values, all_scores)):
        ax1.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Cluster size distribution for best k
    best_labels = clustering_results[best_k]['labels']
    cluster_sizes = np.bincount(best_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
    
    bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes, color=colors, alpha=0.8)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Documents')
    ax2.set_title(f'📊 Cluster Sizes for Best K={best_k}')
    ax2.set_xticks(range(len(cluster_sizes)))
    
    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{size}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print detailed analysis
    print(f"\n📊 Detailed Analysis for {method_name}:")
    print("=" * 50)
    print(f"🏆 Best K: {best_k} (Silhouette Score: {best_score:.3f})")
    print(f"📊 K values tested: {k_values}")
    print(f"📈 Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
    
    print(f"\n📊 All K Results:")
    for k, score in zip(k_values, all_scores):
        status = "🏆 BEST" if k == best_k else "   "
        print(f"   K={k}: {score:.3f} {status}")
    
    print(f"\n💡 Interpretation Guide:")
    print(f"   • Higher silhouette score = better separated clusters")
    print(f"   • Score > 0.5: Strong cluster structure")
    print(f"   • Score 0.25-0.5: Moderate cluster structure")
    print(f"   • Score < 0.25: Weak cluster structure")
    
    print(f"\n✅ Multiple K silhouette analysis completed!")
    print(f"🎯 Recommendation: Consider K={best_k} for best separation")
    
    return best_k, best_score


def plot_cluster_comparison(results_dict, metrics=['silhouette', 'inertia'],
                            figsize=(15, 10)):
    """
    Compare different clustering results.

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results
    metrics : list
        List of metrics to compare
    figsize : tuple
        Figure size
    """
    print("📊 Creating Cluster Comparison Plot")
    print("=" * 50)

    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric == 'silhouette':
            # Plot silhouette scores
            methods = list(results_dict.keys())
            k_values = []
            scores = []
            method_labels = []

            for method, results in results_dict.items():
                for k, result in results.items():
                    if 'silhouette_score' in result:
                        k_values.append(k)
                        scores.append(result['silhouette_score'])
                        method_labels.append(method)

            # Group by method
            method_data = {}
            for method, k, score in zip(method_labels, k_values, scores):
                if method not in method_data:
                    method_data[method] = {'k': [], 'scores': []}
                method_data[method]['k'].append(k)
                method_data[method]['scores'].append(score)

            # Plot each method
            colors = plt.cm.Set1(np.linspace(0, 1, len(method_data)))
            for j, (method, data) in enumerate(method_data.items()):
                # Sort data by k values to ensure proper line plotting
                sorted_indices = np.argsort(data['k'])
                sorted_k = np.array(data['k'])[sorted_indices]
                sorted_scores = np.array(data['scores'])[sorted_indices]
                
                ax.plot(sorted_k, sorted_scores, 'o-', label=method,
                        color=colors[j], linewidth=2, markersize=8)

            ax.set_title('📊 Silhouette Score Comparison')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.legend()
            ax.grid(True, alpha=0.3)

        elif metric == 'inertia':
            # Plot inertia/distortion
            methods = list(results_dict.keys())
            k_values = []
            inertias = []
            method_labels = []

            for method, results in results_dict.items():
                for k, result in results.items():
                    if 'inertia' in result:
                        k_values.append(k)
                        inertias.append(result['inertia'])
                        method_labels.append(method)

            # Group by method
            method_data = {}
            for method, k, inertia in zip(method_labels, k_values, inertias):
                if method not in method_data:
                    method_data[method] = {'k': [], 'inertias': []}
                method_data[method]['k'].append(k)
                method_data[method]['inertias'].append(inertia)

            # Plot each method
            colors = plt.cm.Set1(np.linspace(0, 1, len(method_data)))
            for j, (method, data) in enumerate(method_data.items()):
                # Sort data by k values to ensure proper line plotting
                sorted_indices = np.argsort(data['k'])
                sorted_k = np.array(data['k'])[sorted_indices]
                sorted_inertias = np.array(data['inertias'])[sorted_indices]
                
                ax.plot(sorted_k, sorted_inertias, 'o-', label=method,
                        color=colors[j], linewidth=2, markersize=8)

            ax.set_title('📊 Inertia/Distortion Comparison')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✅ Cluster comparison plot completed!")


def plot_dimensionality_reduction(data, method='PCA', n_components=2, figsize=(12, 8)):
    """
    Plot dimensionality reduction results.

    Parameters:
    -----------
    data : array-like
        Data to reduce
    method : str
        Dimensionality reduction method
    n_components : int
        Number of components
    figsize : tuple
        Figure size
    """
    print(f"📊 Creating {method} Dimensionality Reduction Plot")
    print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    if method == 'PCA':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(data_dense)
        explained_variance = reducer.explained_variance_ratio_

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Scatter plot
        ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
        ax1.set_title('📊 PCA Visualization')
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        ax1.grid(True, alpha=0.3)

        # Explained variance plot
        ax2.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax2.set_title('📊 Explained Variance')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')

    elif method == 'SVD':
        reducer = TruncatedSVD(n_components=n_components)
        reduced_data = reducer.fit_transform(data_dense)
        explained_variance = reducer.explained_variance_ratio_

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Scatter plot
        ax1.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
        ax1.set_title('📊 SVD Visualization')
        ax1.set_xlabel(f'Component 1 ({explained_variance[0]:.2%} variance)')
        ax1.set_ylabel(f'Component 2 ({explained_variance[1]:.2%} variance)')
        ax1.grid(True, alpha=0.3)

        # Explained variance plot
        ax2.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax2.set_title('📊 Explained Variance')
        ax2.set_xlabel('SVD Component')
        ax2.set_ylabel('Explained Variance Ratio')

    plt.tight_layout()
    plt.show()

    print(f"✅ {method} dimensionality reduction plot completed!")


def create_interactive_cluster_plot(data, labels, method='PCA',
                                    save_path=None, figsize=(12, 8)):
    """
    Create an interactive cluster plot (if plotly is available).

    Parameters:
    -----------
    data : array-like
        Data to visualize
    labels : array-like
        Cluster labels
    method : str
        Dimensionality reduction method
    save_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        print("📊 Creating Interactive Cluster Plot")
        print("=" * 50)

        # Convert sparse matrix to dense if needed
        if hasattr(data, 'toarray'):
            data_dense = data.toarray()
        else:
            data_dense = data

        # Apply dimensionality reduction
        if method == 'PCA':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(data_dense)
            explained_variance = reducer.explained_variance_ratio_

            df_plot = pd.DataFrame({
                'PC1': reduced_data[:, 0],
                'PC2': reduced_data[:, 1],
                'Cluster': labels
            })

            fig = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster',
                             title='📊 Interactive Cluster Visualization',
                             labels={'PC1': f'PC1 ({explained_variance[0]:.2%} variance)',
                                     'PC2': f'PC2 ({explained_variance[1]:.2%} variance)'})

        elif method == 'SVD':
            reducer = TruncatedSVD(n_components=2)
            reduced_data = reducer.fit_transform(data_dense)
            explained_variance = reducer.explained_variance_ratio_

            df_plot = pd.DataFrame({
                'Component1': reduced_data[:, 0],
                'Component2': reduced_data[:, 1],
                'Cluster': labels
            })

            fig = px.scatter(df_plot, x='Component1', y='Component2', color='Cluster',
                             title='📊 Interactive Cluster Visualization',
                             labels={'Component1': f'Component 1 ({explained_variance[0]:.2%} variance)',
                                     'Component2': f'Component 2 ({explained_variance[1]:.2%} variance)'})

        fig.update_layout(
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            title_font_size=16
        )

        if save_path:
            fig.write_html(save_path)
            print(f"📁 Interactive plot saved to {save_path}")

        fig.show()

        print("✅ Interactive cluster plot created!")

    except ImportError:
        print("⚠️  Plotly not available. Creating static plot instead...")
        plot_beautiful_clusters(data, labels, method=method, figsize=figsize,
                                title="📊 Static Cluster Visualization")
