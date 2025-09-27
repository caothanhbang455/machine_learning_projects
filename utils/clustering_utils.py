"""
Clustering Utilities

This module contains functions for clustering analysis including
KMeans, Gaussian Mixture Models, and cluster evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import multiprocessing as mp
from itertools import repeat
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calculate_gmm_wsse(data, gmm_model):
    """
    Calculate Within-cluster Sum of Squared Errors (WSSE) for GMM.
    
    For GMM, WSSE is calculated as the sum of squared distances from each point
    to its assigned cluster center (mean of the Gaussian component).
    
    Parameters:
    -----------
    data : array-like
        Data points
    gmm_model : GaussianMixture
        Fitted GMM model
        
    Returns:
    --------
    float
        WSSE value (always positive)
    """
    # Get cluster assignments
    labels = gmm_model.predict(data)
    
    # Get cluster centers (means)
    centers = gmm_model.means_
    
    # Calculate WSSE
    wsse = 0.0
    for i in range(len(data)):
        cluster_id = labels[i]
        center = centers[cluster_id]
        # Sum of squared distances to assigned cluster center
        wsse += np.sum((data[i] - center) ** 2)
    
    return wsse


def perform_clustering_analysis(data, method='kmeans', k_range=range(2, 10),
                                n_runs=5, random_state=42, verbose=True):
    """
    Perform comprehensive clustering analysis.

    Parameters:
    -----------
    data : array-like
        Data to cluster
    method : str
        Clustering method ('kmeans' or 'gmm')
    k_range : range
        Range of k values to test
    n_runs : int
        Number of runs for each k
    random_state : int
        Random state for reproducibility
    verbose : bool
        Whether to print progress

    Returns:
    --------
    dict
        Dictionary containing clustering results
    """
    if verbose:
        print(f"🔍 Performing {method.upper()} Clustering Analysis")
        print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    results = {}

    for k in k_range:
        if verbose:
            print(f"📊 Testing k={k}...")

        k_results = []

        for run in range(n_runs):
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=random_state + run,
                               n_init=20, max_iter=300)
            elif method == 'gmm':
                model = GaussianMixture(n_components=k, random_state=random_state + run,
                                        max_iter=300, n_init=3)
            else:
                raise ValueError("Method must be 'kmeans' or 'gmm'")

            # Fit the model
            model.fit(data_dense)

            # Get predictions
            if method == 'kmeans':
                labels = model.labels_
                inertia = model.inertia_
            else:
                labels = model.predict(data_dense)
                # Calculate proper WSSE for GMM instead of negative log-likelihood
                inertia = calculate_gmm_wsse(data_dense, model)

            # Calculate metrics
            silhouette_avg = silhouette_score(data_dense, labels)

            k_results.append({
                'model': model,
                'labels': labels,
                'inertia': inertia,
                'silhouette_score': silhouette_avg,
                'n_clusters': k,
                'run': run
            })

        # Select best run based on silhouette score
        best_run = max(k_results, key=lambda x: x['silhouette_score'])
        results[k] = best_run

        if verbose:
            print(f"   Best silhouette score: {best_run['silhouette_score']:.3f}")

    if verbose:
        print(f"\n✅ {method.upper()} clustering analysis completed!")
        print(f"   📊 Tested k values: {list(k_range)}")
        print(f"   🔄 Runs per k: {n_runs}")

    return results


def find_optimal_clusters(data, methods=['kmeans', 'gmm'], k_range=range(2, 10),
                          criteria=['silhouette', 'elbow'], verbose=True):
    """
    Find optimal number of clusters using multiple methods and criteria.

    Parameters:
    -----------
    data : array-like
        Data to cluster
    methods : list
        List of clustering methods to try
    k_range : range
        Range of k values to test
    criteria : list
        List of criteria to use for optimization
    verbose : bool
        Whether to print progress

    Returns:
    --------
    dict
        Dictionary containing optimal k for each method/criteria combination
    """
    if verbose:
        print("🎯 Finding Optimal Number of Clusters")
        print("=" * 50)

    results = {}

    for method in methods:
        if verbose:
            print(f"\n📊 Analyzing {method.upper()} method...")

        method_results = perform_clustering_analysis(
            data, method=method, k_range=k_range, verbose=False
        )

        results[method] = method_results

        # Find optimal k for each criterion
        for criterion in criteria:
            if criterion == 'silhouette':
                optimal_k = max(method_results.keys(),
                                key=lambda k: method_results[k]['silhouette_score'])
                optimal_score = method_results[optimal_k]['silhouette_score']

                if verbose:
                    print(f"   🎯 Optimal k (silhouette): {optimal_k} "
                          f"(score: {optimal_score:.3f})")

            elif criterion == 'elbow':
                # Calculate elbow using inertia
                inertias = [method_results[k]['inertia'] for k in k_range]

                # Find elbow point using second derivative
                if len(inertias) >= 3:
                    second_derivatives = []
                    for i in range(1, len(inertias) - 1):
                        second_deriv = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
                        second_derivatives.append(second_deriv)

                    # Find point of maximum curvature
                    elbow_idx = np.argmax(second_derivatives) + 1
                    optimal_k = list(k_range)[elbow_idx]

                    if verbose:
                        print(f"   🎯 Optimal k (elbow): {optimal_k}")

    return results


def compare_clustering_methods(data, methods=['kmeans', 'gmm'], k_values=[3, 5, 7],
                               verbose=True):
    """
    Compare different clustering methods.

    Parameters:
    -----------
    data : array-like
        Data to cluster
    methods : list
        List of clustering methods to compare
    k_values : list
        List of k values to test
    verbose : bool
        Whether to print comparison results

    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    if verbose:
        print("🔍 Comparing Clustering Methods")
        print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    results = {}

    for method in methods:
        if verbose:
            print(f"\n📊 Testing {method.upper()} method...")

        method_results = {}

        for k in k_values:
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=20)
            elif method == 'gmm':
                model = GaussianMixture(n_components=k, random_state=42, n_init=3)
            else:
                continue

            # Fit and predict
            model.fit(data_dense)

            if method == 'kmeans':
                labels = model.labels_
                inertia = model.inertia_
            else:
                labels = model.predict(data_dense)
                # Calculate proper WSSE for GMM instead of negative log-likelihood
                inertia = calculate_gmm_wsse(data_dense, model)

            # Calculate metrics
            silhouette_avg = silhouette_score(data_dense, labels)

            method_results[k] = {
                'model': model,
                'labels': labels,
                'inertia': inertia,
                'silhouette_score': silhouette_avg,
                'method': method
            }

            if verbose:
                print(f"   k={k}: Silhouette={silhouette_avg:.3f}, "
                      f"Inertia={inertia:.2f}")

        results[method] = method_results

    if verbose:
        print("\n📋 Method Comparison Summary:")
        print(f"{'Method':<10} {'k':<5} {'Silhouette':<12} {'Inertia':<12}")
        print("-" * 45)

        for method, method_results in results.items():
            for k, result in method_results.items():
                print(f"{method:<10} {k:<5} {result['silhouette_score']:<12.3f} "
                      f"{result['inertia']:<12.2f}")

    return results


def evaluate_clustering_performance(data, labels, verbose=True):
    """
    Evaluate clustering performance using multiple metrics.

    Parameters:
    -----------
    data : array-like
        Original data
    labels : array-like
        Cluster labels
    verbose : bool
        Whether to print evaluation results

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    if verbose:
        print("📊 Evaluating Clustering Performance")
        print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    metrics = {}

    # Silhouette analysis
    silhouette_avg = silhouette_score(data_dense, labels)
    metrics['silhouette_score'] = silhouette_avg

    # Cluster statistics
    n_clusters = len(np.unique(labels))
    cluster_sizes = np.bincount(labels)

    metrics['n_clusters'] = n_clusters
    metrics['cluster_sizes'] = cluster_sizes
    metrics['min_cluster_size'] = np.min(cluster_sizes)
    metrics['max_cluster_size'] = np.max(cluster_sizes)
    metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)

    # Intra-cluster distances
    intra_cluster_distances = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]

        if len(cluster_data) > 1:
            # Calculate pairwise distances within cluster
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(cluster_data)
            # Get upper triangle (excluding diagonal)
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            intra_cluster_distances.extend(upper_tri)

    metrics['mean_intra_cluster_distance'] = np.mean(intra_cluster_distances)
    metrics['std_intra_cluster_distance'] = np.std(intra_cluster_distances)

    # Inter-cluster distances
    cluster_centers = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]
        cluster_center = np.mean(cluster_data, axis=0)
        cluster_centers.append(cluster_center)

    cluster_centers = np.array(cluster_centers)

    if len(cluster_centers) > 1:
        from sklearn.metrics.pairwise import pairwise_distances
        inter_distances = pairwise_distances(cluster_centers)
        upper_tri = inter_distances[np.triu_indices_from(inter_distances, k=1)]
        metrics['mean_inter_cluster_distance'] = np.mean(upper_tri)
        metrics['std_inter_cluster_distance'] = np.std(upper_tri)
    else:
        metrics['mean_inter_cluster_distance'] = 0
        metrics['std_inter_cluster_distance'] = 0

    # Separation ratio
    if metrics['mean_intra_cluster_distance'] > 0:
        metrics['separation_ratio'] = (metrics['mean_inter_cluster_distance']
                                       / metrics['mean_intra_cluster_distance'])
    else:
        metrics['separation_ratio'] = float('inf')

    if verbose:
        print(f"📊 Clustering Performance Metrics:")
        print(f"   🎯 Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"   🎯 Number of Clusters: {metrics['n_clusters']}")
        print(f"   📊 Cluster Sizes: {metrics['cluster_sizes']}")
        print(f"   📊 Size Range: {metrics['min_cluster_size']} - {metrics['max_cluster_size']}")
        print(f"   📊 Cluster Balance: {metrics['cluster_balance']:.3f}")
        print(f"   📊 Mean Intra-cluster Distance: {metrics['mean_intra_cluster_distance']:.3f}")
        print(f"   📊 Mean Inter-cluster Distance: {metrics['mean_inter_cluster_distance']:.3f}")
        print(f"   📊 Separation Ratio: {metrics['separation_ratio']:.3f}")

    return metrics


def parallel_clustering_analysis(data, method='kmeans', k_range=range(2, 10),
                                 n_processes=None, verbose=True):
    """
    Perform clustering analysis in parallel for faster execution.

    Parameters:
    -----------
    data : array-like
        Data to cluster
    method : str
        Clustering method ('kmeans' or 'gmm')
    k_range : range
        Range of k values to test
    n_processes : int, optional
        Number of processes to use
    verbose : bool
        Whether to print progress

    Returns:
    --------
    dict
        Dictionary containing clustering results
    """
    if verbose:
        print(f"🚀 Parallel {method.upper()} Clustering Analysis")
        print("=" * 50)

    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(k_range))

    if verbose:
        print(f"📊 Using {n_processes} processes")
        print(f"📊 Testing k values: {list(k_range)}")

    start_time = datetime.now()

    # Create pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Map clustering function to different k values
        if method == 'kmeans':
            results = pool.starmap(
                _parallel_kmeans_worker,
                zip(repeat(data), k_range, repeat(42))
            )
        elif method == 'gmm':
            results = pool.starmap(
                _parallel_gmm_worker,
                zip(repeat(data), k_range, repeat(42))
            )
        else:
            raise ValueError("Method must be 'kmeans' or 'gmm'")

    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    # Convert results to dictionary
    results_dict = {}
    for result in results:
        k = result['n_clusters']
        results_dict[k] = result

    if verbose:
        print(f"\n✅ Parallel clustering completed in {execution_time:.2f} seconds")
        print(f"📊 Results for {len(results_dict)} k values")

    return results_dict


def _parallel_kmeans_worker(data, k, random_state):
    """
    Worker function for parallel KMeans clustering.

    Parameters:
    -----------
    data : array-like
        Data to cluster
    k : int
        Number of clusters
    random_state : int
        Random state

    Returns:
    --------
    dict
        Clustering results
    """
    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    # Perform clustering
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    model.fit(data_dense)

    # Calculate metrics
    labels = model.labels_
    silhouette_avg = silhouette_score(data_dense, labels)

    return {
        'model': model,
        'labels': labels,
        'inertia': model.inertia_,
        'silhouette_score': silhouette_avg,
        'n_clusters': k,
        'method': 'kmeans'
    }


def _parallel_gmm_worker(data, k, random_state):
    """
    Worker function for parallel GMM clustering.

    Parameters:
    -----------
    data : array-like
        Data to cluster
    k : int
        Number of components
    random_state : int
        Random state

    Returns:
    --------
    dict
        Clustering results
    """
    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    # Perform clustering
    model = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
    model.fit(data_dense)

    # Calculate metrics
    labels = model.predict(data_dense)
    silhouette_avg = silhouette_score(data_dense, labels)

    return {
        'model': model,
        'labels': labels,
        'inertia': calculate_gmm_wsse(data_dense, model),  # Proper WSSE calculation
        'silhouette_score': silhouette_avg,
        'n_clusters': k,
        'method': 'gmm'
    }


def cluster_summary_report(data, labels, feature_names=None, top_n_features=10,
                           verbose=True):
    """
    Generate a comprehensive cluster summary report.

    Parameters:
    -----------
    data : array-like
        Original data
    labels : array-like
        Cluster labels
    feature_names : list, optional
        Names of features
    top_n_features : int
        Number of top features to show per cluster
    verbose : bool
        Whether to print the report

    Returns:
    --------
    dict
        Dictionary containing cluster summary
    """
    if verbose:
        print("📋 Generating Cluster Summary Report")
        print("=" * 50)

    # Convert sparse matrix to dense if needed
    if hasattr(data, 'toarray'):
        data_dense = data.toarray()
    else:
        data_dense = data

    n_clusters = len(np.unique(labels))
    report = {}

    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_data = data_dense[cluster_mask]

        # Basic statistics
        cluster_size = np.sum(cluster_mask)
        cluster_percentage = (cluster_size / len(labels)) * 100

        # Feature analysis
        mean_features = np.mean(cluster_data, axis=0)
        std_features = np.std(cluster_data, axis=0)

        # Top features
        top_feature_indices = np.argsort(mean_features)[-top_n_features:][::-1]

        if feature_names is not None:
            top_features = [
                {
                    'name': feature_names[i],
                    'mean': mean_features[i],
                    'std': std_features[i]
                }
                for i in top_feature_indices
            ]
        else:
            top_features = [
                {
                    'index': i,
                    'mean': mean_features[i],
                    'std': std_features[i]
                }
                for i in top_feature_indices
            ]

        # Cluster summary
        cluster_summary = {
            'cluster_id': cluster_id,
            'size': cluster_size,
            'percentage': cluster_percentage,
            'top_features': top_features,
            'mean_feature_activity': np.mean(np.sum(cluster_data > 0, axis=1)),
            'total_feature_activity': np.sum(cluster_data),
            'cluster_density': np.mean(cluster_data > 0)
        }

        report[cluster_id] = cluster_summary

        if verbose:
            print(f"\n🔍 Cluster {cluster_id} Summary:")
            print(f"   📊 Size: {cluster_size} documents ({cluster_percentage:.1f}%)")
            print(f"   📊 Mean feature activity: {cluster_summary['mean_feature_activity']:.1f}")
            print(f"   📊 Cluster density: {cluster_summary['cluster_density']:.3f}")
            print(f"   🔝 Top {top_n_features} features:")

            for j, feature in enumerate(top_features[:5]):  # Show top 5
                if feature_names is not None:
                    print(f"      {j+1}. {feature['name']}: {feature['mean']:.3f} ± {feature['std']:.3f}")
                else:
                    print(f"      {j+1}. Feature {feature['index']}: {feature['mean']:.3f} ± {feature['std']:.3f}")

    if verbose:
        print(f"\n✅ Cluster summary report completed!")
        print(f"   📊 Total clusters: {n_clusters}")
        print(f"   📊 Total documents: {len(labels)}")

    return report
