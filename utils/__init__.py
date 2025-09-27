"""
Text Preprocessing Demo Utils

This module contains utility functions for text preprocessing, clustering,
and visualization for the BBC News clustering demo.
"""

from .text_preprocessing_utils import (
    load_bbc_dataset,
    comprehensive_text_clean,
    analyze_text_statistics,
    compare_preprocessing_steps,
    plot_preprocessing_comparison
)

from .vectorization_utils import (
    create_tfidf_features,
    analyze_feature_importance,
    plot_tfidf_analysis,
    compare_vectorization_methods,
    apply_dimensionality_reduction
)

from .clustering_utils import (
    perform_clustering_analysis,
    find_optimal_clusters,
    compare_clustering_methods,
    evaluate_clustering_performance
)

from .visualization_utils import (
    plot_beautiful_clusters,
    create_cluster_dashboard,
    plot_silhouette_analysis,
    plot_silhouette_analysis_multiple_k,
    plot_cluster_comparison,
    plot_dimensionality_reduction,
    create_interactive_cluster_plot
)

from .evaluation_utils import (
    calculate_clustering_metrics,
    evaluate_cluster_quality,
    create_evaluation_report,
    plot_evaluation_metrics
)

__all__ = [
    # Text preprocessing
    'load_bbc_dataset',
    'comprehensive_text_clean',
    'analyze_text_statistics',
    'compare_preprocessing_steps',
    'plot_preprocessing_comparison',

    # Vectorization
    'create_tfidf_features',
    'analyze_feature_importance',
    'plot_tfidf_analysis',
    'compare_vectorization_methods',
    'apply_dimensionality_reduction',

    # Clustering
    'perform_clustering_analysis',
    'find_optimal_clusters',
    'compare_clustering_methods',
    'evaluate_clustering_performance',

    # Visualization
    'plot_beautiful_clusters',
    'create_cluster_dashboard',
    'plot_silhouette_analysis',
    'plot_silhouette_analysis_multiple_k',
    'plot_cluster_comparison',
    'plot_dimensionality_reduction',
    'create_interactive_cluster_plot',

    # Evaluation
    'calculate_clustering_metrics',
    'evaluate_cluster_quality',
    'create_evaluation_report',
    'plot_evaluation_metrics'
]
