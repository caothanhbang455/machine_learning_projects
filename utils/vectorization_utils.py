import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def create_tfidf_features(texts, max_features=None, ngram_range=(1, 2),
                          min_df=0.02, max_df=0.95, verbose=True):
    if verbose:
        print("Creating TF-IDF Features")
        print("=" * 50)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )

    if verbose:
        print(f"TF-IDF Parameters:")
        print(f"   Max features: {max_features}")
        print(f"   N-gram range: {ngram_range}")
        print(f"   Min document frequency: {min_df}")
        print(f"   Max document frequency: {max_df}")

    feature_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    if verbose:
        print(f"\nTF-IDF Features Created:")
        print(f"   Documents: {feature_matrix.shape[0]}")
        print(f"   Features: {feature_matrix.shape[1]}")
        print(f"   Sparsity: {1.0 - (feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1])):.3f}")

    return feature_matrix, vectorizer, feature_names


def analyze_feature_importance(feature_matrix, feature_names, method='mean', top_n=20):
    print(f"Analyzing Feature Importance ({method} method)")
    print("=" * 50)

    if method == 'mean':
        scores = np.array(feature_matrix.mean(axis=0)).flatten()
    elif method == 'max':
        scores = np.array(feature_matrix.max(axis=0)).flatten()
    elif method == 'std':
        scores = np.array(feature_matrix.std(axis=0)).flatten()
    else:
        raise ValueError("Method must be 'mean', 'max', or 'std'")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': scores
    }).sort_values('importance', ascending=False)

    print(f"Top {top_n} Most important features:")
    print(importance_df.head(top_n))

    return importance_df


def plot_tfidf_analysis(feature_matrix, feature_names, figsize=(16, 10)):
    print("Creating TF-IDF Analysis Plots")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('TF-IDF Feature Analysis Dashboard', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    mean_scores = np.array(feature_matrix.mean(axis=0)).flatten()
    top_features_idx = np.argsort(mean_scores)[-20:]

    ax1.barh(range(len(top_features_idx)), mean_scores[top_features_idx], color='skyblue')
    ax1.set_yticks(range(len(top_features_idx)))
    ax1.set_yticklabels([feature_names[i] for i in top_features_idx])
    ax1.set_title('Top 20 Features by Mean TF-IDF')
    ax1.set_xlabel('Mean TF-IDF Score')

    ax2 = axes[0, 1]
    non_zero_scores = feature_matrix.data
    ax2.hist(non_zero_scores, bins=50, alpha=0.7, color='lightcoral')
    ax2.set_title('TF-IDF Score Distribution')
    ax2.set_xlabel('TF-IDF Score')
    ax2.set_ylabel('Frequency')
    ax2.axvline(non_zero_scores.mean(), color='red', linestyle='--',
                label=f'Mean: {non_zero_scores.mean():.3f}')
    ax2.legend()

    ax3 = axes[0, 2]
    sparsity_per_feature = 1.0 - (np.array((feature_matrix > 0).sum(axis=0)).flatten() / feature_matrix.shape[0])
    ax3.hist(sparsity_per_feature, bins=50, alpha=0.7, color='lightgreen')
    ax3.set_title('Feature Sparsity Distribution')
    ax3.set_xlabel('Sparsity (1 - presence ratio)')
    ax3.set_ylabel('Number of Features')
    ax3.axvline(sparsity_per_feature.mean(), color='red', linestyle='--',
                label=f'Mean: {sparsity_per_feature.mean():.3f}')
    ax3.legend()

    ax4 = axes[1, 0]
    doc_lengths = np.array((feature_matrix > 0).sum(axis=1)).flatten()
    ax4.hist(doc_lengths, bins=30, alpha=0.7, color='gold')
    ax4.set_title('Document Length Distribution')
    ax4.set_xlabel('Number of Non-zero Features')
    ax4.set_ylabel('Number of Documents')
    ax4.axvline(doc_lengths.mean(), color='red', linestyle='--',
                label=f'Mean: {doc_lengths.mean():.1f}')
    ax4.legend()

    ax5 = axes[1, 1]
    if hasattr(feature_matrix, 'toarray'):
        feature_dense = feature_matrix.toarray()
    else:
        feature_dense = feature_matrix
    feature_vars = np.var(feature_dense, axis=0)
    ax5.hist(feature_vars, bins=50, alpha=0.7, color='plum')
    ax5.set_title('Feature Variance Distribution')
    ax5.set_xlabel('Variance')
    ax5.set_ylabel('Number of Features')
    ax5.axvline(feature_vars.mean(), color='red', linestyle='--',
                label=f'Mean: {feature_vars.mean():.4f}')
    ax5.legend()

    ax6 = axes[1, 2]
    unigrams = sum(1 for name in feature_names if ' ' not in name)
    bigrams = sum(1 for name in feature_names if ' ' in name)

    ax6.pie([unigrams, bigrams], labels=['Unigrams', 'Bigrams'], autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'])
    ax6.set_title('Feature Type Distribution')

    plt.tight_layout()
    plt.show()

    print("TF-IDF analysis plots completed!")


def compare_vectorization_methods(texts, methods=['tfidf', 'count'], verbose=True):
    if verbose:
        print("Comparing Vectorization Methods")
        print("=" * 50)

    results = {}

    for method in methods:
        if verbose:
            print(f"\nAnalyzing {method.upper()} method...")

        if method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                                         min_df=0.02, stop_words='english')
        elif method == 'count':
            vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2),
                                         min_df=0.02, stop_words='english')
        else:
            continue

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        sparsity = 1.0 - (feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1]))
        mean_score = np.mean(feature_matrix.data)
        std_score = np.std(feature_matrix.data)

        results[method] = {
            'vectorizer': vectorizer,
            'feature_matrix': feature_matrix,
            'feature_names': feature_names,
            'sparsity': sparsity,
            'mean_score': mean_score,
            'std_score': std_score,
            'n_features': feature_matrix.shape[1],
            'n_documents': feature_matrix.shape[0]
        }

        if verbose:
            print(f"   Shape: {feature_matrix.shape}")
            print(f"   Sparsity: {sparsity:.3f}")
            print(f"   Mean score: {mean_score:.3f}")
            print(f"   Std score: {std_score:.3f}")

    if verbose:
        print("\nMethod Comparison Summary:")
        print(f"{'Method':<10} {'Features':<10} {'Sparsity':<10} {'Mean Score':<12} {'Std Score':<12}")
        print("-" * 60)
        for method, result in results.items():
            print(f"{method:<10} {result['n_features']:<10} {result['sparsity']:<10.3f} "
                  f"{result['mean_score']:<12.3f} {result['std_score']:<12.3f}")

    return results


def apply_dimensionality_reduction(feature_matrix, method='pca', n_components=None,
                                   variance_threshold=0.9, verbose=True):
    if verbose:
        print(f"Applying {method.upper()} Dimensionality Reduction")
        print("=" * 50)

    if hasattr(feature_matrix, 'toarray'):
        feature_dense = feature_matrix.toarray()
    else:
        feature_dense = feature_matrix

    if method == 'pca':
        if n_components is None:
            reducer = PCA(n_components=variance_threshold)
        else:
            reducer = PCA(n_components=n_components)

        reduced_matrix = reducer.fit_transform(feature_dense)
        explained_variance = reducer.explained_variance_ratio_

        if verbose:
            print(f"Original shape: {feature_dense.shape}")
            print(f"Reduced shape: {reduced_matrix.shape}")
            print(f"Variance explained: {explained_variance.sum():.3f}")
            print(f"Components kept: {len(explained_variance)}")
    else:
        raise ValueError("Only 'pca' method is currently supported")

    return reduced_matrix, reducer, explained_variance


def plot_dimensionality_reduction_analysis(original_matrix, reduced_matrix,
                                           explained_variance, figsize=(16, 10)):
    print("Creating Dimensionality Reduction Analysis Plots")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Dimensionality Reduction Analysis', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
    ax1.set_title('Explained Variance Ratio')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    cumulative_variance = np.cumsum(explained_variance)
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    ax2.axhline(y=0.9, color='green', linestyle='--', label='90% Variance')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    top_components = min(20, len(explained_variance))
    ax3.bar(range(1, top_components + 1), explained_variance[:top_components],
            color='skyblue', alpha=0.7)
    ax3.set_title(f'Top {top_components} Components')
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Explained Variance')

    ax4 = axes[1, 0]
    dimensions = ['Original', 'Reduced']
    dim_values = [original_matrix.shape[1], reduced_matrix.shape[1]]
    colors = ['lightcoral', 'lightblue']

    bars = ax4.bar(dimensions, dim_values, color=colors, alpha=0.7)
    ax4.set_title('Dimensionality Comparison')
    ax4.set_ylabel('Number of Dimensions')

    for bar, value in zip(bars, dim_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(dim_values) * 0.01,
                 f'{value}', ha='center', va='bottom', fontweight='bold')

    ax5 = axes[1, 1]
    if len(explained_variance) >= 2:
        ax5.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], alpha=0.6, c='purple')
        ax5.set_title('First Two Principal Components')
        ax5.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        ax5.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    compression_ratio = reduced_matrix.shape[1] / original_matrix.shape[1]
    compression_data = [compression_ratio, 1 - compression_ratio]
    labels = ['Retained', 'Compressed']
    colors = ['lightgreen', 'lightcoral']

    ax6.pie(compression_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('Compression Ratio')

    plt.tight_layout()
    plt.show()

    print("Dimensionality reduction analysis completed!")


def get_feature_statistics(feature_matrix, feature_names, verbose=True):
    if verbose:
        print("Calculating Feature Statistics")
        print("=" * 50)

    stats = {}

    stats['n_features'] = feature_matrix.shape[1]
    stats['n_documents'] = feature_matrix.shape[0]
    stats['sparsity'] = 1.0 - (feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1]))

    non_zero_scores = feature_matrix.data
    stats['mean_score'] = np.mean(non_zero_scores)
    stats['std_score'] = np.std(non_zero_scores)
    stats['min_score'] = np.min(non_zero_scores)
    stats['max_score'] = np.max(non_zero_scores)

    feature_presence = np.array((feature_matrix > 0).sum(axis=0)).flatten()
    stats['mean_presence'] = np.mean(feature_presence)
    stats['std_presence'] = np.std(feature_presence)

    unigrams = sum(1 for name in feature_names if ' ' not in name)
    bigrams = sum(1 for name in feature_names if ' ' in name)
    stats['unigrams'] = unigrams
    stats['bigrams'] = bigrams

    if verbose:
        print(f"Matrix Statistics:")
        print(f"   Documents: {stats['n_documents']}")
        print(f"   Features: {stats['n_features']}")
        print(f"   Sparsity: {stats['sparsity']:.3f}")
        print(f"   Mean score: {stats['mean_score']:.3f}")
        print(f"   Score range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
        print(f"   Mean feature presence: {stats['mean_presence']:.1f} documents")
        print(f"   Unigrams: {stats['unigrams']}")
        print(f"   Bigrams: {stats['bigrams']}")

    return stats
