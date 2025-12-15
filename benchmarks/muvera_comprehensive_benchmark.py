"""
Comprehensive MUVERA Benchmark Script

Tests MUVERA with different encoding dimensions across multiple datasets
and generates recall@N and NDCG@N plots for analysis.
"""

from __future__ import annotations

import time
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pylate import evaluation, models
from datasets import load_dataset

# Import our custom MUVERA encoder
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))
from lateinteractionmodels.muvera import (
    MuveraEncoder,
    MuveraConfig,
    ProjectionType,
    batch_cosine_similarity,
)


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def _load_hf_information_retrieval_dataset(
    config: dict[str, str]
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, dict[str, float]]]:
    """Load dataset from Hugging Face."""
    
    dataset_path = config["hf_corpus_path"]
    
    # Check if dataset uses 'name' parameter (trmteb datasets) or separate paths
    use_names = config.get("use_names", False)
    
    if use_names:
        # trmteb format: use name parameter
        corpus_split = load_dataset(dataset_path, name="corpus", split="corpus")
        queries_split = load_dataset(dataset_path, name="queries", split="queries")
        qrels_split = load_dataset(dataset_path, name="default", split="test")
    else:
        # AbdulkaderSaoud format: separate dataset paths
        corpus_split = load_dataset(
            path=config["hf_corpus_path"],
            split=config["hf_corpus_split"],
        )
        queries_path = config.get("hf_queries_path") or config["hf_corpus_path"]
        queries_split = load_dataset(
            path=queries_path,
            split=config["hf_queries_split"],
        )
        qrels_split = load_dataset(
            path=config["hf_qrels_path"],
            split=config["hf_qrels_split"],
        )

    # Field mappings
    corpus_id_field = config.get("corpus_id_field", "_id")
    corpus_text_field = config.get("corpus_text_field", "text")
    query_id_field = config.get("query_id_field", "_id")
    query_text_field = config.get("query_text_field", "text")
    qrels_query_field = config.get("qrels_query_field", "query-id")
    qrels_doc_field = config.get("qrels_doc_field", "corpus-id")
    qrels_score_field = config.get("qrels_score_field", "score")

    # Build documents
    documents = [
        {
            "id": str(row[corpus_id_field]),
            "text": row[corpus_text_field],
        }
        for row in corpus_split
    ]
    
    # Build queries
    queries = {
        str(row[query_id_field]): row[query_text_field]
        for row in queries_split
    }
    
    # Build qrels
    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_split:
        query_id = str(row[qrels_query_field])
        doc_id = str(row[qrels_doc_field])
        score = row[qrels_score_field]
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 1.0
        qrels.setdefault(query_id, {})[doc_id] = score_value

    return documents, queries, qrels


def run_muvera_with_dimension(
    dataset_name: str,
    model_name: str,
    dataset_config: dict,
    encoding_dimension: int,
    recall_at_n: List[int] = [100, 250, 500, 750, 1000],
) -> Dict[str, float]:
    """
    Run MUVERA benchmark with a specific encoding dimension.
    
    Args:
        dataset_name: Name of the dataset
        model_name: ColBERT model to use
        dataset_config: Dataset configuration
        encoding_dimension: Target encoding dimension (128, 512, 1024, 2048)
        recall_at_n: List of N values for Recall@N and NDCG@N
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Encoding Dimension: {encoding_dimension}")
    print(f"{'='*80}\n")

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=(
            int(dataset_config["query_length"]) if dataset_config.get("query_length") else None
        ),
        trust_remote_code=True,
    )
    
    # Configure MUVERA based on target encoding dimension
    # For encoding_dim, we use: num_partitions * projection_dim = encoding_dim
    # Strategy: Use fewer partitions for better quality
    if encoding_dimension == 128:
        num_bits = 0
        proj_dim = 128
        proj_type = ProjectionType.IDENTITY
    elif encoding_dimension == 512:
        num_bits = 2  # 4 partitions
        proj_dim = 128
        proj_type = ProjectionType.IDENTITY
    elif encoding_dimension == 1024:
        num_bits = 3  # 8 partitions
        proj_dim = 128
        proj_type = ProjectionType.IDENTITY
    elif encoding_dimension == 2048:
        num_bits = 4  # 16 partitions
        proj_dim = 128
        proj_type = ProjectionType.IDENTITY
    else:
        raise ValueError(f"Unsupported encoding dimension: {encoding_dimension}")
    
    muvera_config = MuveraConfig(
        dimension=128,  # ColBERT dimension
        num_simhash_projections=num_bits,
        projection_type=proj_type,
        projection_dimension=proj_dim,
        num_repetitions=1,
        fill_empty_partitions=True,
    )
    
    print(f"MUVERA Config: {num_bits} bits, {proj_dim}D projection, {proj_type.value}")
    
    encoder = MuveraEncoder(muvera_config)

    # Load dataset
    print("Loading dataset...")
    documents, queries, qrels = _load_hf_information_retrieval_dataset(dataset_config)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Encode documents
    print("\nEncoding documents with ColBERT...")
    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=128,
        is_query=False,
        show_progress_bar=True,
    )

    # Apply MUVERA encoding
    print("\nApplying MUVERA encoding to documents...")
    indexing_start = time.time()
    
    doc_muvera_encodings = []
    for doc_emb in documents_embeddings:
        encoded = encoder.encode_document(doc_emb)
        doc_muvera_encodings.append(encoded)
    
    doc_muvera_encodings = np.array(doc_muvera_encodings)
    indexing_time = time.time() - indexing_start
    print(f"MUVERA encoding time: {indexing_time:.2f} seconds")
    print(f"Encoded shape: {doc_muvera_encodings.shape}")

    # Encode queries
    print("\nEncoding queries with ColBERT...")
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=128,
        is_query=True,
        show_progress_bar=True,
    )

    # Apply MUVERA encoding and retrieve
    print("\nApplying MUVERA encoding and retrieving...")
    query_start = time.time()
    
    query_muvera_encodings = []
    for query_emb in queries_embeddings:
        encoded = encoder.encode_query(query_emb)
        query_muvera_encodings.append(encoded)
    
    query_muvera_encodings = np.array(query_muvera_encodings)
    
    # Compute similarities
    similarities = batch_cosine_similarity(query_muvera_encodings, doc_muvera_encodings)
    
    query_time = time.time() - query_start
    avg_query_time = query_time / len(queries)
    print(f"Total query time: {query_time:.2f} seconds")
    print(f"Average query time: {avg_query_time*1000:.2f} ms")

    # Convert to retrieval scores format
    print("\nFormatting results...")
    scores = []
    doc_ids = [doc["id"] for doc in documents]
    query_ids = list(queries.keys())
    
    max_k = max(recall_at_n)
    
    for i, query_id in enumerate(query_ids):
        query_similarities = similarities[i]
        top_k_indices = np.argsort(-query_similarities)[:max_k]
        
        query_matches = [
            {"id": doc_ids[idx], "score": float(query_similarities[idx])}
            for idx in top_k_indices
        ]
        scores.append(query_matches)

    # Evaluate
    print("\nEvaluating...")
    
    # Build metrics list - only using standard IR metrics
    metrics = []
    for n in recall_at_n:
        metrics.extend([f"ndcg@{n}", f"recall@{n}", f"precision@{n}"])
    metrics.append("map")
    
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=metrics,
    )

    # Add timing and config info
    results["indexing_time_seconds"] = round(indexing_time, 2)
    results["avg_query_time_ms"] = round(avg_query_time * 1000, 2)
    results["encoding_dimension"] = encoding_dimension
    results["num_documents"] = len(documents)
    results["num_queries"] = len(queries)

    return results


def run_comprehensive_benchmark(
    model_name: str = "ozayezerceli/col-ettin-encoder-32M-TR",
    encoding_dimensions: List[int] = [128, 512, 1024, 2048],
    recall_at_n: List[int] = [100, 250, 500, 750, 1000],
    output_dir: str = "benchmarks/BenchmarkResults/muvera_results",
):
    """
    Run comprehensive MUVERA benchmark across datasets and dimensions.
    
    Args:
        model_name: ColBERT model to use
        encoding_dimensions: List of encoding dimensions to test
        recall_at_n: List of N values for metrics
        output_dir: Directory to save results and plots
    """
    
    # Dataset configurations
    dataset_configs = {
        "SciFact-TR": {
            "hf_corpus_path": "AbdulkaderSaoud/scifact-tr",
            "hf_corpus_split": "corpus",
            "hf_queries_path": "AbdulkaderSaoud/scifact-tr",
            "hf_queries_split": "queries",
            "hf_qrels_path": "AbdulkaderSaoud/scifact-tr-qrels",
            "hf_qrels_split": "test",
            "query_length": None,
            "use_names": False,  # Uses separate dataset paths
        },
        "NFCorpus-TR": {
            "hf_corpus_path": "trmteb/nfcorpus-tr",
            "query_length": None,
            "use_names": True,  # Uses name parameter
        },
        "ArguAna-TR": {
            "hf_corpus_path": "trmteb/arguana-tr",
            "query_length": None,
            "use_names": True,  # Uses name parameter
        },
        "Scidocs-TR": {
            "hf_corpus_path": "trmteb/scidocs-tr",
            "query_length": None,
            "use_names": True,  # Uses name parameter
        },
        "FiQA-TR": {
            "hf_corpus_path": "trmteb/fiqa-tr",
            "query_length": None,
            "use_names": True,  # Uses name parameter
        },
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Run benchmarks
    for dataset_name, dataset_config in dataset_configs.items():
        print(f"\n{'#'*80}")
        print(f"# BENCHMARKING: {dataset_name}")
        print(f"{'#'*80}\n")
        
        dataset_results = {}
        
        for dim in encoding_dimensions:
            try:
                results = run_muvera_with_dimension(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    dataset_config=dataset_config,
                    encoding_dimension=dim,
                    recall_at_n=recall_at_n,
                )
                dataset_results[dim] = results
                
                print(f"\nResults for {dataset_name} @ {dim}D:")
                print(f"  NDCG@100: {results.get('ndcg@100', 0):.4f}")
                print(f"  Recall@100: {results.get('recall@100', 0):.4f}")
                print(f"  Query time: {results.get('avg_query_time_ms', 0):.2f}ms")
                
            except Exception as e:
                print(f"ERROR with {dataset_name} @ {dim}D: {e}")
                continue
        
        all_results[dataset_name] = dataset_results
    
    # Save results to JSON
    results_file = output_path / "muvera_comprehensive_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save to CSV
    csv_file = output_path / "muvera_comprehensive_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Dimension', 'Metric', 'Value'])
        
        for dataset_name, dataset_results in all_results.items():
            for dim, results in dataset_results.items():
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        writer.writerow([dataset_name, dim, metric, value])
    
    print(f"CSV saved to: {csv_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results, recall_at_n, output_path)
    
    return all_results


def generate_plots(
    all_results: Dict[str, Dict[int, Dict[str, float]]],
    recall_at_n: List[int],
    output_path: Path,
):
    """Generate recall@N and NDCG@N plots for each dataset."""
    
    # Color palette for different dimensions
    colors = {
        128: '#1f77b4',   # Blue
        512: '#ff7f0e',   # Orange
        1024: '#2ca02c',  # Green
        2048: '#d62728',  # Red
    }
    
    line_styles = {
        128: '-',
        512: '--',
        1024: '-.',
        2048: ':',
    }
    
    markers = {
        128: 'o',
        512: 's',
        1024: '^',
        2048: 'D',
    }
    
    # Create plots for each dataset
    for dataset_name, dataset_results in all_results.items():
        if not dataset_results:
            continue
        
        # Create figure with 2 subplots (Recall and NDCG)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{dataset_name} - MUVERA Performance', fontsize=16, fontweight='bold')
        
        # Plot Recall@N
        for dim in sorted(dataset_results.keys()):
            results = dataset_results[dim]
            recall_values = [results.get(f'recall@{n}', 0) for n in recall_at_n]
            
            ax1.plot(
                recall_at_n,
                recall_values,
                label=f'{dim}D',
                color=colors.get(dim, 'gray'),
                linestyle=line_styles.get(dim, '-'),
                marker=markers.get(dim, 'o'),
                markersize=8,
                linewidth=2.5,
            )
        
        ax1.set_xlabel('Recall@N', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Recall', fontsize=12, fontweight='bold')
        ax1.set_title('Recall@N vs N', fontsize=14, fontweight='bold')
        ax1.legend(title='Encoding Dim', fontsize=10, title_fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Plot NDCG@N
        for dim in sorted(dataset_results.keys()):
            results = dataset_results[dim]
            ndcg_values = [results.get(f'ndcg@{n}', 0) for n in recall_at_n]
            
            ax2.plot(
                recall_at_n,
                ndcg_values,
                label=f'{dim}D',
                color=colors.get(dim, 'gray'),
                linestyle=line_styles.get(dim, '-'),
                marker=markers.get(dim, 'o'),
                markersize=8,
                linewidth=2.5,
            )
        
        ax2.set_xlabel('Recall@N', fontsize=12, fontweight='bold')
        ax2.set_ylabel('NDCG', fontsize=12, fontweight='bold')
        ax2.set_title('NDCG@N vs N', fontsize=14, fontweight='bold')
        ax2.legend(title='Encoding Dim', fontsize=10, title_fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save figure
        safe_dataset_name = dataset_name.replace('/', '_').replace(' ', '_')
        plot_file = output_path / f"{safe_dataset_name}_performance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_file}")
        plt.close()
    
    # Create combined comparison plot
    create_combined_plot(all_results, recall_at_n, output_path, colors, line_styles, markers)


def create_combined_plot(
    all_results: Dict[str, Dict[int, Dict[str, float]]],
    recall_at_n: List[int],
    output_path: Path,
    colors: Dict,
    line_styles: Dict,
    markers: Dict,
):
    """Create a grid comparing all datasets."""
    
    datasets = list(all_results.keys())
    if len(datasets) == 0:
        return
    
    # Create appropriate grid based on number of datasets
    if len(datasets) <= 4:
        nrows, ncols = 2, 2
    elif len(datasets) <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
    fig.suptitle('MUVERA Performance Across Datasets', fontsize=18, fontweight='bold')
    
    # Flatten axes for easy iteration
    axes_flat = axes.flat if len(datasets) > 1 else [axes]
    
    for idx, dataset_name in enumerate(datasets):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        dataset_results = all_results[dataset_name]
        
        for dim in sorted(dataset_results.keys()):
            results = dataset_results[dim]
            recall_values = [results.get(f'recall@{n}', 0) for n in recall_at_n]
            
            ax.plot(
                recall_at_n,
                recall_values,
                label=f'{dim}D',
                color=colors.get(dim, 'gray'),
                linestyle=line_styles.get(dim, '-'),
                marker=markers.get(dim, 'o'),
                markersize=6,
                linewidth=2,
            )
        
        ax.set_xlabel('Recall@N', fontsize=11, fontweight='bold')
        ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
        ax.set_title(dataset_name, fontsize=13, fontweight='bold')
        ax.legend(title='Encoding Dim', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    # Hide unused subplots
    for idx in range(len(datasets), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    
    combined_file = output_path / "all_datasets_comparison.png"
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {combined_file}")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE MUVERA BENCHMARK")
    print("="*80)
    print("\nThis will test MUVERA with different encoding dimensions:")
    print("  - 128D, 512D, 1024D, 2048D")
    print("\nOn datasets:")
    print("  - SciFact-TR")
    print("  - NFCorpus-TR")
    print("  - ArguAna-TR")
    print("  - Scidocs-TR")
    print("  - FiQA-TR")
    print("\nMetrics: NDCG@N, Recall@N, Precision@N for N = 100, 250, 500, 750, 1000")
    print("\nThis may take 2-3 hours to complete...\n")
    
    results = run_comprehensive_benchmark(
        model_name="ozayezerceli/col-ettin-encoder-32M-TR",
        encoding_dimensions=[128, 512, 1024, 2048],
        recall_at_n=[100, 250, 500, 750, 1000],
        output_dir="benchmarks/BenchmarkResults/muvera_results",
    )
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print("\nResults and plots saved to: benchmarks/BenchmarkResults/muvera_results/")

