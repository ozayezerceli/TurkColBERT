"""
Evaluation script comparing baseline PLAID vs custom MUVERA implementation.
This uses our own MUVERA encoding inspired by Google's implementation.
"""

from __future__ import annotations

import time
import csv
import sys
from pathlib import Path
from typing import Literal
import numpy as np

from pylate import evaluation, indexes, models, retrieve, rank
from datasets import load_dataset

# Ensure local package is importable when running the script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

# Import our custom MUVERA encoder
from lateinteractionmodels.muvera import (
    MuveraEncoder,
    MuveraConfig,
    ProjectionType,
    batch_cosine_similarity,
)


def _load_hf_information_retrieval_dataset(
    config: dict[str, str]
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, dict[str, float]]]:
    """Load dataset from Hugging Face."""
    
    dataset_path = config["hf_corpus_path"]
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

    corpus_id_field = config.get("corpus_id_field", "_id")
    corpus_text_field = config.get("corpus_text_field", "text")
    query_id_field = config.get("query_id_field", "_id")
    query_text_field = config.get("query_text_field", "text")
    qrels_query_field = config.get("qrels_query_field", "query-id")
    qrels_doc_field = config.get("qrels_doc_field", "corpus-id")
    qrels_score_field = config.get("qrels_score_field", "score")

    documents = [
        {
            "id": str(row[corpus_id_field]),
            "text": row[corpus_text_field],
        }
        for row in corpus_split
    ]
    queries = {
        str(row[query_id_field]): row[query_text_field]
        for row in queries_split
    }
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


def run_baseline_benchmark(
    dataset_name: str,
    model_name: str,
    dataset_config: dict,
    k_values: list[int] = [100, 250, 500],
) -> dict[str, float | int]:
    """Run baseline benchmark using PLAID index."""
    
    print(f"\n{'='*80}")
    print(f"BASELINE: Running with PLAID index")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=(
            int(dataset_config["query_length"]) if dataset_config["query_length"] else None
        ),
        trust_remote_code=True,
    )
    
    index = indexes.PLAID(override=True)
    retriever = retrieve.ColBERT(index=index)

    # Load dataset
    print("Loading dataset...")
    documents, queries, qrels = _load_hf_information_retrieval_dataset(dataset_config)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Encode documents
    print("\nEncoding documents...")
    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=128,
        is_query=False,
        show_progress_bar=True,
    )

    # Measure indexing time
    print("\nIndexing documents...")
    indexing_start = time.time()
    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )
    indexing_time = time.time() - indexing_start
    print(f"Indexing time: {indexing_time:.2f} seconds")

    # Encode queries
    print("\nEncoding queries...")
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=128,
        is_query=True,
        show_progress_bar=True,
    )

    # Measure query time
    print("\nRetrieving results...")
    query_start = time.time()
    max_k = max(k_values)
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=max_k)
    query_time = time.time() - query_start
    avg_query_time = query_time / len(queries)
    print(f"Total query time: {query_time:.2f} seconds")
    print(f"Average query time: {avg_query_time*1000:.2f} ms")

    # Evaluate
    print("\nEvaluating...")
    metrics = []
    for k in k_values:
        metrics.extend([f"ndcg@{k}", f"recall@{k}", f"hits@{k}", f"precision@{k}"])
    metrics.append("map")
    
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=metrics,
    )

    results["indexing_time_seconds"] = round(indexing_time, 2)
    results["total_query_time_seconds"] = round(query_time, 2)
    results["avg_query_time_ms"] = round(avg_query_time * 1000, 2)
    results["num_documents"] = len(documents)
    results["num_queries"] = len(queries)

    return results


def run_muvera_benchmark(
    dataset_name: str,
    model_name: str,
    dataset_config: dict,
    muvera_config: MuveraConfig,
) -> dict[str, float | int]:
    """Run benchmark using custom MUVERA encoding."""
    
    print(f"\n{'='*80}")
    print(f"CUSTOM MUVERA: Using fixed-dimensional encoding")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Config: {muvera_config.num_simhash_projections} bits, {muvera_config.projection_dimension}D")
    print(f"{'='*80}\n")

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=(
            int(dataset_config["query_length"]) if dataset_config["query_length"] else None
        ),
        trust_remote_code=True,
    )
    
    # Initialize MUVERA encoder
    encoder = MuveraEncoder(muvera_config)

    # Load dataset
    print("Loading dataset...")
    documents, queries, qrels = _load_hf_information_retrieval_dataset(dataset_config)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Encode documents with ColBERT
    print("\nEncoding documents with ColBERT...")
    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=128,
        is_query=False,
        show_progress_bar=True,
    )

    # Apply MUVERA encoding and measure indexing time
    print("\nApplying MUVERA encoding to documents...")
    indexing_start = time.time()
    
    # Convert to fixed-dimensional encodings
    doc_muvera_encodings = []
    for doc_emb in documents_embeddings:
        # doc_emb is (num_tokens, dim)
        encoded = encoder.encode_document(doc_emb)
        doc_muvera_encodings.append(encoded)
    
    doc_muvera_encodings = np.array(doc_muvera_encodings)
    indexing_time = time.time() - indexing_start
    print(f"MUVERA encoding time: {indexing_time:.2f} seconds")
    print(f"Encoded shape: {doc_muvera_encodings.shape}")

    # Encode queries with ColBERT
    print("\nEncoding queries with ColBERT...")
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=128,
        is_query=True,
        show_progress_bar=True,
    )

    # Apply MUVERA encoding to queries and measure query time
    print("\nApplying MUVERA encoding and retrieving...")
    query_start = time.time()
    
    # Encode queries
    query_muvera_encodings = []
    for query_emb in queries_embeddings:
        encoded = encoder.encode_query(query_emb)
        query_muvera_encodings.append(encoded)
    
    query_muvera_encodings = np.array(query_muvera_encodings)
    
    # Compute similarities (this is the retrieval step)
    similarities = batch_cosine_similarity(query_muvera_encodings, doc_muvera_encodings)
    
    query_time = time.time() - query_start
    avg_query_time = query_time / len(queries)
    print(f"Total query time (encoding + retrieval): {query_time:.2f} seconds")
    print(f"Average query time: {avg_query_time*1000:.2f} ms")

    # Convert similarities to retrieval scores format
    # PyLate expects: List of List of dicts with 'id' and 'score' keys
    print("\nFormatting results...")
    scores = []
    doc_ids = [doc["id"] for doc in documents]
    query_ids = list(queries.keys())
    
    for i, query_id in enumerate(query_ids):
        # Get top-100 documents for this query
        query_similarities = similarities[i]
        top_k_indices = np.argsort(-query_similarities)[:100]
        
        # Create list of matches for this query
        query_matches = [
            {"id": doc_ids[idx], "score": float(query_similarities[idx])}
            for idx in top_k_indices
        ]
        scores.append(query_matches)

    # Evaluate
    print("\nEvaluating...")
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=[f"ndcg@{k}" for k in [1, 3, 5, 10, 100]]
        + [f"hits@{k}" for k in [1, 3, 5, 10, 100]]
        + ["map"]
        + ["recall@10", "recall@100"]
        + ["precision@10", "precision@100"],
    )

    results["indexing_time_seconds"] = round(indexing_time, 2)
    results["total_query_time_seconds"] = round(query_time, 2)
    results["avg_query_time_ms"] = round(avg_query_time * 1000, 2)
    results["num_documents"] = len(documents)
    results["num_queries"] = len(queries)
    results["encoding_dimension"] = len(doc_muvera_encodings[0])

    return results


def run_muvera_rerank_benchmark(
    dataset_name: str,
    model_name: str,
    dataset_config: dict,
    muvera_config: MuveraConfig,
    rerank_top_k: int = 100,
) -> dict[str, float | int]:
    """Run benchmark using MUVERA for candidate retrieval + ColBERT reranking."""
    
    print(f"\n{'='*80}")
    print(f"MUVERA + RERANKING: Fast candidate retrieval + exact scoring")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Rerank top-{rerank_top_k} candidates")
    print(f"{'='*80}\n")

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=(
            int(dataset_config["query_length"]) if dataset_config["query_length"] else None
        ),
        trust_remote_code=True,
    )
    
    # Initialize MUVERA encoder
    encoder = MuveraEncoder(muvera_config)

    # Load dataset
    print("Loading dataset...")
    documents, queries, qrels = _load_hf_information_retrieval_dataset(dataset_config)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Encode documents with ColBERT (keep original embeddings for reranking)
    print("\nEncoding documents with ColBERT...")
    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=128,
        is_query=False,
        show_progress_bar=True,
    )

    # Apply MUVERA encoding and measure indexing time
    print("\nApplying MUVERA encoding to documents...")
    indexing_start = time.time()
    
    doc_muvera_encodings = []
    for doc_emb in documents_embeddings:
        encoded = encoder.encode_document(doc_emb)
        doc_muvera_encodings.append(encoded)
    
    doc_muvera_encodings = np.array(doc_muvera_encodings)
    indexing_time = time.time() - indexing_start
    print(f"MUVERA encoding time: {indexing_time:.2f} seconds")

    # Encode queries with ColBERT (keep original for reranking)
    print("\nEncoding queries with ColBERT...")
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=128,
        is_query=True,
        show_progress_bar=True,
    )

    # Phase 1: Fast candidate retrieval with MUVERA
    print(f"\nPhase 1: MUVERA candidate retrieval (top-{rerank_top_k})...")
    query_start = time.time()
    
    query_muvera_encodings = []
    for query_emb in queries_embeddings:
        encoded = encoder.encode_query(query_emb)
        query_muvera_encodings.append(encoded)
    
    query_muvera_encodings = np.array(query_muvera_encodings)
    
    # Get candidate documents using MUVERA
    muvera_similarities = batch_cosine_similarity(query_muvera_encodings, doc_muvera_encodings)
    
    # Phase 2: Rerank top-k candidates with exact ColBERT scoring using PyLate
    print(f"Phase 2: Reranking with exact ColBERT MaxSim using PyLate...")
    doc_ids = [doc["id"] for doc in documents]
    query_ids = list(queries.keys())
    
    # Prepare batch data for PyLate reranking
    batch_documents_ids = []
    batch_queries_embeddings = []
    batch_documents_embeddings = []
    
    for i, query_id in enumerate(query_ids):
        # Get top-k candidates from MUVERA
        query_muvera_scores = muvera_similarities[i]
        candidate_indices = np.argsort(-query_muvera_scores)[:rerank_top_k]
        
        # Prepare data for this query
        candidate_doc_ids = [doc_ids[idx] for idx in candidate_indices]
        candidate_doc_embeddings = [documents_embeddings[idx] for idx in candidate_indices]
        
        batch_documents_ids.append(candidate_doc_ids)
        batch_queries_embeddings.append(queries_embeddings[i])
        batch_documents_embeddings.append(candidate_doc_embeddings)
    
    # Rerank using PyLate's efficient batch reranking
    scores = rank.rerank(
        documents_ids=batch_documents_ids,
        queries_embeddings=batch_queries_embeddings,
        documents_embeddings=batch_documents_embeddings,
    )
    
    query_time = time.time() - query_start
    avg_query_time = query_time / len(queries)
    print(f"Total query time (MUVERA + reranking): {query_time:.2f} seconds")
    print(f"Average query time: {avg_query_time*1000:.2f} ms")

    # Evaluate
    print("\nEvaluating...")
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=[f"ndcg@{k}" for k in [1, 3, 5, 10, 100]]
        + [f"hits@{k}" for k in [1, 3, 5, 10, 100]]
        + ["map"]
        + ["recall@10", "recall@100"]
        + ["precision@10", "precision@100"],
    )

    results["indexing_time_seconds"] = round(indexing_time, 2)
    results["total_query_time_seconds"] = round(query_time, 2)
    results["avg_query_time_ms"] = round(avg_query_time * 1000, 2)
    results["num_documents"] = len(documents)
    results["num_queries"] = len(queries)
    results["rerank_top_k"] = rerank_top_k

    return results


def compare_baseline_vs_custom_muvera(
    dataset_name: str = "SciFact-TR",
    model_name: str = "ozayezerceli/col-ettin-encoder-150M-TR",
    num_simhash_bits: int = 0,
    projection_dim: int = 128,
    rerank_top_k: int = 100,
    output_file: str | None = None,
) -> dict[str, dict[str, float]]:
    """Compare baseline PLAID vs custom MUVERA vs MUVERA+Reranking."""
    
    # Dataset configurations
    dataset_configs = {
        "SciFact-TR": {
            "dataset_name": "SciFact-TR",
            "hf_corpus_path": "AbdulkaderSaoud/scifact-tr",
            "hf_corpus_split": "corpus",
            "hf_queries_path": "AbdulkaderSaoud/scifact-tr",
            "hf_queries_split": "queries",
            "hf_qrels_path": "AbdulkaderSaoud/scifact-tr-qrels",
            "hf_qrels_split": "test",
            "query_length": None,
            "corpus_id_field": "_id",
            "corpus_text_field": "text",
            "query_id_field": "_id",
            "query_text_field": "text",
            "qrels_query_field": "query-id",
            "qrels_doc_field": "corpus-id",
            "qrels_score_field": "score",
        },
    }
    
    dataset_config = dataset_configs[dataset_name]
    
    # Run baseline
    baseline_results = run_baseline_benchmark(
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_config=dataset_config,
    )
    
    # Configure MUVERA
    muvera_config = MuveraConfig(
        dimension=128,  # ColBERT dimension
        num_simhash_projections=num_simhash_bits,
        projection_dimension=projection_dim,
        num_repetitions=1,
        fill_empty_partitions=True,
    )
    
    # Run MUVERA
    muvera_results = run_muvera_benchmark(
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_config=dataset_config,
        muvera_config=muvera_config,
    )
    
    # Run MUVERA + Reranking
    muvera_rerank_results = run_muvera_rerank_benchmark(
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_config=dataset_config,
        muvera_config=muvera_config,
        rerank_top_k=rerank_top_k,
    )
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"MUVERA Config: {num_simhash_bits} bits, {projection_dim}D projection")
    print(f"\n{'Metric':<30} {'Baseline':<15} {'MUVERA':<15} {'MUVERA+Rerank':<15} {'Diff M (%)':<12} {'Diff MR (%)':<12}")
    print("-" * 110)
    
    comparison_metrics = [
        "indexing_time_seconds",
        "avg_query_time_ms",
        "ndcg@10",
        "ndcg@100",
        "recall@10",
        "recall@100",
        "map",
    ]
    
    for metric in comparison_metrics:
        baseline_val = baseline_results.get(metric, 0)
        muvera_val = muvera_results.get(metric, 0)
        muvera_rerank_val = muvera_rerank_results.get(metric, 0)
        
        if baseline_val != 0:
            diff_muvera = ((muvera_val - baseline_val) / baseline_val) * 100
            diff_rerank = ((muvera_rerank_val - baseline_val) / baseline_val) * 100
        else:
            diff_muvera = 0
            diff_rerank = 0
            
        print(f"{metric:<30} {baseline_val:<15.4f} {muvera_val:<15.4f} {muvera_rerank_val:<15.4f} {diff_muvera:+11.2f}% {diff_rerank:+11.2f}%")
    
    # Save to CSV if requested
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Metric', 'Value'])
            
            for key, value in baseline_results.items():
                if isinstance(value, (int, float)):
                    writer.writerow(['Baseline', key, value])
            
            for key, value in muvera_results.items():
                if isinstance(value, (int, float)):
                    writer.writerow(['Custom_MUVERA', key, value])
            
            for key, value in muvera_rerank_results.items():
                if isinstance(value, (int, float)):
                    writer.writerow(['MUVERA_Reranking', key, value])
        
        print(f"\nResults saved to: {output_path}")
    
    return {
        "baseline": baseline_results,
        "muvera": muvera_results,
        "muvera_reranking": muvera_rerank_results,
    }


if __name__ == "__main__":
    results = compare_baseline_vs_custom_muvera(
        dataset_name="SciFact-TR",
        model_name="ozayezerceli/col-ettin-encoder-32M-TR",
        num_simhash_bits=0,   # No partitioning for small dataset (recommended)
        projection_dim=128,   # Keep full dimension (recommended)
        rerank_top_k=100,     # Rerank top-100 candidates
        output_file="benchmarks/custom_muvera_comparison_results.csv",
    )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print("\nResults summary:")
    print(f"  Baseline metrics: {len([k for k, v in results['baseline'].items() if isinstance(v, (int, float))])}")
    print(f"  MUVERA metrics: {len([k for k, v in results['muvera'].items() if isinstance(v, (int, float))])}")
    print(f"  MUVERA+Reranking metrics: {len([k for k, v in results['muvera_reranking'].items() if isinstance(v, (int, float))])}")
    
    # Print key takeaways
    baseline = results['baseline']
    muvera = results['muvera']
    muvera_rerank = results['muvera_reranking']
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    
    print("\n1. MUVERA (Fixed-dimensional encoding):")
    print(f"   Query time: {muvera.get('avg_query_time_ms', 0):.2f}ms "
          f"(vs {baseline.get('avg_query_time_ms', 0):.2f}ms baseline)")
    print(f"   NDCG@10: {muvera.get('ndcg@10', 0):.4f} "
          f"({(muvera.get('ndcg@10', 0) / baseline.get('ndcg@10', 1) * 100):.1f}% of baseline)")
    
    print("\n2. MUVERA + Reranking (Best of both worlds):")
    print(f"   Query time: {muvera_rerank.get('avg_query_time_ms', 0):.2f}ms "
          f"(vs {baseline.get('avg_query_time_ms', 0):.2f}ms baseline)")
    print(f"   NDCG@10: {muvera_rerank.get('ndcg@10', 0):.4f} "
          f"({(muvera_rerank.get('ndcg@10', 0) / baseline.get('ndcg@10', 1) * 100):.1f}% of baseline)")
    
    print("\n3. Recommendation:")
    if muvera_rerank.get('ndcg@10', 0) >= baseline.get('ndcg@10', 0) * 0.95:
        if muvera_rerank.get('avg_query_time_ms', float('inf')) < baseline.get('avg_query_time_ms', 0):
            print("   ✅ MUVERA+Reranking achieves >95% quality with speedup!")
        else:
            print("   ⚠️  MUVERA+Reranking maintains quality but may be slower on small datasets")
    else:
        print("   ⚠️  Try adjusting hyperparameters (see MUVERA_Hyperparameter_Guide.md)")
    
    print("\n" + "="*80)

