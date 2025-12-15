"""Evaluation script comparing baseline vs MUVERA algorithm."""

from __future__ import annotations

import time
import csv
from pathlib import Path
from typing import Literal

from pylate import evaluation, indexes, models, retrieve
from datasets import load_dataset
import numpy as np


def _load_hf_information_retrieval_dataset(
    config: dict[str, str]
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, dict[str, float]]]:
    """Load dataset from Hugging Face."""
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


def run_benchmark(
    dataset_name: str = "SciFact-TR",
    model_name: str = "ozayezerceli/col-ettin-encoder-150M-TR",
    use_muvera: bool = False,
    use_reranking: bool = False,
    muvera_top_k: int = 100,
    reranking_top_k: int = 100,
) -> dict[str, float | int]:
    """Run benchmark with or without MUVERA algorithm and reranking.
    
    Args:
        dataset_name: Name of the dataset to evaluate
        model_name: HuggingFace model name or path
        use_muvera: Whether to use MUVERA algorithm
        use_reranking: Whether to use reranking after MUVERA
        muvera_top_k: Top-k value for MUVERA filtering
        reranking_top_k: Top-k documents to retrieve before reranking
        
    Returns:
        Dictionary containing metrics and timing information
    """
    # Define dataset configurations
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
        "Arguana-TR": {
            "dataset_name": "Arguana-TR",
            "hf_corpus_path": "AbdulkaderSaoud/arguana-tr",
            "hf_corpus_split": "corpus",
            "hf_queries_path": "AbdulkaderSaoud/arguana-tr",
            "hf_queries_split": "queries",
            "hf_qrels_path": "AbdulkaderSaoud/arguana-tr-qrels",
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
        "Scidocs-TR": {
            "dataset_name": "Scidocs-TR",
            "hf_corpus_path": "AbdulkaderSaoud/scidocs-tr",
            "hf_corpus_split": "corpus",
            "hf_queries_path": "AbdulkaderSaoud/scidocs-tr",
            "hf_queries_split": "queries",
            "hf_qrels_path": "AbdulkaderSaoud/scidocs-tr-qrels",
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
    
    config = dataset_configs.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\n{'='*80}")
    print(f"Running benchmark for {dataset_name}")
    print(f"Model: {model_name}")
    print(f"MUVERA: {'Enabled' if use_muvera else 'Disabled'}")
    print(f"Reranking: {'Enabled' if use_reranking else 'Disabled'}")
    print(f"{'='*80}\n")

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=(
            int(config["query_length"]) if config["query_length"] else None
        ),
        trust_remote_code=True,
    )
    
    # Initialize index with or without MUVERA
    if use_muvera:
        index = indexes.Voyager(override=True)  # MUVERA uses Voyager index
    else:
        index = indexes.PLAID(override=True)  # Baseline uses PLAID
    
    retriever = retrieve.ColBERT(index=index)

    # Load dataset
    print("Loading dataset...")
    documents, queries, qrels = _load_hf_information_retrieval_dataset(config)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Encode documents and measure indexing time
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
    
    if use_muvera and use_reranking:
        # MUVERA + Reranking: First retrieve candidates with Voyager (fast), then rerank with exact scoring
        print(f"Phase 1: MUVERA retrieval (top-{reranking_top_k} candidates)...")
        candidate_scores = retriever.retrieve(
            queries_embeddings=queries_embeddings, 
            k=reranking_top_k
        )
        
        print(f"Phase 2: Reranking with exact ColBERT scoring...")
        # Rerank using exact ColBERT scoring
        scores = retriever.rerank(
            queries_embeddings=queries_embeddings,
            scores=candidate_scores,
            k=100,
        )
    elif use_muvera:
        # MUVERA only: Fast approximate retrieval
        scores = retriever.retrieve(
            queries_embeddings=queries_embeddings, 
            k=muvera_top_k
        )
    else:
        # Baseline: Direct retrieval with PLAID
        scores = retriever.retrieve(
            queries_embeddings=queries_embeddings, 
            k=100
        )
    
    query_time = time.time() - query_start
    avg_query_time = query_time / len(queries)
    print(f"Total query time: {query_time:.2f} seconds")
    print(f"Average query time: {avg_query_time*1000:.2f} ms")

    # Evaluate
    print("\nEvaluating...")
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=[f"ndcg@{k}" for k in [1, 3, 5, 10, 100]]
        + [f"hits@{k}" for k in [1, 3, 5, 10, 100]]
        + ["map"]
        + ["recall@10", "recall@100"]
        + ["precision@10", "precision@100"],
    )

    # Add timing information to results
    results["indexing_time_seconds"] = round(indexing_time, 2)
    results["total_query_time_seconds"] = round(query_time, 2)
    results["avg_query_time_ms"] = round(avg_query_time * 1000, 2)
    results["num_documents"] = len(documents)
    results["num_queries"] = len(queries)
    results["muvera_enabled"] = use_muvera

    return results


def compare_muvera_vs_baseline(
    dataset_name: str = "SciFact-TR",
    model_name: str = "ozayezerceli/col-ettin-encoder-150M-TR",
    muvera_top_k: int = 100,
    reranking_top_k: int = 100,
    output_file: str | None = None,
) -> dict[str, dict[str, float]]:
    """Compare MUVERA vs baseline performance.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Model to evaluate
        muvera_top_k: Top-k for MUVERA
        reranking_top_k: Top-k for reranking phase
        output_file: Optional CSV file to save results
        
    Returns:
        Dictionary with baseline, muvera, and muvera+reranking results
    """
    # Run baseline
    print("\n" + "="*80)
    print("BASELINE EVALUATION (without MUVERA)")
    print("="*80)
    baseline_results = run_benchmark(
        dataset_name=dataset_name,
        model_name=model_name,
        use_muvera=False,
        use_reranking=False,
    )
    
    # Run MUVERA only
    print("\n" + "="*80)
    print("MUVERA EVALUATION (without reranking)")
    print("="*80)
    muvera_results = run_benchmark(
        dataset_name=dataset_name,
        model_name=model_name,
        use_muvera=True,
        use_reranking=False,
        muvera_top_k=muvera_top_k,
    )
    
    # Run MUVERA + Reranking
    print("\n" + "="*80)
    print("MUVERA + RERANKING EVALUATION")
    print("="*80)
    muvera_rerank_results = run_benchmark(
        dataset_name=dataset_name,
        model_name=model_name,
        use_muvera=True,
        use_reranking=True,
        muvera_top_k=muvera_top_k,
        reranking_top_k=reranking_top_k,
    )
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"\n{'Metric':<30} {'Baseline':<15} {'MUVERA':<15} {'MUVERA+Rerank':<15} {'Diff B->M (%)':<15} {'Diff B->MR (%)':<15}")
    print("-" * 120)
    
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
            diff_muvera_pct = ((muvera_val - baseline_val) / baseline_val) * 100
            diff_rerank_pct = ((muvera_rerank_val - baseline_val) / baseline_val) * 100
        else:
            diff_muvera_pct = 0
            diff_rerank_pct = 0
            
        print(f"{metric:<30} {baseline_val:<15.4f} {muvera_val:<15.4f} {muvera_rerank_val:<15.4f} {diff_muvera_pct:+14.2f}% {diff_rerank_pct:+14.2f}%")
    
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
                    writer.writerow(['MUVERA', key, value])
            
            for key, value in muvera_rerank_results.items():
                if isinstance(value, (int, float)):
                    writer.writerow(['MUVERA+Reranking', key, value])
        
        print(f"\nResults saved to: {output_path}")
    
    return {
        "baseline": baseline_results,
        "muvera": muvera_results,
        "muvera_reranking": muvera_rerank_results,
    }


if __name__ == "__main__":
    # Example usage: Compare Baseline vs MUVERA vs MUVERA+Reranking on SciFact-TR
    results = compare_muvera_vs_baseline(
        dataset_name="SciFact-TR",
        model_name="ozayezerceli/col-ettin-encoder-150M-TR",
        muvera_top_k=100,
        reranking_top_k=100,
        output_file="benchmarks/muvera_comparison_results.csv",
    )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print("\nResults summary:")
    print(f"  Baseline: {len([k for k, v in results['baseline'].items() if isinstance(v, (int, float))])} metrics")
    print(f"  MUVERA: {len([k for k, v in results['muvera'].items() if isinstance(v, (int, float))])} metrics")
    print(f"  MUVERA+Reranking: {len([k for k, v in results['muvera_reranking'].items() if isinstance(v, (int, float))])} metrics")

