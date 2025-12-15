"""
Comprehensive MUVERA Comparison Script

Compares PLAID, MUVERA, and MUVERA+Reranking across:
- Multiple datasets (SciFact-TR, ArguAna-TR, Scidocs-TR, FiQA-TR)
- Multiple encoding dimensions (128D, 512D, 1024D)  
- Multiple K values (100, 250, 500)
"""

from __future__ import annotations

import time
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

from pylate import evaluation, indexes, models, retrieve, rank
from datasets import load_dataset

# Import custom MUVERA encoder
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))
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
        # trmteb format
        corpus_split = load_dataset(dataset_path, name="corpus", split="corpus")
        queries_split = load_dataset(dataset_path, name="queries", split="queries")
        qrels_split = load_dataset(dataset_path, name="default", split="test")
    else:
        # AbdulkaderSaoud format
        corpus_split = load_dataset(config["hf_corpus_path"], split=config["hf_corpus_split"])
        queries_split = load_dataset(config.get("hf_queries_path") or config["hf_corpus_path"], 
                                     split=config["hf_queries_split"])
        qrels_split = load_dataset(config["hf_qrels_path"], split=config["hf_qrels_split"])

    documents = [{"id": str(row["_id"]), "text": row["text"]} for row in corpus_split]
    queries = {str(row["_id"]): row["text"] for row in queries_split}
    
    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_split:
        query_id = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        try:
            score_value = float(row["score"])
        except (TypeError, ValueError):
            score_value = 1.0
        qrels.setdefault(query_id, {})[doc_id] = score_value

    return documents, queries, qrels


def get_muvera_config_for_dimension(encoding_dim: int) -> MuveraConfig:
    """Get MUVERA configuration for target encoding dimension."""
    if encoding_dim == 128:
        return MuveraConfig(
            dimension=128,
            num_simhash_projections=0,
            projection_type=ProjectionType.IDENTITY,
            projection_dimension=128,
            num_repetitions=1,
            fill_empty_partitions=True,
        )
    elif encoding_dim == 512:
        return MuveraConfig(
            dimension=128,
            num_simhash_projections=2,  # 4 partitions × 128D = 512D
            projection_type=ProjectionType.IDENTITY,
            projection_dimension=128,
            num_repetitions=1,
            fill_empty_partitions=True,
        )
    elif encoding_dim == 1024:
        return MuveraConfig(
            dimension=128,
            num_simhash_projections=3,  # 8 partitions × 128D = 1024D
            projection_type=ProjectionType.IDENTITY,
            projection_dimension=128,
            num_repetitions=1,
            fill_empty_partitions=True,
        )
    else:
        raise ValueError(f"Unsupported encoding dimension: {encoding_dim}")


def run_single_benchmark(
    method: str,  # "plaid", "muvera", "muvera_rerank"
    dataset_name: str,
    model_name: str,
    dataset_config: dict,
    k_values: List[int],
    encoding_dim: int = None,  # Only for MUVERA methods
) -> Dict[str, float]:
    """Run a single benchmark configuration."""
    
    print(f"\n{'='*80}")
    print(f"Method: {method.upper()}")
    print(f"Dataset: {dataset_name}")
    if encoding_dim:
        print(f"Encoding Dimension: {encoding_dim}D")
    print(f"{'='*80}\n")

    # Initialize model
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=None,
        trust_remote_code=True,
    )

    # Load dataset
    print("Loading dataset...")
    documents, queries, qrels = _load_hf_information_retrieval_dataset(dataset_config)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Encode documents
    print("\nEncoding documents with ColBERT...")
    documents_embeddings = model.encode(
        sentences=[doc["text"] for doc in documents],
        batch_size=128,
        is_query=False,
        show_progress_bar=True,
    )

    # Encode queries
    print("\nEncoding queries with ColBERT...")
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=128,
        is_query=True,
        show_progress_bar=True,
    )

    doc_ids = [doc["id"] for doc in documents]
    query_ids = list(queries.keys())
    max_k = max(k_values)

    if method == "plaid":
        # PLAID baseline
        print("\nIndexing with PLAID...")
        index = indexes.PLAID(override=True)
        retriever = retrieve.ColBERT(index=index)
        
        indexing_start = time.time()
        index.add_documents(documents_ids=doc_ids, documents_embeddings=documents_embeddings)
        indexing_time = time.time() - indexing_start
        
        print("\nRetrieving with PLAID...")
        query_start = time.time()
        scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=max_k)
        query_time = time.time() - query_start

    elif method == "muvera":
        # MUVERA only
        muvera_config = get_muvera_config_for_dimension(encoding_dim)
        encoder = MuveraEncoder(muvera_config)
        
        print("\nApplying MUVERA encoding...")
        indexing_start = time.time()
        doc_encodings = np.array([encoder.encode_document(emb) for emb in documents_embeddings])
        indexing_time = time.time() - indexing_start
        
        print("\nRetrieving with MUVERA...")
        query_start = time.time()
        query_encodings = np.array([encoder.encode_query(emb) for emb in queries_embeddings])
        similarities = batch_cosine_similarity(query_encodings, doc_encodings)
        
        scores = []
        for i in range(len(query_ids)):
            top_k_indices = np.argsort(-similarities[i])[:max_k]
            scores.append([
                {"id": doc_ids[idx], "score": float(similarities[i][idx])}
                for idx in top_k_indices
            ])
        query_time = time.time() - query_start

    elif method == "muvera_rerank":
        # MUVERA + Reranking
        muvera_config = get_muvera_config_for_dimension(encoding_dim)
        encoder = MuveraEncoder(muvera_config)
        
        print("\nApplying MUVERA encoding...")
        indexing_start = time.time()
        doc_encodings = np.array([encoder.encode_document(emb) for emb in documents_embeddings])
        indexing_time = time.time() - indexing_start
        
        print("\nMUVERA retrieval + reranking...")
        query_start = time.time()
        
        # Phase 1: MUVERA retrieval
        query_encodings = np.array([encoder.encode_query(emb) for emb in queries_embeddings])
        similarities = batch_cosine_similarity(query_encodings, doc_encodings)
        
        # Phase 2: Reranking
        batch_documents_ids = []
        batch_queries_embeddings = []
        batch_documents_embeddings = []
        
        for i in range(len(query_ids)):
            candidate_indices = np.argsort(-similarities[i])[:max_k]
            batch_documents_ids.append([doc_ids[idx] for idx in candidate_indices])
            batch_queries_embeddings.append(queries_embeddings[i])
            batch_documents_embeddings.append([documents_embeddings[idx] for idx in candidate_indices])
        
        scores = rank.rerank(
            documents_ids=batch_documents_ids,
            queries_embeddings=batch_queries_embeddings,
            documents_embeddings=batch_documents_embeddings,
        )
        query_time = time.time() - query_start

    # Evaluate
    print("\nEvaluating...")
    metrics = []
    for k in k_values:
        metrics.extend([f"ndcg@{k}", f"recall@{k}", f"precision@{k}"])
    metrics.append("map")
    
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=metrics,
    )

    # Add metadata
    results["method"] = method
    results["encoding_dim"] = encoding_dim if encoding_dim else "N/A"
    results["indexing_time_s"] = round(indexing_time, 2)
    results["query_time_ms"] = round((query_time / len(queries)) * 1000, 2)

    return results


def run_comprehensive_comparison(
    model_name: str = "ozayezerceli/col-ettin-encoder-32M-TR",
    encoding_dimensions: List[int] = [128, 512, 1024],
    k_values: List[int] = [100, 250, 500],
    output_dir: str = "benchmarks/comprehensive_results",
):
    """Run comprehensive comparison across all configurations."""
    
    # Dataset configurations
    dataset_configs = {
        "SciFact-TR": {
            "hf_corpus_path": "AbdulkaderSaoud/scifact-tr",
            "hf_corpus_split": "corpus",
            "hf_queries_path": "AbdulkaderSaoud/scifact-tr",
            "hf_queries_split": "queries",
            "hf_qrels_path": "AbdulkaderSaoud/scifact-tr-qrels",
            "hf_qrels_split": "test",
            "use_names": False,
        },
        "NFCorpus-TR": {
            "hf_corpus_path": "trmteb/nfcorpus-tr",
            "use_names": True,
        },
        "ArguAna-TR": {
            "hf_corpus_path": "trmteb/arguana-tr",
            "use_names": True,
        },
        "Scidocs-TR": {
            "hf_corpus_path": "trmteb/scidocs-tr",
            "use_names": True,
        },
        "FiQA-TR": {
            "hf_corpus_path": "trmteb/fiqa-tr",
            "use_names": True,
        },
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for dataset_name, dataset_config in dataset_configs.items():
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*80}")
        
        # Run PLAID baseline
        try:
            result = run_single_benchmark(
                method="plaid",
                dataset_name=dataset_name,
                model_name=model_name,
                dataset_config=dataset_config,
                k_values=k_values,
            )
            result["dataset"] = dataset_name
            all_results.append(result)
        except Exception as e:
            print(f"ERROR with PLAID on {dataset_name}: {e}")
        
        # Run MUVERA with different dimensions
        for dim in encoding_dimensions:
            try:
                result = run_single_benchmark(
                    method="muvera",
                    dataset_name=dataset_name,
                    model_name=model_name,
                    dataset_config=dataset_config,
                    k_values=k_values,
                    encoding_dim=dim,
                )
                result["dataset"] = dataset_name
                all_results.append(result)
            except Exception as e:
                print(f"ERROR with MUVERA {dim}D on {dataset_name}: {e}")
            
            try:
                result = run_single_benchmark(
                    method="muvera_rerank",
                    dataset_name=dataset_name,
                    model_name=model_name,
                    dataset_config=dataset_config,
                    k_values=k_values,
                    encoding_dim=dim,
                )
                result["dataset"] = dataset_name
                all_results.append(result)
            except Exception as e:
                print(f"ERROR with MUVERA+Rerank {dim}D on {dataset_name}: {e}")
    
    # Save results
    json_file = output_path / "comprehensive_results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {json_file}")
    
    # Save CSV
    csv_file = output_path / "comprehensive_results.csv"
    with open(csv_file, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"CSV saved to: {csv_file}")
    
    # Print summary table
    print_summary_table(all_results, k_values)
    
    return all_results


def print_summary_table(results: List[Dict], k_values: List[int]):
    """Print a summary comparison table."""
    
    print("\n" + "="*120)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*120)
    
    for dataset_name in set(r["dataset"] for r in results):
        print(f"\n{'='*120}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*120}")
        
        dataset_results = [r for r in results if r["dataset"] == dataset_name]
        
        # Print header
        print(f"\n{'Method':<20} {'Dim':<8} {'Query(ms)':<12}", end="")
        for k in k_values:
            print(f"NDCG@{k:<4} Recall@{k:<4}", end=" ")
        print("MAP")
        print("-" * 120)
        
        # Print results
        for r in dataset_results:
            method_label = r["method"].replace("_", "+").upper()
            dim_label = str(r["encoding_dim"]) if r["encoding_dim"] != "N/A" else "N/A"
            
            print(f"{method_label:<20} {dim_label:<8} {r['query_time_ms']:<12.2f}", end="")
            for k in k_values:
                ndcg = r.get(f"ndcg@{k}", 0)
                recall = r.get(f"recall@{k}", 0)
                print(f"{ndcg:6.4f} {recall:7.4f}", end=" ")
            print(f"{r.get('map', 0):.4f}")


if __name__ == "__main__":
    print("="*120)
    print("COMPREHENSIVE MUVERA COMPARISON")
    print("="*120)
    print("\nComparing:")
    print("  Methods: PLAID, MUVERA, MUVERA+Reranking")
    print("  Dimensions: 128D, 512D, 1024D")
    print("  K values: 100, 250, 500")
    print("  Datasets: SciFact-TR, NFCorpus-TR, ArguAna-TR, Scidocs-TR, FiQA-TR")
    print("\nThis will take 2-3 hours...\n")
    
    results = run_comprehensive_comparison(
        model_name="ozayezerceli/col-ettin-encoder-32M-TR",
        encoding_dimensions=[128, 512, 1024],
        k_values=[100, 250, 500],
    )
    
    print("\n" + "="*120)
    print("EVALUATION COMPLETE!")
    print("="*120)

