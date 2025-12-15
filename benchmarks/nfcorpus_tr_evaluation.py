"""Evaluation script for NFCorpus-TR dataset."""

from __future__ import annotations

import csv
from pathlib import Path

from pylate import evaluation, indexes, models, retrieve
from datasets import load_dataset


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


def run(dataset_name: str = "NFCorpus-TR", model_name: str = "ozayezerceli/col-ettin-encoder-32M-TR") -> dict[str, float]:
    """Run benchmark on specified dataset."""
    
    # All dataset configurations
    dataset_configs = {
        "NFCorpus-TR": {
            "dataset_name": "NFCorpus-TR",
            "hf_corpus_path": "trmteb/nfcorpus-tr",
            "query_length": None,
            "use_names": True,
        },
        "SciFact-TR": {
            "dataset_name": "SciFact-TR",
            "hf_corpus_path": "AbdulkaderSaoud/scifact-tr",
            "hf_corpus_split": "corpus",
            "hf_queries_path": "AbdulkaderSaoud/scifact-tr",
            "hf_queries_split": "queries",
            "hf_qrels_path": "AbdulkaderSaoud/scifact-tr-qrels",
            "hf_qrels_split": "test",
            "query_length": None,
            "use_names": False,
        },
        "ArguAna-TR": {
            "dataset_name": "ArguAna-TR",
            "hf_corpus_path": "trmteb/arguana-tr",
            "query_length": None,
            "use_names": True,
        },
        "Scidocs-TR": {
            "dataset_name": "Scidocs-TR",
            "hf_corpus_path": "trmteb/scidocs-tr",
            "query_length": None,
            "use_names": True,
        },
        "FiQA-TR": {
            "dataset_name": "FiQA-TR",
            "hf_corpus_path": "trmteb/fiqa-tr",
            "query_length": None,
            "use_names": True,
        },
    }
    
    # Get configuration for the specified dataset
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    
    # Add common field mappings
    config.update({
        "corpus_id_field": "_id",
        "corpus_text_field": "text",
        "query_id_field": "_id",
        "query_text_field": "text",
        "qrels_query_field": "query-id",
        "qrels_doc_field": "corpus-id",
        "qrels_score_field": "score",
    })

    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name}")
    print(f"Dataset: {config['hf_corpus_path']}")
    print(f"{'='*80}\n")

    # Initialize model
    print(f"Model: {model_name}\n")
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=(
            int(config["query_length"]) if config["query_length"] else None
        ),
        trust_remote_code=True,
    )
    
    # Initialize index and retriever
    index = indexes.PLAID(override=True)
    retriever = retrieve.ColBERT(index=index)

    # Load dataset
    print("Loading dataset...")
    documents, queries, qrels = _load_hf_information_retrieval_dataset(config)
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    # Encode documents
    print("\nEncoding documents...")
    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=128,
        is_query=False,
        show_progress_bar=True,
    )

    # Index documents
    print("\nIndexing documents...")
    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )

    # Encode queries
    print("\nEncoding queries...")
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=128,
        is_query=True,
        show_progress_bar=True,
    )

    # Retrieve
    print("\nRetrieving results...")
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

    # Evaluate
    print("\nEvaluating...")
    results = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=[f"ndcg@{k}" for k in [1, 3, 5, 10, 100]]  # NDCG for different k values
        + ["map"]                                           # Mean Average Precision (MAP)
        + [f"recall@{k}" for k in [10, 100]]                # Recall at k
        + [f"precision@{k}" for k in [10, 100]],            # Precision at k
    )
    
    # Note: PyLate's "hits@k" metric appears to return average count, not hit rate
    # So we're excluding it to avoid confusion. The metrics above are standard IR metrics.

    return results


def run_all_datasets(model_name: str = "ozayezerceli/col-ettin-encoder-32M-TR") -> dict[str, dict[str, float]]:
    """Run benchmark on all available datasets."""
    
    datasets = ["NFCorpus-TR", "SciFact-TR", "ArguAna-TR", "Scidocs-TR", "FiQA-TR"]
    all_results = {}
    
    print("\n" + "="*80)
    print(f"RUNNING ALL DATASETS WITH MODEL: {model_name}")
    print("="*80)
    
    for dataset_name in datasets:
        try:
            print(f"\n{'#'*80}")
            print(f"# Processing: {dataset_name}")
            print(f"{'#'*80}")
            
            results = run(dataset_name=dataset_name, model_name=model_name)
            all_results[dataset_name] = results
            
            print(f"\n✓ {dataset_name} completed successfully")
            print(f"  NDCG@10: {results.get('ndcg@10', 0):.4f}")
            print(f"  Recall@100: {results.get('recall@100', 0):.4f}")
            print(f"  MAP: {results.get('map', 0):.4f}")
            
        except Exception as e:
            print(f"\n✗ ERROR with {dataset_name}: {e}")
            continue
    
    # Save all results to CSV
    output_file = Path(f"benchmarks/BenchmarkResults/All_Datasets_{model_name.split('/')[-1]}_Results.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Metric', 'Value'])
        
        for dataset_name, results in all_results.items():
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    writer.writerow([dataset_name, metric, value])
    
    print(f"\n\nAll results saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Dataset':<15} {'NDCG@10':<10} {'NDCG@100':<10} {'Recall@10':<12} {'Recall@100':<12} {'MAP':<10}")
    print("-" * 80)
    
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<15} "
              f"{results.get('ndcg@10', 0):<10.4f} "
              f"{results.get('ndcg@100', 0):<10.4f} "
              f"{results.get('recall@10', 0):<12.4f} "
              f"{results.get('recall@100', 0):<12.4f} "
              f"{results.get('map', 0):<10.4f}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    # Check if specific dataset is requested
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else "ozayezerceli/col-ettin-encoder-32M-TR"
        
        print(f"Running single dataset: {dataset_name}")
        metrics = run(dataset_name=dataset_name, model_name=model_name)
    else:
        # Run all datasets
        print("Running all datasets (use: python script.py <dataset_name> to run single dataset)")
        all_results = run_all_datasets()
        
        print("\n" + "="*80)
        print("ALL DATASETS COMPLETED!")
        print("="*80)
        
        # Exit early to avoid the single-dataset output
        sys.exit(0)
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS - {dataset_name}")
    print("="*80)
    
    # Print results in a formatted table
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    
    # Group metrics by type
    ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg')}
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith('recall')}
    precision_metrics = {k: v for k, v in metrics.items() if k.startswith('precision')}
    other_metrics = {k: v for k, v in metrics.items() if k not in ndcg_metrics and k not in recall_metrics and k not in precision_metrics}
    
    # Print in order
    print("\nNDCG (Normalized Discounted Cumulative Gain):")
    for metric in ['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@100']:
        if metric in ndcg_metrics:
            print(f"  {metric:<18} {ndcg_metrics[metric]:.4f}")
    
    print("\nRecall:")
    for metric in sorted(recall_metrics.keys()):
        print(f"  {metric:<18} {recall_metrics[metric]:.4f}")
    
    print("\nPrecision:")
    for metric in sorted(precision_metrics.keys()):
        print(f"  {metric:<18} {precision_metrics[metric]:.4f}")
    
    print("\nOther Metrics:")
    for metric, value in other_metrics.items():
        print(f"  {metric:<18} {value:.4f}")
    
    # Save to markdown file
    output_file = Path("benchmarks/BenchmarkResults/NFCorpus_TR_Benchmark_Results.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# NFCorpus-TR Benchmark Results\n\n")
        f.write("## Model\n")
        f.write("- ozayezerceli/col-ettin-encoder-32M-TR\n\n")
        f.write("## Dataset Information\n")
        f.write("- Source: trmteb/nfcorpus-tr\n")
        f.write("- Domain: Medical/Scientific information retrieval\n\n")
        f.write("## Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        for metric in ['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@100']:
            if metric in ndcg_metrics:
                f.write(f"| {metric} | {ndcg_metrics[metric]:.4f} |\n")
        
        for metric in sorted(recall_metrics.keys()):
            f.write(f"| {metric} | {recall_metrics[metric]:.4f} |\n")
        
        for metric in sorted(precision_metrics.keys()):
            f.write(f"| {metric} | {precision_metrics[metric]:.4f} |\n")
        
        for metric, value in other_metrics.items():
            f.write(f"| {metric} | {value:.4f} |\n")
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)

