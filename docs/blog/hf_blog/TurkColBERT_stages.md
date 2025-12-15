## Building TurkColBERT: A Three-Stage Adaptation Pipeline

<img src="turk_colbert_figs\3_stages.svg" alt="Diagram alt text" width="700"/>

### Stage 1 — Semantic Fine-Tuning for Turkish

In this first stage, we strengthen the semantic understanding of Turkish across several pretrained encoder families before moving to retrieval-specific adaptation.

Objective: Improve Turkish sentence-level comprehension through two key supervised tasks — Natural Language Inference (All-NLI-TR) and Semantic Textual Similarity (STSb-TR).

Model families: We fine-tune mmBERT (base & small), Ettin, and BERT-Hash variants (nano, pico, femto) using the Sentence Transformers framework.

**Training setup:**

- Architecture: Siamese/triplet networks with mean pooling for fixed-size embeddings
- Loss functions: MultipleNegativesRankingLoss + MatryoshkaLoss for multi-level embedding spaces
- Optimization: Batch size 8, learning rate 3e−6 (NLI) / 2e−5 (STS), mixed precision (BF16) on NVIDIA A100 GPUs
- Monitoring: TripletEvaluator and EmbeddingSimilarityEvaluator for continuous performance tracking

**Process:**

- Step 1 – NLI training: Focus on sentence-level entailment and contradiction using All-NLI-TR triplets.
- Step 2 – STS training: Refine semantic similarity across graded sentence pairs in STSb-TR.

Results:

- mmBERT-small achieves 93% triplet accuracy (NLI) and 0.78 Spearman correlation (STS)
- Represents +25% semantic performance gain over pretrained checkpoints.

Impact: These Turkish-aware encoders offer a robust semantic foundation for later ColBERT-style retrieval fine-tuning, enabling higher precision in search, QA, and document understanding tasks in Turkish.


### Stage 2 — Late-Interaction Adaptation with PyLate

Next, we transform Turkish-aware encoders into ColBERT-style retrievers using PyLate and MS MARCO-TR triplets. This enables token-level matching for high-precision retrieval.

- **Goal:** Equip models with late-interaction retrieval capability using MaxSim scoring and contrastive triplet loss (margin = 0.2).
- **Data:** Turkish query–positive–negative triplets from MS MARCO-TR.

**Models trained:**

- mmBERT (base, small) — multilingual encoders for Turkish.
- Ettin (150M, 32M) — compact cross-lingual encoders.
- BERT-Hash (nano, pico, femto) — hash-projected lightweight variants.
- Dense baselines — XLM-RoBERTa, GTE for comparison.

**Setup:**

- Framework: PyLate ColBERT module with per-token embeddings and ColBERTCollator batching.
- Optimization: mixed precision on A100 GPUs, monitored via Weights & Biases.

**Outcome:**

- Family of TurkColBERT models (0.2M–310M parameters) achieving a strong efficiency–accuracy balance.
- Provides the base for large-scale Turkish semantic search and QA systems.

<!-- ```python
from pylate import ColBERT, Trainer

model = ColBERT.from_pretrained("")
trainer = Trainer(
    model=model,
    train_dataset="parsak/msmarco-tr",
    max_length=256,
)
trainer.train()
``` -->


### Stage 3 — Scalable Deployment with MUVERA

Finally, we integrate the late-interaction models with MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encoding):

- Compresses token-level vectors into compact sketches using LSH and AMS.
- Supports different embedding sizes (e.g., 128–2048D).
- Keeps retrieval 3.3× faster than PLAID on average, while slightly improving mAP thanks to MUVERA + Rerank.

This stage turns TurkColBERT from a research prototype into a production-ready retrieval stack.


### Final Stage — Evaluation on Turkish BEIR Benchmarks

#### Campaign 1 — Model Comparison Across Architectures

In the final stage, we perform a comprehensive zero-shot evaluation using the BEIR framework to assess retrieval quality and efficiency across Turkish domains.

**Goal:** Benchmark all TurkColBERT, mmBERT, Ettin, and BERT-Hash models (0.2M–600M params) under identical conditions.

**Datasets:** Five Turkish BEIR tasks — SciFact-TR, ArguAna-TR, Scidocs-TR, FiQA-TR, and NFCorpus-TR — covering science, finance, and health.

**Metrics:** Key retrieval indicators (Recall@10, Precision@10, mAP) plus computational measures like query latency and indexing time.

**Outcome:** Delivers a unified performance view, showing how late-interaction retrievers outperform dense bi-encoders while maintaining strong efficiency–accuracy balance for Turkish IR applications.


#### Campaign 2 — MUVERA Indexing Ablation Study

The second evaluation examines the quality–efficiency trade-offs of MUVERA-based indexing for Turkish retrieval.

**Models tested:** Four top late-interaction retrievers — TurkEmbed4Retrieval, col-ettin-encoder-32M-TR, ColmmBERT-base-TR, and ColmmBERT-small-TR.

**Configurations:**

- PLAID – high-fidelity baseline with exact MaxSim scoring.
- MUVERA – approximate search using fixed-dimensional encodings (128D–2048D).
- MUVERA + Reranking – re-scores top-K candidates via exact ColBERT MaxSim.

**Metrics:** Retrieval quality (NDCG@100, Recall, Precision, mAP) and efficiency (indexing time, query latency).

**Findings:** Results show how embedding dimensionality shapes the balance between accuracy and speed; e.g., MUVERA delivers near-ColBERT quality with markedly lower latency.

This ablation helps practitioners tune Turkish IR systems for their target latency-vs-accuracy requirements.






