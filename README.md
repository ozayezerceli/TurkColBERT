# TurkColBERT: Late-Interaction Retrieval for Turkish IR

[![arXiv](https://img.shields.io/badge/arXiv-2511.16528-B31B1B.svg)](https://arxiv.org/abs/2511.16528) [![Hugging Face Blog](https://img.shields.io/badge/Blog-Hugging%20Face-ffbd2e.svg)](https://huggingface.co/blog/nmmursit/late-interaction-models) [![🤗 Models](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/collections/newmindai/turkcolbert-turkish-late-interaction-models) [![YouTube](https://img.shields.io/badge/YouTube-Demo-red.svg)](https://www.youtube.com/watch?v=bZamvZojMA0)


This repository accompanies the TurkColBERT paper and releases the MUVERA-based late-interaction implementation, benchmarks, and reproducibility assets for Turkish information retrieval.

![TurkColBERT](docs/figures/TurkColBERT.png)

## Key Contributions

- **TurkColBERT**: First systematic comparison of dense bi-encoders and late-interaction models for Turkish IR
- **Semantic Adaptation**: Fine-tuned multilingual encoders to Turkish using NLI + STS tasks, then adapted to ColBERT-style retrievers
- **Parameter Efficiency**: Ultra-compact BERT-Hash variants retain strong performance with as few as 0.2–1M parameters
- **Production Ready**: MUVERA + Rerank delivers 3.3× speedup over PLAID with +1–2% mAP gain for scalable Turkish IR

## What’s inside
- `src/muvera.py`: MUVERA implementation (SimHash + sparse projection + aggregation).
- `benchmarks/`: runnable scripts for MUVERA, PLAID, and reranking comparisons plus generated results/plots.
- `benchmarks/BenchmarkResults/`: aggregated metrics, figures, and helper scripts to regenerate visualizations.
- `notebooks/`: fine-tuning and analysis notebooks used in the paper.
- `docs/`: paper PDFs, blog materials, and supporting figures.

```
TurkColBERT/
├── src/              # Python package
│   └── muvera.py
├── benchmarks/                             # Experiments & outputs
│   ├── benchmark_with_muvera.py
│   ├── benchmark_with_custom_muvera.py
│   ├── muvera_comprehensive_benchmark.py
│   ├── comprehensive_muvera_comparison.py
│   ├── nfcorpus_tr_evaluation.py
│   ├── BenchmarkResults/
│   │   ├── Visualizations/
│   │   └── *.md / *.json / *.csv
│   └── EvaluationScripts/                  # Jupyter eval notebooks
├── notebooks/                              # Training/analysis notebooks
├── docs/                                   # Paper + blog assets
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Minimum requirements
- Python 3.9+
- CUDA GPU recommended (tested on A100/24GB; CPU works for small subsets)
- Hugging Face access for datasets/models (`huggingface-cli login` if needed)

## Running benchmarks
- Comprehensive MUVERA sweep (all datasets × dimensions):
  ```bash
  python benchmarks/muvera_comprehensive_benchmark.py
  # results → benchmarks/BenchmarkResults/muvera_results/
  ```
- Method comparison (PLAID vs MUVERA vs MUVERA+rerank):
  ```bash
  python benchmarks/comprehensive_muvera_comparison.py
  # results → benchmarks/comprehensive_results/
  ```
- Single-dataset, configurable MUVERA run:
  ```bash
  python benchmarks/benchmark_with_custom_muvera.py
  ```
- Baseline vs MUVERA vs MUVERA+rerank (Voyager/PLAID):
  ```bash
  python benchmarks/benchmark_with_muvera.py
  ```

## Using MUVERA directly
```python
from pylate import models
from lateinteractionmodels.muvera import (
    MuveraEncoder, MuveraConfig, ProjectionType, batch_cosine_similarity
)
import numpy as np

model = models.ColBERT(
    model_name_or_path="newmindai/col-ettin-encoder-32M-TR",
    device="cuda",
    trust_remote_code=True,
)

config = MuveraConfig(
    dimension=128,
    num_simhash_projections=0,      # 128D
    projection_type=ProjectionType.IDENTITY,
    projection_dimension=128,
)
encoder = MuveraEncoder(config)

doc_embeddings = model.encode(["birinci belge", "ikinci belge"], is_query=False)
docs = np.array([encoder.encode_document(e) for e in doc_embeddings])

query_emb = model.encode(["arama sorgusu"], is_query=True)[0]
query = encoder.encode_query(query_emb)

sims = batch_cosine_similarity(np.array([query]), docs)
top_k = np.argsort(-sims[0])[:5]
```

Evaluations cover five Turkish BEIR benchmark datasets spanning diverse domains:

| Dataset | Domain | # Queries | # Corpus | Task Type |
|---------|--------|-----------|----------|-----------|
| [SciFact-TR](https://huggingface.co/datasets/AbdulkaderSaoud/scifact-tr) | Scientific Claims | 1,110 | 5,180 | Fact Checking |
| [Arguana-TR](https://huggingface.co/datasets/trmteb/arguana-tr) | Argument Mining | 500 | 10,000 | Argument Retrieval |
| [FiQA-TR](https://huggingface.co/datasets/selmanbaysan/fiqa-tr) | Financial | 600 | 50,000 | Answer Retrieval |
| [Scidocs-TR](https://huggingface.co/datasets/trmteb/scidocs-tr) | Scientific | 1,000 | 25,000 | Citation Prediction |
| [NFCorpus-TR](https://huggingface.co/datasets/trmteb/nfcorpus-tr) | Nutrition | 3,240 | 3,630 | Document Retrieval |


## Models (Hugging Face)
 Model | Parameters (M) | Type |
|-------|---------------|------|
| **Dense Bi-Encoder Models** | | |
| TurkEmbed4Retrieval | 300 | Dense |
| turkish-e5-large | 600 | Dense |
| **Late-Interaction Models (Token-Level Matching)** | | |
| turkish-colbert | 100 | Late-interaction |
| ColumBERT-small-TR | 140 | Late-interaction |
| ColumBERT-base-TR | 310 | Late-interaction |
| col-ettin-150M-TR | 150 | Late-interaction |
| col-ettin-32M-TR | 32 | Late-interaction |
| mxbai-edge-colbert-v0-32m-tr | 32 | Late-interaction |
| mxbai-edge-colbert-v0-17m-tr | 17 | Late-interaction |
| **Ultra-Compact Models (BERT-Hash)** | | |
| colbert-hash-nano-tr | 1.0 | Hash-based |
| colbert-hash-pico-tr | 0.4 | Hash-based |
| colbert-hash-femto-tr | 0.2 | Hash-based |

<img width="4608" height="1036" alt="image" src="https://github.com/user-attachments/assets/cad30415-ca0b-4e58-b8e9-61190dcaf574" />

## Three-Stage Training Pipeline

TurkColBERT uses a systematic three-stage adaptation pipeline:

1. **Stage 1 — Semantic Fine-Tuning**: Strengthen Turkish sentence comprehension through NLI + STS tasks using Sentence Transformers
2. **Stage 2 — Late-Interaction Adaptation**: Transform encoders into ColBERT-style retrievers using PyLate and MS MARCO-TR triplets
3. **Stage 3 — Scalable Deployment**: Integrate MUVERA for efficient indexing with 3.3× speedup over PLAID

<img width="1184" height="453" alt="image" src="https://github.com/user-attachments/assets/718d1595-ed54-4bf7-b140-3c6e6e08faf8" />

## Key Results

- **Late-interaction models consistently outperform dense baselines** across all Turkish BEIR benchmarks
- **ColumBERT-base-TR achieves highest mAP** on 4 out of 5 datasets with strong efficiency balance
- **Ultra-compact BERT-Hash variants** (0.2–1M parameters) retain 70%+ of larger model performance
- **MUVERA + Rerank delivers 3.3× speedup** over PLAID with +1–2% mAP gain for production deployment


## Results & artifacts
- Metrics, CSV/JSON dumps, and plots live in `benchmarks/BenchmarkResults/`.
- Ready-made figures: `benchmarks/BenchmarkResults/Visualizations/`.
- Paper PDF: `docs/paper/2511.16528v1.pdf`.

## Notebooks & docs
- Training/analysis notebooks are under `notebooks/`.
- Blog draft and figures: `docs/blog/`.

## Citation
If you use TurkColBERT in your research, please cite our paper:

**TurkColBERT: A Benchmark of Dense and Late-Interaction Models for Turkish Information Retrieval** has been ACCEPTED at ACLing-2025 and will be published in Procedia Computer Science by ELSEVIER. The preprint is available on arXiv: [2511.16528](https://arxiv.org/abs/2511.16528).
```bibtex
@inproceedings{ezerceli2025turkcolbert,
  title={TurkColBERT: A Benchmark of Dense and Late-Interaction Models for Turkish Information Retrieval},
  author={Ezerceli, {\"O}zay and Bayraktar, Reyhan and ElHussieni, Mahmoud and Terzioglu, Fatma Bet{\"u}l and Ta{\c{s}}, Selva and {\c{C}}elebi, Yusuf and Asker, Ya{\u{g}}{\i}z},
  booktitle={Proceedings of ACLing-2025},
  year={2025},
  publisher={Elsevier},
  series={Procedia Computer Science}
}
```
You can watch [Full Local Demo of TurkColBERT](https://www.youtube.com/watch?v=bZamvZojMA0) by [Fahd Mirza](https://www.youtube.com/@fahdmirza). We appreciated his work and support to open source community!

## License
MIT License — see `LICENSE`.

## Contact
Questions or collaboration: oezerceli@newmind.ai or open an issue.
