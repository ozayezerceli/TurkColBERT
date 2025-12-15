# TurkColBERT: Late-Interaction Retrieval for Turkish IR

This repository accompanies the TurkColBERT paper and releases the MUVERA-based late-interaction implementation, benchmarks, and reproducibility assets for Turkish information retrieval. The paper PDF lives in `docs/paper/2511.16528v1.pdf`.

![TurkColBERT](docs/figures/TurkColBERT.png)

## What’s inside
- `src/lateinteractionmodels/muvera.py`: MUVERA implementation (SimHash + sparse projection + aggregation).
- `benchmarks/`: runnable scripts for MUVERA, PLAID, and reranking comparisons plus generated results/plots.
- `benchmarks/BenchmarkResults/`: aggregated metrics, figures, and helper scripts to regenerate visualizations.
- `notebooks/`: fine-tuning and analysis notebooks used in the paper.
- `docs/`: paper PDFs, blog materials, and supporting figures.

```
lateinteractionmodels/
├── src/lateinteractionmodels/              # Python package
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

## Datasets
Evaluations cover five Turkish BEIR collections:
- SciFact-TR — `AbdulkaderSaoud/scifact-tr`
- ArguAna-TR — `trmteb/arguana-tr`
- Scidocs-TR — `trmteb/scidocs-tr`
- FiQA-TR — `trmteb/fiqa-tr`
- NFCorpus-TR — `trmteb/nfcorpus-tr`

## Models (Hugging Face)
- `newmindai/col-ettin-encoder-150M-TR`
- `newmindai/col-ettin-encoder-32M-TR`
- `newmindai/ColmmBERT-base-TR`
- `newmindai/ColmmBERT-small-TR`
- `newmindai/TurkEmbed4Retrieval`
- `ytu-ce-cosmos/turkish-e5-large`, `ytu-ce-cosmos/turkish-colbert`
- Hash variants: `newmindai/colbert-hash-{nano,pico,femto}-tr`

## Results & artifacts
- Metrics, CSV/JSON dumps, and plots live in `benchmarks/BenchmarkResults/`.
- Ready-made figures: `benchmarks/BenchmarkResults/Visualizations/`.
- Paper PDF: `docs/paper/2511.16528v1.pdf`.

## Notebooks & docs
- Training/analysis notebooks are under `notebooks/`.
- Blog draft and figures: `docs/blog/`.

## Citation
If you use this work, please cite the TurkColBERT paper:

Ezerceli, Ö., Bayraktar, R., ElHussieni, M., Terzioglu, F. B., Taş, S., Çelebi, Y., Asker, Y. (2025). TurkColBERT: A Benchmark of Dense and Late-Interaction Models for Turkish Information Retrieval. Available at [`docs/paper/2511.16528v1.pdf`](https://arxiv.org/abs/2511.16528).

You can watch [Full Local Demo of TurkColBERT](https://www.youtube.com/watch?v=bZamvZojMA0) by [Fahd Mirza](https://www.youtube.com/@fahdmirza). We appreciated his work and support to open source community!

## License
MIT License — see `LICENSE`.

## Contact
Questions or collaboration: oezerceli@newmind.ai (lead author) or open an issue.
