 ========================================================================================================================
COMPREHENSIVE MUVERA COMPARISON
========================================================================================================================

Comparing:
  Methods: PLAID, MUVERA, MUVERA+Reranking
  Dimensions: 128D, 512D, 1024D
  K values: 100, 250, 500
  Datasets: SciFact-TR, NFCorpus-TR, ArguAna-TR, Scidocs-TR, FiQA-TR

This will take 2-3 hours...


################################################################################
# DATASET: SciFact-TR
################################################################################

================================================================================
Method: PLAID
Dataset: SciFact-TR
================================================================================

/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
modules.json: 100% 349/349 [00:00<00:00, 43.6kB/s]config_sentence_transformers.json: 100% 205/205 [00:00<00:00, 24.9kB/s]README.md:  19.7k/? [00:00<00:00, 2.27MB/s]sentence_bert_config.json: 100% 53.0/53.0 [00:00<00:00, 7.04kB/s]config.json:  1.52k/? [00:00<00:00, 193kB/s]configuration.py:  7.13k/? [00:00<00:00, 860kB/s]A new version of the following files was downloaded from https://huggingface.co/Alibaba-NLP/new-impl:
- configuration.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
modeling.py:  59.0k/? [00:00<00:00, 6.90MB/s]A new version of the following files was downloaded from https://huggingface.co/Alibaba-NLP/new-impl:
- modeling.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
model.safetensors: 100% 1.22G/1.22G [00:03<00:00, 535MB/s]tokenizer_config.json:  1.37k/? [00:00<00:00, 167kB/s]tokenizer.json: 100% 17.1M/17.1M [00:07<00:00, 2.41MB/s]special_tokens_map.json: 100% 964/964 [00:00<00:00, 132kB/s]config.json: 100% 296/296 [00:00<00:00, 39.2kB/s]The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
Loading dataset...
README.md: 100% 842/842 [00:00<00:00, 113kB/s]data/corpus-00000-of-00001.parquet: 100% 4.63M/4.63M [00:00<00:00, 6.62MB/s]data/queries-00000-of-00001.parquet: 100% 72.0k/72.0k [00:00<00:00, 137kB/s]Generating corpus split: 100% 5183/5183 [00:00<00:00, 70905.79 examples/s]Generating queries split: 100% 1109/1109 [00:00<00:00, 85256.02 examples/s]README.md: 100% 755/755 [00:00<00:00, 103kB/s]data/test-00000-of-00001.parquet: 100% 5.52k/5.52k [00:00<00:00, 8.93kB/s]Generating test split: 100% 339/339 [00:00<00:00, 26041.08 examples/s]Loaded 5183 documents and 1109 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 41/41 [00:24<00:00,  2.56it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 9/9 [00:00<00:00, 11.44it/s]
Indexing with PLAID...

Retrieving with PLAID...

Evaluating...
/usr/local/lib/python3.12/dist-packages/ranx/data_structures/report.py:202: SyntaxWarning: invalid escape sequence '\c'
  + """\\begin{table*}[ht]\n\centering\n\caption{\nOverall effectiveness of the models.\nThe best results are highlighted in boldface.\nSuperscripts denote significant differences in """
/usr/local/lib/python3.12/dist-packages/ranx/data_structures/report.py:204: SyntaxWarning: invalid escape sequence '\l'
  + """ with $p \le """
/usr/local/lib/python3.12/dist-packages/ranx/data_structures/report.py:211: SyntaxWarning: invalid escape sequence '\#'
  + "\n\\textbf{\#}"
/usr/local/lib/python3.12/dist-packages/ranx/data_structures/report.py:216: SyntaxWarning: invalid escape sequence '\m'
  + " \\\\ \n\midrule"
/usr/local/lib/python3.12/dist-packages/ranx/data_structures/report.py:246: SyntaxWarning: invalid escape sequence '\e'
  "\\bottomrule\n\end{tabular}\n}\n\label{tab:results}\n\end{table*}"
/usr/local/lib/python3.12/dist-packages/ranx/metrics/ndcg.py:72: NumbaTypeSafetyWarning: unsafe cast from uint64 to int64. Precision may be lost.
  scores[i] = _ndcg(qrels[i], run[i], k, rel_lvl, jarvelin)

================================================================================
Method: MUVERA
Dataset: SciFact-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 5183 documents and 1109 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 41/41 [00:24<00:00,  2.55it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 9/9 [00:00<00:00, 12.19it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: SciFact-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 5183 documents and 1109 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 41/41 [00:24<00:00,  2.55it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 9/9 [00:00<00:00, 12.00it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: SciFact-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 5183 documents and 1109 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 41/41 [00:24<00:00,  2.56it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 9/9 [00:00<00:00, 11.15it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: SciFact-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 5183 documents and 1109 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 41/41 [00:24<00:00,  2.56it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 9/9 [00:02<00:00,  2.22it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: SciFact-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 5183 documents and 1109 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 41/41 [00:23<00:00,  2.58it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 9/9 [00:00<00:00, 11.46it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: SciFact-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 5183 documents and 1109 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 41/41 [00:23<00:00,  2.58it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 9/9 [00:00<00:00, 11.42it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

################################################################################
# DATASET: NFCorpus-TR
################################################################################

================================================================================
Method: PLAID
Dataset: NFCorpus-TR
================================================================================

Loading dataset...
README.md:  1.21k/? [00:00<00:00, 144kB/s]corpus/corpus-00000-of-00001.parquet: 100% 3.19M/3.19M [00:01<00:00, 2.89MB/s]Generating corpus split: 100% 3633/3633 [00:00<00:00, 101385.97 examples/s]queries/queries-00000-of-00001.parquet: 100% 94.4k/94.4k [00:00<00:00, 180kB/s]Generating queries split: 100% 3237/3237 [00:00<00:00, 198935.68 examples/s]data/train-00000-of-00001.parquet: 100% 637k/637k [00:00<00:00, 1.23MB/s]data/dev-00000-of-00001.parquet: 100% 70.1k/70.1k [00:00<00:00, 136kB/s]data/test-00000-of-00001.parquet: 100% 76.3k/76.3k [00:00<00:00, 149kB/s]Generating train split: 100% 110575/110575 [00:00<00:00, 2676105.64 examples/s]Generating dev split: 100% 11385/11385 [00:00<00:00, 841254.89 examples/s]Generating test split: 100% 12334/12334 [00:00<00:00, 857478.67 examples/s]Loaded 3633 documents and 3237 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 29/29 [00:16<00:00,  2.60it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 26/26 [00:02<00:00, 12.46it/s]
Indexing with PLAID...

Retrieving with PLAID...

Evaluating...

================================================================================
Method: MUVERA
Dataset: NFCorpus-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 3633 documents and 3237 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 29/29 [00:16<00:00,  2.60it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 26/26 [00:02<00:00, 13.03it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: NFCorpus-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 3633 documents and 3237 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 29/29 [00:16<00:00,  2.60it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 26/26 [00:02<00:00, 12.97it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: NFCorpus-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 3633 documents and 3237 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 29/29 [00:16<00:00,  2.60it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 26/26 [00:02<00:00, 12.98it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: NFCorpus-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 3633 documents and 3237 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 29/29 [00:16<00:00,  2.60it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 26/26 [00:02<00:00, 12.96it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: NFCorpus-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 3633 documents and 3237 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 29/29 [00:16<00:00,  2.59it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 26/26 [00:04<00:00, 10.34it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: NFCorpus-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 3633 documents and 3237 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 29/29 [00:16<00:00,  2.59it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 26/26 [00:02<00:00, 12.95it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

################################################################################
# DATASET: ArguAna-TR
################################################################################

================================================================================
Method: PLAID
Dataset: ArguAna-TR
================================================================================

Loading dataset...
README.md:  1.01k/? [00:00<00:00, 114kB/s]corpus/corpus-00000-of-00001.parquet: 100% 5.12M/5.12M [00:01<00:00, 3.92MB/s]Generating corpus split: 100% 8674/8674 [00:00<00:00, 182196.66 examples/s]queries/queries-00000-of-00001.parquet: 100% 949k/949k [00:00<00:00, 1.82MB/s]Generating queries split: 100% 1406/1406 [00:00<00:00, 71716.69 examples/s]data/test-00000-of-00001.parquet: 100% 24.4k/24.4k [00:00<00:00, 48.1kB/s]Generating test split: 100% 1406/1406 [00:00<00:00, 110055.08 examples/s]Loaded 8674 documents and 1406 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 68/68 [00:31<00:00,  5.51it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 11/11 [00:01<00:00,  9.64it/s]
Indexing with PLAID...

Retrieving with PLAID...

Evaluating...

================================================================================
Method: MUVERA
Dataset: ArguAna-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 8674 documents and 1406 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 68/68 [00:31<00:00,  5.48it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 11/11 [00:01<00:00,  9.83it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: ArguAna-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 8674 documents and 1406 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 68/68 [00:31<00:00,  5.54it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 11/11 [00:01<00:00,  9.82it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: ArguAna-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 8674 documents and 1406 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 68/68 [00:31<00:00,  5.47it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 11/11 [00:01<00:00,  9.51it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: ArguAna-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 8674 documents and 1406 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 68/68 [00:31<00:00,  5.47it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 11/11 [00:01<00:00,  9.58it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: ArguAna-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 8674 documents and 1406 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 68/68 [00:34<00:00,  5.43it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 11/11 [00:01<00:00,  9.96it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: ArguAna-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 8674 documents and 1406 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 68/68 [00:31<00:00,  5.46it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 11/11 [00:01<00:00,  9.63it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

################################################################################
# DATASET: Scidocs-TR
################################################################################

================================================================================
Method: PLAID
Dataset: Scidocs-TR
================================================================================

Loading dataset...
README.md:  1.01k/? [00:00<00:00, 137kB/s]corpus/corpus-00000-of-00001.parquet: 100% 18.9M/18.9M [00:01<00:00, 14.0MB/s]Generating corpus split: 100% 25657/25657 [00:00<00:00, 202052.15 examples/s]queries/queries-00000-of-00001.parquet: 100% 97.8k/97.8k [00:00<00:00, 176kB/s]Generating queries split: 100% 1000/1000 [00:00<00:00, 82574.79 examples/s]data/test-00000-of-00001.parquet: 100% 1.32M/1.32M [00:00<00:00, 2.31MB/s]Generating test split: 100% 29928/29928 [00:00<00:00, 1071846.25 examples/s]Loaded 25657 documents and 1000 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 201/201 [01:47<00:00,  6.63it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 8/8 [00:00<00:00, 11.92it/s]
Indexing with PLAID...

Retrieving with PLAID...

Evaluating...

================================================================================
Method: MUVERA
Dataset: Scidocs-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 25657 documents and 1000 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 201/201 [01:48<00:00,  7.94it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 8/8 [00:00<00:00, 11.80it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: Scidocs-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 25657 documents and 1000 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 201/201 [01:47<00:00,  6.63it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 8/8 [00:00<00:00, 11.52it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: Scidocs-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 25657 documents and 1000 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 201/201 [01:47<00:00,  7.81it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 8/8 [00:00<00:00, 11.83it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: Scidocs-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 25657 documents and 1000 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 201/201 [01:47<00:00,  7.98it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 8/8 [00:00<00:00, 11.86it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: Scidocs-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 25657 documents and 1000 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 201/201 [01:47<00:00,  7.91it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 8/8 [00:00<00:00, 11.76it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: Scidocs-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 25657 documents and 1000 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 201/201 [01:47<00:00,  7.91it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 8/8 [00:00<00:00, 11.24it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

################################################################################
# DATASET: FiQA-TR
################################################################################

================================================================================
Method: PLAID
Dataset: FiQA-TR
================================================================================

Loading dataset...
README.md:  1.21k/? [00:00<00:00, 130kB/s]corpus/corpus-00000-of-00001.parquet: 100% 27.7M/27.7M [00:01<00:00, 21.2MB/s]Generating corpus split: 100% 57638/57638 [00:00<00:00, 280761.44 examples/s]queries/queries-00000-of-00001.parquet: 100% 364k/364k [00:00<00:00, 625kB/s]Generating queries split: 100% 6648/6648 [00:00<00:00, 384714.65 examples/s]data/train-00000-of-00001.parquet: 100% 159k/159k [00:00<00:00, 289kB/s]data/dev-00000-of-00001.parquet: 100% 15.6k/15.6k [00:00<00:00, 30.5kB/s]data/test-00000-of-00001.parquet: 100% 20.5k/20.5k [00:00<00:00, 39.5kB/s]Generating train split: 100% 14166/14166 [00:00<00:00, 730632.68 examples/s]Generating dev split: 100% 1238/1238 [00:00<00:00, 108560.31 examples/s]Generating test split: 100% 1706/1706 [00:00<00:00, 157787.00 examples/s]Loaded 57638 documents and 6648 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 451/451 [03:01<00:00,  8.83it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 52/52 [00:04<00:00, 11.58it/s]
Indexing with PLAID...

Retrieving with PLAID...

Evaluating...

================================================================================
Method: MUVERA
Dataset: FiQA-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 57638 documents and 6648 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 451/451 [03:01<00:00,  8.91it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 52/52 [00:04<00:00, 11.56it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: FiQA-TR
Encoding Dimension: 128D
================================================================================

Loading dataset...
Loaded 57638 documents and 6648 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 451/451 [03:03<00:00,  8.94it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 52/52 [00:04<00:00, 11.68it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: FiQA-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 57638 documents and 6648 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 451/451 [03:02<00:00,  8.92it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 52/52 [00:04<00:00, 11.81it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: FiQA-TR
Encoding Dimension: 512D
================================================================================

Loading dataset...
Loaded 57638 documents and 6648 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 451/451 [03:01<00:00,  8.96it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 52/52 [00:04<00:00, 11.63it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...

================================================================================
Method: MUVERA
Dataset: FiQA-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 57638 documents and 6648 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 451/451 [03:01<00:00, 11.39it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 52/52 [00:04<00:00, 11.57it/s]
Applying MUVERA encoding...

Retrieving with MUVERA...

Evaluating...

================================================================================
Method: MUVERA_RERANK
Dataset: FiQA-TR
Encoding Dimension: 1024D
================================================================================

Loading dataset...
Loaded 57638 documents and 6648 queries

Encoding documents with ColBERT...
Encoding documents (bs=128): 100% 451/451 [03:01<00:00,  8.95it/s]
Encoding queries with ColBERT...
Encoding queries (bs=128): 100% 52/52 [00:04<00:00, 11.68it/s]
Applying MUVERA encoding...

MUVERA retrieval + reranking...

Evaluating...


Results saved to: benchmarks/comprehensive_results/comprehensive_results.json
CSV saved to: benchmarks/comprehensive_results/comprehensive_results.csv

========================================================================================================================
COMPREHENSIVE COMPARISON SUMMARY
========================================================================================================================

========================================================================================================================
Dataset: ArguAna-TR
========================================================================================================================

Method               Dim      Query(ms)   NDCG@100  Recall@100  NDCG@250  Recall@250  NDCG@500  Recall@500  MAP
------------------------------------------------------------------------------------------------------------------------
PLAID                N/A      73.56       0.3116  0.8058 0.3195  0.8634 0.3233  0.8954 0.1787
MUVERA               128      0.72        0.2834  0.7290 0.2954  0.8165 0.3009  0.8620 0.1689
MUVERA+RERANK        128      34.33       0.3038  0.7859 0.3100  0.8300 0.3119  0.8457 0.1744
MUVERA               512      0.80        0.2267  0.6472 0.2425  0.7617 0.2498  0.8236 0.1268
MUVERA+RERANK        512      34.01       0.3111  0.7752 0.3165  0.8144 0.3175  0.8229 0.1846
MUVERA               1024     0.84        0.1709  0.4438 0.1874  0.5647 0.1954  0.6330 0.1042
MUVERA+RERANK        1024     26.09       0.2648  0.5982 0.2680  0.6209 0.2685  0.6252 0.1716

========================================================================================================================
Dataset: Scidocs-TR
========================================================================================================================

Method               Dim      Query(ms)   NDCG@100  Recall@100  NDCG@250  Recall@250  NDCG@500  Recall@500  MAP
------------------------------------------------------------------------------------------------------------------------
PLAID                N/A      76.29       0.1220  0.2214 0.1409  0.3037 0.1558  0.3790 0.0503
MUVERA               128      1.26        0.0893  0.1671 0.1075  0.2466 0.1217  0.3175 0.0348
MUVERA+RERANK        128      31.50       0.1267  0.2313 0.1403  0.2903 0.1472  0.3250 0.0509
MUVERA               512      1.37        0.0703  0.1297 0.0835  0.1872 0.0944  0.2417 0.0263
MUVERA+RERANK        512      26.37       0.0922  0.1541 0.1004  0.1898 0.1048  0.2114 0.0390
MUVERA               1024     1.59        0.0596  0.1218 0.0730  0.1802 0.0860  0.2459 0.0199
MUVERA+RERANK        1024     26.81       0.1005  0.1643 0.1086  0.1993 0.1119  0.2157 0.0433

========================================================================================================================
Dataset: NFCorpus-TR
========================================================================================================================

Method               Dim      Query(ms)   NDCG@100  Recall@100  NDCG@250  Recall@250  NDCG@500  Recall@500  MAP
------------------------------------------------------------------------------------------------------------------------
PLAID                N/A      75.52       0.1736  0.2085 0.1964  0.2871 0.2242  0.3771 0.0721
MUVERA               128      0.58        0.1317  0.1654 0.1544  0.2400 0.1832  0.3337 0.0499
MUVERA+RERANK        128      25.31       0.1726  0.2063 0.1933  0.2794 0.2144  0.3511 0.0728
MUVERA               512      0.63        0.0929  0.1236 0.1136  0.1933 0.1399  0.2795 0.0295
MUVERA+RERANK        512      35.99       0.1505  0.1679 0.1707  0.2380 0.1903  0.3062 0.0626
MUVERA               1024     0.68        0.1004  0.1260 0.1268  0.2164 0.1564  0.3178 0.0325
MUVERA+RERANK        1024     36.13       0.1454  0.1539 0.1629  0.2174 0.1812  0.2788 0.0576

========================================================================================================================
Dataset: FiQA-TR
========================================================================================================================

Method               Dim      Query(ms)   NDCG@100  Recall@100  NDCG@250  Recall@250  NDCG@500  Recall@500  MAP
------------------------------------------------------------------------------------------------------------------------
PLAID                N/A      73.06       0.1840  0.3811 0.2039  0.4928 0.2177  0.5851 0.1052
MUVERA               128      2.22        0.1385  0.2960 0.1594  0.4144 0.1727  0.5011 0.0749
MUVERA+RERANK        128      25.72       0.1804  0.3573 0.1956  0.4416 0.2008  0.4745 0.1064
MUVERA               512      2.49        0.0608  0.1419 0.0743  0.2173 0.0843  0.2843 0.0303
MUVERA+RERANK        512      26.02       0.1434  0.2492 0.1504  0.2912 0.1538  0.3125 0.0901
MUVERA               1024     2.66        0.1059  0.2496 0.1246  0.3595 0.1364  0.4358 0.0535
MUVERA+RERANK        1024     25.60       0.1477  0.2702 0.1548  0.3093 0.1571  0.3239 0.0886

========================================================================================================================
Dataset: SciFact-TR
========================================================================================================================

Method               Dim      Query(ms)   NDCG@100  Recall@100  NDCG@250  Recall@250  NDCG@500  Recall@500  MAP
------------------------------------------------------------------------------------------------------------------------
PLAID                N/A      78.32       0.4936  0.7739 0.5087  0.8803 0.5137  0.9220 0.4185
MUVERA               128      0.62        0.3330  0.6764 0.3472  0.7782 0.3588  0.8737 0.2426
MUVERA+RERANK        128      35.22       0.5253  0.8289 0.5315  0.8736 0.5326  0.8809 0.4412
MUVERA               512      0.70        0.1104  0.4328 0.1421  0.6599 0.1544  0.7606 0.0426
MUVERA+RERANK        512      35.76       0.4348  0.6611 0.4392  0.6918 0.4396  0.6951 0.3678
MUVERA               1024     0.74        0.2586  0.5528 0.2739  0.6607 0.2833  0.7386 0.1837
MUVERA+RERANK        1024     30.55       0.4239  0.6392 0.4282  0.6698 0.4298  0.6832 0.3628

========================================================================================================================
EVALUATION COMPLETE!
========================================================================================================================
