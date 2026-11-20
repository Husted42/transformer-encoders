# 02456 Retrieval Project

Custom Transformer dual encoders and lexical baselines for passage retrieval.
Python 3.10+ is required.

```bash
python -m pip install -r requirements.txt
```

## Data and evaluation

FIQA now uses the BEIR FiQA corpus, queries and train/validation/test qrels
from `BeIR/fiqa` and `BeIR/fiqa-qrels`, at pinned dataset revisions. A single
57,638-document collection is shared across splits. Passage titles, when
present, are joined to their text. The first run downloads and caches the
benchmark under `retrieval_data/fiqa-retrieval-v1`; the older
`retrieval_data/fiqa` answer-based cache is not modified or reused. Results
from the old cache and the BEIR benchmark are not directly comparable.

SQuAD contexts are deduplicated and shared by questions with identical
contexts; its retrieval cache is `retrieval_data/squad-retrieval-v1`. SQuAD
has no test split in this workflow.

Compare lexical methods on the same corpus and validation qrels:

```bash
python evaluate.py --method tfidf --split validation --k 10
python evaluate.py --method bm25 --split validation --k 10
# Equivalent Hydra lexical entry point (validation only):
python main.py retriever=bm25 top_k=10
```

Evaluation reports recall@k (fraction of all known relevant passages
retrieved), precision@k (hits divided by k), and MRR@k (reciprocal rank of
the first hit). Relevance labels are incomplete; unlabelled passages are not
necessarily irrelevant. Use validation for model selection and invoke
`--split test` separately only for final evaluation.

## Training DPR

The model has separate custom query and document Transformers initialized from
scratch. `bert-base-uncased` supplies a tokenizer, **not pretrained weights**.
The default tokenizer revision is pinned in the model configuration.
The trainer uses one seeded sampled positive per query, other in-batch
documents as negatives, and masks all known positives in that batch using
training qrels. Hard negatives are not currently mined.

```bash
python train.py retriever=dpr training.output_dir=outputs/dpr/run-01
# Example experiment, not a guaranteed improvement:
python train.py retriever=dpr model.pooling_type=mean \
  training.output_dir=outputs/dpr/mean-01
```

Each run saves the resolved configuration, epoch and best checkpoints,
validation recall and MRR, loss and score history. The `best` checkpoint is
selected by validation recall@k. Choose a new output directory for each run.
Use `training.validation_max_queries` and
`training.validation_every_epochs` to explicitly control validation cost;
the sampled query count is recorded in checkpoint metadata. Every validation
index still contains the full corpus. Encoding the full BEIR corpus at each
validation can be slow, especially on CPU.

```bash
python evaluate.py --method dpr --checkpoint outputs/dpr/run-01/best \
  --split validation --k 10
python -m src.retrievers.build_index --checkpoint outputs/dpr/run-01/best \
  --output outputs/dpr/run-01/index
python -m src.retrievers.search "financial question" \
  --checkpoint outputs/dpr/run-01/best --index outputs/dpr/run-01/index
```

The persisted FAISS index is tied to the checkpoint's encoder hashes and
settings; rebuild it whenever the encoders change. New checkpoints also record
the corpus version and encoder hashes. Old answer-based checkpoints are not
compatible with the new benchmark evaluation.

## Experiments and checks

```bash
python grid_search.py --output outputs/dpr/grid-01
python -m unittest tests.test_retrieval_contract tests.test_training \
  tests.test_dpr tests.test_vector_database tests.test_faiss_pipeline
python -m tests.test_encoder
python -m unittest tests.debugging.test_fiqa_overfit
```

The last check requires the cached BEIR dataset and tokenizer and fits eight
fixed training pairs; it tests learning mechanics, **not** generalization.
For meaningful comparisons, record the seed, pooling, learning rate, batch
size, passage length, corpus version, validation query count and checkpoint
selection metric. Tune on validation rather than test.
