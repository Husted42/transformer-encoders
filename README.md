# 02456 Retrieval Project

Starter code for information retrieval project work in the 02456 Deep Learning course (Fall 2026).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py            # Run the configured retriever on FIQA
python main.py top_k=20   # Override a config value
```

This prints recall@k for the train, validation and test splits. Settings are managed with Hydra in `config/`.

### Dense passage retrieval

DPR tokenizes text, encodes passages into the vector database, and evaluates
retrieved documents through the same pipeline:

```bash
python main.py retriever=dpr
```

The default tokenizer is `bert-base-uncased`; it must be available in the
Hugging Face cache, or `model.tokenizer_name` can point to a local tokenizer
directory. Set `model.max_len` for truncation and, if needed, set
`retriever.batch_size` to control encoding memory usage. Vocabulary size and
padding ID are taken from the tokenizer. CLS pooling requires a tokenizer that
places a CLS token first. The current pipeline indexes documents from all
dataset splits, as the TF-IDF path does. Its reported recall divides the
number of relevant results by the smaller of `k` and the number of relevant
documents, which differs from conventional recall when more than `k` documents
are relevant.

To query from Python:

```python
from hydra import compose, initialize
from src.data_utils.corpus import Corpus
from src.retrievers.dpr import DPR

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="fiqa_experiment", overrides=["retriever=dpr"])
corpus = Corpus(dataset_name=cfg.dataset.name, base_dir="./retrieval_data")
retriever = DPR(cfg, documents=corpus.all_documents)
ids, scores = retriever.Search(["What is an index fund?"], k=5)
documents = retriever.Lookup(["What is an index fund?"], k=5)
```

### Training DPR

Train both encoders on FIQA's training split:

```bash
python train.py retriever=dpr training.epochs=3 training.output_dir=outputs/dpr/run-01
```

The trainer samples one relevant passage per query per epoch and uses the other
passages in the batch as negatives. Other known positives are identified by
document ID or passage text, including positives associated with the same question in another
training row; the loss rewards the total probability assigned to that positive
set. Unknown positives can still be treated as negatives. Empty passages and
rows without a nonempty passage are excluded with a warning, and mismatched
passage IDs are rejected. Validation and test queries are not used for training.

Configure `training.batch_size`, `training.learning_rate`, and `training.seed`
as needed. A final singleton batch is merged into the preceding batch, which
may therefore contain one extra example. Each epoch reports loss, mean
positive and negative scores, and both encoder gradient norms.

Training saves both encoder weights at the first and final epochs and every
`training.checkpoint_every_epochs` epochs (default 5), plus final weights and
the run configuration in `outputs/dpr/checkpoints`. Existing output directories
are not overwritten. Epoch checkpoints can be loaded using the same two
weight-path settings as final weights; they don't include optimizer state.

Validation also runs at the first and final epochs and every
`training.validation_every_epochs` epochs (default 5). It rebuilds the index
of document occurrences from the dataset splits each time and uses a fixed random sample of up to
`training.validation_max_queries` validation queries (default 128). It uses
the same document collection and recall calculation as `main.py`, so indexing
can take substantial time and the test split's documents are candidates, but
test queries are never used for validation.
For FIQA, these documents are the `main` split's ground-truth answer
occurrences, not the separate FIQA `corpus` collection.

Every epoch also computes validation loss on the same sampled validation
queries, using fixed batches and one fixed nonempty positive passage per
query. It uses the training loss and known-positive masking, with dropout
disabled and no parameter updates. Rows without nonempty passages are omitted
from this loss; if fewer than two remain, validation loss is omitted with a
warning. Training loss is averaged during updates with dropout enabled, so the
two losses are not measured under identical conditions.

The output directory contains `loss.png` (training and validation loss),
`training_scores.png` (mean positive and negative training scores and their
gap), `validation_recall.png` (recall at each scheduled validation), and
`training_history.csv` with the underlying epoch values. These files are
updated during the run; recall is only recorded at the first and final epochs
and every `training.validation_every_epochs` epochs. Score averages are
measured during training, not validation; if an epoch has no negative pairs,
its negative score and gap are omitted.

Evaluate the saved encoders by rebuilding the index:

```bash
python main.py retriever=dpr model.query_encoder_weights_path=outputs/dpr/run-01/query_encoder.pt model.document_encoder_weights_path=outputs/dpr/run-01/document_encoder.pt
```

Use the same model settings and tokenizer as in the saved `config.yaml`,
including any command-line overrides used during training. Without both
checkpoints, at least one encoder is randomly initialized and retrieval
rankings will not be meaningful. Training from scratch may require more
epochs and experimentation before retrieval quality is useful.

To diagnose training on one fixed batch of eight real FIQA pairs:

```powershell
python -m unittest discover -s tests\debugging -p "test_*.py" -v
```

The diagnostic checks scores and gradients, then prints positive/negative
scores and top-1 accuracy before and after overfitting. It uses mean pooling
and a learning rate of `0.0001` and skips if cached FIQA or the cached
`bert-base-uncased` tokenizer is unavailable. It does not write checkpoints;
overfitting this small batch does not demonstrate generalization.

For a reproducible pooling/learning-rate comparison on a restricted FIQA
train/validation sample (without test queries or test documents):

```powershell
python tests\debugging\compare_fiqa.py
```

Its results use the project's recall calculation over a smaller candidate pool
and are not directly comparable to the full evaluation.

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval and a preliminary evaluation pipeline.
train.py          Trains and saves DPR encoders.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall metric.
```
