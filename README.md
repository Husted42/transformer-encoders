# 02456 Retrieval Project

Starter code for information retrieval project work in the 02456 Deep Learning course (Fall 2026).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py            # Run TF-IDF on FIQA
python main.py top_k=20   # Override a config value
```

This prints recall@k for the train, validation and test splits. Settings are managed with Hydra in `config/`.

## Training the custom DPR encoders

DPR trains separate query and passage instances of the project's custom Transformer from scratch. 
`bert-base-uncased` provides the tokenizer only; 

```bash
python train.py retriever=dpr training.output_dir=outputs/dpr/run-01
```

### DPR grid search

```bash
python grid_search.py --output outputs/dpr/grid-01
```

```bash
python grid_search.py --output outputs/dpr/grid-smoke --epochs 2 \
  --override training.batch_size=16 --override retriever.batch_size=16
```

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval using TF-IDF and a preliminary evaluation pipeline.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall metric.
```
