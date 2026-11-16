import math
import random
import time
import warnings

import torch

from src.training.loss import DPRLoss


def _prepare_examples(dataset, split):
    examples = []
    positives_by_query = {}
    empty_passages = 0
    empty_rows = 0
    for row in dataset:
        question, texts, ids = row["question"], row["text"], row["document_id"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Each {split} row must have a nonempty question.")
        if (len(texts) != len(ids) or any(not isinstance(text, str) for text in texts)
                or any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in ids)):
            raise ValueError(f"Each {split} row must have aligned passages and document IDs with string text.")
        passages = [(text, doc_id) for text, doc_id in zip(texts, ids) if text.strip()]
        empty_passages += len(texts) - len(passages)
        if not passages:
            empty_rows += 1
            continue
        examples.append((question, passages))
        # A question may have several relevant passages, including across duplicate rows.
        known_texts, known_ids = positives_by_query.setdefault(question, (set(), set()))
        known_texts.update(text for text, _ in passages)
        known_ids.update(doc_id for _, doc_id in passages)
    if empty_passages or empty_rows:
        warnings.warn(
            f"Excluded {empty_passages} empty passages and {empty_rows} {split} rows "
            "without a nonempty passage.",
            UserWarning,
            stacklevel=2,
        )
    return examples, positives_by_query


def _batches(examples, batch_size):
    batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    # Keep the last batch from becoming a single example with no in-batch negative.
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def _positive_mask(questions, selected, positives_by_query, device):
    return torch.tensor(
        [[text in positives_by_query[question][0] or doc_id in positives_by_query[question][1]
          for text, doc_id in selected] for question in questions],
        dtype=torch.bool,
        device=device,
    )


class Trainer:
    def __init__(self, dpr, cfg):
        if cfg.batch_size < 2:
            raise ValueError("Training batch_size must be at least 2 for in-batch negatives.")
        if cfg.epochs < 1:
            raise ValueError("Training epochs must be positive.")
        if cfg.learning_rate <= 0:
            raise ValueError("Training learning_rate must be positive.")
        self.dpr = dpr
        self.cfg = cfg
        self.criterion = DPRLoss()
        self.optimizer = torch.optim.AdamW(
            list(dpr.query_encoder.parameters()) + list(dpr.document_encoder.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.get("weight_decay", 0.01),
        )
        self.losses = []
        self.positive_scores = []
        self.negative_scores = []
        self.epoch_durations = []

    def train(self, dataset, on_epoch_end=None) -> list[float]:
        """Train on query/passage pairs from the training split only."""
        examples, positives_by_query = _prepare_examples(dataset, "training")
        if len(examples) < 2:
            raise ValueError("Training requires at least two query/passage examples.")

        self.dpr.retriever = None
        self.dpr.query_encoder.train()
        self.dpr.document_encoder.train()
        rng = random.Random(self.cfg.seed)
        self.losses = []
        self.positive_scores = []
        self.negative_scores = []
        self.epoch_durations = []

        for epoch in range(self.cfg.epochs):
            epoch_start = time.monotonic()
            # Callbacks can build a retrieval index; discard it before the next training epoch.
            self.dpr.retriever = None
            rng.shuffle(examples)
            batches = _batches(examples, self.cfg.batch_size)
            total_loss = 0.0
            positive_sum = negative_sum = 0.0
            positive_count = negative_count = 0
            gradient_sums = [0.0, 0.0]

            for batch in batches:
                questions = [question for question, _ in batch]
                selected = [rng.choice(passages) for _, passages in batch]
                passages = [text for text, _ in selected]
                # Other known positives are positives too, not in-batch negatives.
                positive_mask = _positive_mask(questions, selected, positives_by_query, self.dpr.device)

                self.optimizer.zero_grad()
                query_vectors = self.dpr.query_encoder(self.dpr.tokenize(questions))
                document_vectors = self.dpr.document_encoder(self.dpr.tokenize(passages))
                loss = self.criterion(query_vectors, document_vectors, positive_mask)
                loss.backward()
                encoders = (self.dpr.query_encoder, self.dpr.document_encoder)
                for i, encoder in enumerate(encoders):
                    # Report each encoder's combined gradient norm for this batch.
                    squared_norm = sum(
                        parameter.grad.detach().square().sum().item()
                        for parameter in encoder.parameters()
                        if parameter.grad is not None
                    )
                    gradient_sums[i] += math.sqrt(squared_norm)
                self.optimizer.step()
                self.dpr.encoders_trained = True
                scores = query_vectors.detach() @ document_vectors.detach().T
                positive_sum += scores[positive_mask].sum().item()
                negative_sum += scores[~positive_mask].sum().item()
                positive_count += positive_mask.sum().item()
                negative_count += (~positive_mask).sum().item()
                total_loss += loss.item() * len(batch)

            epoch_loss = total_loss / len(examples)
            self.losses.append(epoch_loss)
            positive_score = positive_sum / positive_count
            negative_score = negative_sum / negative_count if negative_count else None
            self.positive_scores.append(positive_score)
            self.negative_scores.append(negative_score)
            self.epoch_durations.append(time.monotonic() - epoch_start)
            negative_mean = f"{negative_score:.4f}" if negative_score is not None else "n/a"
            # Keep the epoch summary together so progress is easy to follow.
            print(f"Epoch {epoch + 1}/{self.cfg.epochs}: training loss {epoch_loss:.4f}, "
                  f"positive {positive_score:.4f}, negative {negative_mean}, "
                  f"query grad {gradient_sums[0] / len(batches):.4f}, "
                  f"document grad {gradient_sums[1] / len(batches):.4f}")
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1)

        return self.losses

    def prepare_validation(self, dataset):
        """Fix validation positives and batch membership for comparable epoch losses."""
        examples, positives_by_query = _prepare_examples(dataset, "validation")
        if len(examples) < 2:
            warnings.warn(
                "Validation loss requires at least two nonempty query/passage examples; it will be omitted.",
                UserWarning,
                stacklevel=2,
            )
            return []
        rng = random.Random(self.cfg.seed)
        result = []
        for batch in _batches(examples, self.cfg.batch_size):
            questions = [question for question, _ in batch]
            selected = [rng.choice(passages) for _, passages in batch]
            # Fixed validation selections make losses comparable across epochs.
            mask = _positive_mask(questions, selected, positives_by_query, self.dpr.device)
            result.append((questions, [text for text, _ in selected], mask))
        return result

    @torch.no_grad()
    def validation_loss(self, batches):
        if not batches:
            raise ValueError("Validation loss requires at least two query/passage examples.")
        encoders = (self.dpr.query_encoder, self.dpr.document_encoder)
        was_training = [encoder.training for encoder in encoders]
        try:
            # Disable dropout for validation without changing the encoders' prior modes.
            for encoder in encoders:
                encoder.eval()
            total_loss = 0.0
            count = 0
            for questions, passages, mask in batches:
                query_vectors = encoders[0](self.dpr.tokenize(questions))
                document_vectors = encoders[1](self.dpr.tokenize(passages))
                loss = self.criterion(query_vectors, document_vectors, mask)
                total_loss += loss.item() * len(questions)
                count += len(questions)
            return total_loss / count
        finally:
            for encoder, training in zip(encoders, was_training):
                encoder.train(training)
