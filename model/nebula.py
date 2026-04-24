"""PA-adapted Nebula transformer stub (Section 5.3.2).

Nebula is a transformer-based dynamic malware detector [Trizna et al. 2024].
The architecture itself is input-agnostic; we adapt it to PA execute logs
by serializing each package into the token sequence produced by
:func:`model.pa_features.pa_execute_to_tokens`, building a byte-pair
vocabulary from the training fold, and training a reduced encoder
(vocab=30_000, sequence=256, d_model=128, 4 heads, 2 layers).

This release module provides:

- :class:`NebulaVocab`: token <-> id mapping with BPE-free whitespace
  tokenization (sufficient for the PA token alphabet).
- :class:`PANebulaDataset`: turns documents into fixed-length id tensors.
- :func:`train_and_eval`: a minimal training loop that reproduces the
  setting in the paper (lr=1e-3, batch=16, epochs=20).

The heavy lifting (BPE tokenizer, positional encodings, full transformer
stack) is deferred to PyTorch. The class is intentionally small so it
can be swapped out for the upstream Nebula package if installed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"


class NebulaVocab:
    def __init__(self, max_size: int = 30_000):
        self.max_size = max_size
        self.itos: List[str] = [PAD, UNK, CLS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def fit(self, documents: Sequence[str]) -> None:
        counter: Counter = Counter()
        for doc in documents:
            counter.update(doc.split())
        most_common = [tok for tok, _ in counter.most_common(self.max_size - len(self.itos))]
        for tok in most_common:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, doc: str, max_len: int = 256) -> List[int]:
        ids = [self.stoi[CLS]]
        for tok in doc.split()[: max_len - 1]:
            ids.append(self.stoi.get(tok, self.stoi[UNK]))
        if len(ids) < max_len:
            ids += [self.stoi[PAD]] * (max_len - len(ids))
        return ids[:max_len]


if _HAS_TORCH:

    class PANebulaDataset(Dataset):
        def __init__(self, docs: Sequence[str], labels: Sequence[int],
                     vocab: NebulaVocab, max_len: int = 256):
            self.ids = [vocab.encode(d, max_len) for d in docs]
            self.labels = list(labels)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            return (
                torch.tensor(self.ids[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long),
            )

    class Nebula(nn.Module):
        def __init__(self, vocab_size: int, d_model: int = 128,
                     nhead: int = 4, num_layers: int = 2, max_len: int = 256):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos = nn.Embedding(max_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers)
            self.cls_head = nn.Linear(d_model, 2)
            self._max_len = max_len

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            h = self.embed(x) + self.pos(positions)
            h = self.enc(h, src_key_padding_mask=(x == 0))
            return self.cls_head(h[:, 0])

    def train_and_eval(
        train_docs: Sequence[str],
        train_labels: Sequence[int],
        test_docs: Sequence[str],
        test_labels: Sequence[int],
        *,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
        max_len: int = 256,
        device: str = "cpu",
    ) -> dict:
        vocab = NebulaVocab(max_size=30_000)
        vocab.fit(train_docs)
        train_ds = PANebulaDataset(train_docs, train_labels, vocab, max_len)
        test_ds = PANebulaDataset(test_docs, test_labels, vocab, max_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = Nebula(len(vocab.itos), max_len=max_len).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        model.train()
        for _epoch in range(epochs):
            for x, y in train_loader:
                opt.zero_grad()
                logits = model(x.to(device))
                loss = crit(logits, y.to(device))
                loss.backward()
                opt.step()

        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
                y = y.numpy()
                tp += int(((pred == 1) & (y == 1)).sum())
                fp += int(((pred == 1) & (y == 0)).sum())
                fn += int(((pred == 0) & (y == 1)).sum())
                tn += int(((pred == 0) & (y == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        return {
            "model": "Nebula",
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

else:

    def train_and_eval(*args, **kwargs):
        raise SystemExit(
            "Nebula requires PyTorch. Install with: pip install torch"
        )
