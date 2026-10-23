"""PA-adapted Nebula using the same model/tokenizer/trainer as the experiments.

Documents must come from pa_execute_to_nebula_document, not the classical
serializer. This module evaluates one outer fold and uses its ORIGIN
training documents to fit a shared vocabulary. The source components and
MIT license are in model/_nebula/.
"""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

MODEL_CONFIG = dict(vocab_size=30000, maxlen=256, dModel=128, nHeads=4,
                    nLayers=2, dropout=0.2, numClasses=1, chunk_size=32)
TRAINER_CONFIG = dict(epochs=30, learning_rate=5e-4, batch_size=16,
                      early_stop_patience=7, out_classes=1,
                      accumulate_grad_batches=2, gradient_clip_val=1.0,
                      monitor_metric="val_f1", scheduler="cosine")


def train_and_eval(train_docs, train_labels, test_docs, test_labels, *,
                   origin_train_docs, fold=0, device="cpu", output_dir=None,
                   epochs=30):
    """Evaluate one paired fold. ``epochs`` may be reduced for a smoke check.

    The default 30-epoch budget, early stopping, last-epoch model and
    validation threshold grid match the stored experiment implementation.
    Input lists must refer to identical package order across scenarios.
    """
    import torch
    from torch.nn import BCEWithLogitsLoss
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_lightning import seed_everything
    from ._nebula.attention import TransformerEncoderChunks
    from ._nebula.tokenizer import NeurLuxPreprocessor
    from ._nebula.lit_utils import LitTrainerWrapper
    from .tfidf_classifier import metrics_at_threshold

    y = np.asarray(train_labels)
    if len(train_docs) != len(y) or len(origin_train_docs) != len(y):
        raise ValueError("Training labels, scenario and ORIGIN documents must be aligned.")
    if len(test_docs) != len(test_labels):
        raise ValueError("Test documents and labels must be aligned.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Nebula training requires both benign and malicious labels.")
    fit, val = train_test_split(np.arange(len(y)), test_size=0.15,
                                random_state=42 + fold, stratify=y)
    pre = NeurLuxPreprocessor(vocab_size=30000, max_length=256)
    pre.train([origin_train_docs[i] for i in fit])
    x = np.asarray(pre.pad_sequence(pre.preprocess_sequence(train_docs)), dtype=np.int64)
    xt = np.asarray(pre.pad_sequence(pre.preprocess_sequence(test_docs)), dtype=np.int64)
    torch_device = "cuda" if device in ("cuda", "gpu") else device
    trainer_device = "gpu" if torch_device.startswith("cuda") else torch_device
    seed_everything(42)
    model = TransformerEncoderChunks(**MODEL_CONFIG)
    yy = torch.tensor(y[fit].astype(np.float32))
    pos = float(yy.sum())
    loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([(len(yy) - pos) / max(1.0, pos)], device=torch_device))

    def loss(logits, labels):
        labels = torch.clamp(labels.to(logits.device), 0, 1)
        value = loss_fn(logits, labels)
        if value.ndim > 1:
            value = value.mean(dim=-1)
        return value.mean()

    tl = DataLoader(TensorDataset(torch.tensor(x[fit]), yy), batch_size=16, shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(x[val]), torch.tensor(y[val].astype(np.float32))),
                    batch_size=16, shuffle=False)
    with TemporaryDirectory(prefix="ladar_nebula_") as temporary:
        destination = Path(output_dir) if output_dir is not None else Path(temporary)
        destination.mkdir(parents=True, exist_ok=True)
        config = dict(TRAINER_CONFIG, epochs=epochs)
        trainer = LitTrainerWrapper(pytorch_model=model, name="Nebula",
                                    log_folder=str(destination), device=trainer_device,
                                    verbose=False, loss=loss, **config)
        # Keep checkpoints under the selected output directory, not the caller's cwd.
        for callback in trainer.trainer.callbacks:
            if callback.__class__.__name__ == "ModelCheckpoint":
                callback.dirpath = str(destination / "checkpoints")
        trainer.train_lit_model(tl, vl)
        model.eval().to(torch_device)

        @torch.no_grad()
        def predict(values):
            # Batch size/order are fixed to the experiment's inference procedure.
            logits = [model(torch.tensor(values[i:i + 64], device=torch_device)).reshape(-1).float().cpu()
                      for i in range(0, len(values), 64)]
            return torch.sigmoid(torch.cat(logits)).numpy()

        pv = predict(x[val])
        threshold, best = 0.5, 0.0
        for candidate in np.arange(0.1, 0.9, 0.05):
            score = f1_score(y[val], pv >= candidate, zero_division=0)
            if score > best:
                threshold, best = float(candidate), float(score)
        pt = predict(xt)
        if output_dir is not None:
            pre.dump_vocab(str(destination))
            torch.save(model.state_dict(), destination / "nebula_model.pt")
            np.savez_compressed(destination / "scores.npz", validation_indices=val,
                                validation_probabilities=pv, test_probabilities=pt,
                                threshold=threshold)
        return dict(model="Nebula", threshold=threshold,
                    **metrics_at_threshold(test_labels, pt, threshold))
