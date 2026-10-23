# Nebula components

These files are reused from the Nebula checkout used in the LADAR experiments.
Upstream: https://github.com/dtrizna/nebula
Base revision: 525b779d9906986739e0d6a181e39216f5742435
License: MIT, reproduced in LICENSE.txt.

`attention.py`, `tokenizer.py` (upstream `models/neurlux/preprocessor.py`),
and `data_utils.py` are unchanged. `lit_utils.py` includes the PyTorch
Lightning compatibility changes used in the experiments. These components
are included directly to avoid the upstream package's PE-analysis dependencies.
