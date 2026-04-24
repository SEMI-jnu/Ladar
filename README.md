# LADAR: Repairing Dynamic-Analysis Failures for Automated Malicious Python Package Detection

Public release artifact for the paper
*"Repairing Dynamic Analysis Failures for Improved Automated Malicious
Python Package Detection"*.

- **preprocessing/** -- LADAR's LLM-based APR modules: the single-stage
  install-phase repair and the two-stage execute-phase repair, plus a
  script that runs the execute-phase repair against a bundled case
  package.
- **model/** -- the downstream malicious-package detectors: TF-IDF
  classifiers and a PA-adapted Nebula transformer.
- **data/** -- three complete malicious case packages
  (`safepackage`, `equests`, `r4quests`) with original source, LADAR's
  patched source, and the pre-/post-repair OpenSSF Package Analysis (PA)
  logs for each, plus 30 benign PA logs and a predictions CSV produced
  by the included classifier.

## Directory layout

```
release/
  README.md
  requirements.txt
  preprocessing/
    install_apr.py                single-stage install-phase repair
    execute_apr.py                two-stage execute-phase repair
    run_apr_example.py            execute-phase repair demo
    __init__.py
  model/
    pa_features.py                PA JSON -> token sequence
    tfidf_classifier.py           LR / KNN / SVM / RF / XGB
    nebula.py                     PA-adapted Nebula transformer
    run_detection_example.py      interface check
    __init__.py
  data/
    README.md                     schema + cases
    example_package/
      safepackage-1337.0.2/
        original/                 pre-APR malicious source
        patched/                  post-LADAR source
      equests-2.27.1/
        original/
        patched/
      r4quests-2.28.1/
        original/
        patched/
    pa_logs/                      6 PA logs (3 cases x ORIGIN/LADAR)
    pa_logs_benign/               30 benign PA logs (Status=completed)
    model_output/                 produced by run_detection_example.py
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Reproduction: repair one malicious package

### 1. Inspect the failure

```bash
python -c "
import json, base64, pathlib
doc = json.loads(pathlib.Path('data/pa_logs/safepackage_origin_execute.json').read_text(encoding='utf-8'))
exe = doc['Analysis']['execute']
print('status =', exe['Status'])
print(base64.b64decode(exe['Stderr']).decode('utf-8', 'replace')[-1200:])
"
```

### 2. Run the two-stage execute-phase repair

```bash
export OPENAI_API_KEY=sk-...
python preprocessing/run_apr_example.py safepackage   # or: equests | r4quests
```

The script calls `gpt-4.1-mini` at `temperature=1.0`, runs Stage 1
(error analysis on the traceback), attaches caller/callee snippets when
Stage 1 reports them, runs Stage 2 (context-based repair), validates the
patch with `compile()`, and writes the result to
`data/example_package/<pkg>-<ver>/patched_runtime/`. It does not rerun
the OpenSSF Package Analysis sandbox; the bundled reference logs under
`data/pa_logs/*_ladar_execute.json` show the output of the full pipeline
when PA is also rerun.

### 3. Compare against the reference patch

```bash
diff -u \
  data/example_package/safepackage-1337.0.2/original/safepackage/Confused.py \
  data/example_package/safepackage-1337.0.2/patched/safepackage/Confused.py
```

### 4. Inspect what the post-repair PA log contains

```bash
python -c "
import json, pathlib
d = json.loads(pathlib.Path('data/pa_logs/safepackage_ladar_execute.json').read_text())
exe = d['Analysis']['execute']
print('status =', exe['Status'])
print('DNS      ->', [q['Hostname'] for e in (exe['DNS'] or [])[:3] for q in e.get('Queries', [])[:2]])
print('sockets  ->', [(s['Address'], s['Port']) for s in (exe['Sockets'] or [])[:3]])
print('commands ->', [' '.join(c['Command'])[:80] for c in (exe['Commands'] or [])[:3]])
print('files    ->', [(f['Path'], f.get('Read'), f.get('Write'))
                       for f in (exe['Files'] or []) if '/.netrc' in (f.get('Path') or '')][:3])
"
```

## Reproduction: detection pipeline

```bash
python model/run_detection_example.py
cat data/model_output/tfidf_lr_metrics.txt
```

This is an interface check for `model.pa_features` and
`model.tfidf_classifier`; it confirms both the feature extractor and the
classifier train/predict path on the bundled small corpus and writes
per-document probabilities to `data/model_output/tfidf_lr_predictions.csv`.
It is not a miniature replication of the paper's Table 8 numbers; the
paper's evaluation uses 19,001 packages with 10-fold stratified cross-
validation at the package level, which requires the full corpus. For
paper-accurate hyperparameters (TF-IDF `min_df=20`, 200 SVD components,
the five-classifier sweep) use `model.tfidf_classifier.train_and_eval`
with the full corpus.
