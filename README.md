# LADAR: Repairing Dynamic-Analysis Failures for Automated Malicious Python Package Detection

Public release artifact for the paper
*"Repairing Dynamic Analysis Failures for Improved Automated Malicious
Python Package Detection"*.

- **preprocessing/** -- LADAR's LLM-based APR modules: the single-stage
  install-phase repair and the two-stage execute-phase repair, plus a
  script that runs the execute-phase repair against a bundled case
  package.
- **model/** -- the downstream malicious-package detectors: TF-IDF
  classifiers and the PA-adapted Nebula implementation used in the experiments.
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
    pa_nebula.py                  experimental PA-to-Nebula normalization
    _nebula/                     reused Nebula components and MIT license
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

This example demonstrates one execute-phase patch attempt. It does not
include literal placeholder processing, retries after a failed syntax check,
or the three-round PA rerun loop. The install-phase module provides error
classification and prompt construction, not an installation runner. PA
integration is required to reproduce analysis-completion rates; a successful
syntax check alone is not a successful LADAR repair.

### Prompt size limit (install phase)

The install-phase prompt embeds the affected file up to
`install_apr.PROMPT_FILE_CHAR_LIMIT` (12,000 characters), matching the
configuration used for the reported results. In the full experimental
pipeline, large base64 and binary literals are replaced by placeholders
before this limit is applied, then restored after repair. That preprocessing
step is not included in this example.

Because the template asks for the complete corrected file, a file longer
than the cap cannot be fully reproduced by the model.
`TargetedPromptGenerator.generate_prompt` returns a `(prompt, truncated)`
pair so callers can skip such files rather than accept a partial patch. In
the reported runs the cap bound for 72 of 1,985 malicious install patches
(3.6%); none of those produced a completed execution log.

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
paper's detection evaluation uses 14,422 packages from the larger collection
of 19,001 packages. It uses 10-fold stratified cross-validation at the
package level and requires the full evaluation corpus.

For classical detection, convert each selected execution log with
`model.pa_features.pa_execute_to_document`, then pass equally ordered
document lists for ORIGIN and each repair scenario, together with the labels,
to `model.tfidf_classifier.evaluate_scenarios`. Keep empty ORIGIN documents.
The caller must apply the population and representative-log selection rules
of manuscript Section 5.4 before this step. The evaluator shares the fold
splits, fits the vocabulary only on ORIGIN training documents, reserves 15%
of each outer training pool for validation, and chooses the decision threshold
on validation F1. TF-IDF uses `min_df=10`, `max_df=0.98`, and
`max_features=360000`; the five model settings match the experiment scripts.

Nebula uses a separate serializer,
`model.pa_features.pa_execute_to_nebula_document`, which retains package
metadata and normalizes the execution events. Its `train_and_eval` function
accepts one outer fold and the corresponding `origin_train_docs` in the same
package order; it makes the same 15% validation split. Apply the same outer
10-fold splits as the classical evaluation. The model, whitespace tokenizer
and training components are reused under `model/_nebula/`, with their license.
Defaults are 30 epochs, learning rate 0.0005, batch size 16, early stopping,
and a validation-selected threshold. A reduced `epochs` value is only a
functional check, not a reproduction of the reported performance.
