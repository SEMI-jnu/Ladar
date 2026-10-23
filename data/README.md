# Example data

Real artifacts from the paper's experiments, sized to inspect the repair
and detection components on a laptop.

## Layout

```
data/
  example_package/                         3 malicious case packages
    safepackage-1337.0.2/
      original/      pre-APR malicious source
      patched/       post-LADAR source
    equests-2.27.1/
      original/
      patched/
    r4quests-2.28.1/
      original/
      patched/
  pa_logs/                                 6 PA execute logs (3 cases x ORIGIN/LADAR)
    safepackage_{origin,ladar}_execute.json
    equests_{origin,ladar}_execute.json
    r4quests_{origin,ladar}_execute.json
  pa_logs_benign/                          30 benign PA execute logs (Status=completed)
  model_output/                            produced by model/run_detection_example.py
    tfidf_lr_predictions.csv
    tfidf_lr_metrics.txt
```

## Cases

| Case                 | ORIGIN error                                | Behavior revealed after repair                                 |
| -------------------- | ------------------------------------------- | -------------------------------------------------------------- |
| safepackage-1337.0.2 | `requests.ConnectTimeout`                   | file access (`/etc/passwd`, `/root/.netrc`), DNS lookup for a webhook host, and a shell command |
| equests-2.27.1       | `HTTPError 404: Not Found`                  | Monero mining (XMRig + supportxmr.com pool)                    |
| r4quests-2.28.1      | `ModuleNotFoundError: No module 'frequest'` | Shared external infrastructure (`34.94.72.179:80`)             |

Each case can be used for one patch-generation attempt through
[`preprocessing/run_apr_example.py`](../preprocessing/run_apr_example.py)
(pass the package name as argv). This example does not rerun PA or reproduce
the paper's repair-success evaluation.

## PA log schema

Each JSON follows the OpenSSF [Package Analysis][pa] format. The fields
the pipeline consumes are:

- `Analysis.execute.Status`   -- `completed` or `error_analysis`
- `Analysis.execute.Stderr`   -- base64 of the traceback (on failure)
- `Analysis.execute.Files`    -- list of `{Path, Read, Write, Delete}`
- `Analysis.execute.Sockets`  -- list of `{Address, Port, Hostnames}`
- `Analysis.execute.DNS`      -- list of `{Queries: [{Hostname, Types}]}`
- `Analysis.execute.Commands` -- list of `{Command: [str]}`

[pa]: https://github.com/ossf/package-analysis
