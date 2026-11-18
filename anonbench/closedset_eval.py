"""Closed-set evaluation of anonymizer runs with the classifiers trained on MTF originals.

This is the papers' OWN protocol (E1: reproduced vs reported) and the closed-set columns of the E2 matrix:
accuracy of each classifier on the anonymized images, RIR raw and floored (EDIALS Eq.1), and the same
classifier's accuracy on the CLEAN sources of the same images as the reference.

    python -m anonbench.closedset_eval --anon-dir <run dir> [...] [--models ...]
      -> work/results/<dataset>/closedset/closedset.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import Paths, load_config
from .embed import set_tag
from .manifest import read_manifest
from .metrics.closedset import closed_set
from .models.classifiers import load_classifier, predict_logits

EXTRA = ["convnext_base_ediAnon_nodecay"]


def _save(out: Path, row: dict):
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[~((old.run == row["run"]) & (old.model == row["model"]))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="mtf")
    ap.add_argument("--anon-dir", nargs="+", required=True)
    ap.add_argument("--models", nargs="*", default=None, help="default: frozen classifier list + EDI-Anon no-decay guide")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    P = Paths(cfg, a.dataset)
    models = a.models or (list(cfg["evaluators"]["classifiers"]) + EXTRA)
    runs = [Path(d) for d in a.anon_dir]
    out = P.results("closedset") / "closedset.csv"
    jobs = [(m, d) for m in models for d in runs]
    for i, (name, d) in enumerate(jobs, 1):
        t0 = time.time()
        md = P.model_dir(name, 0)
        if not (md / "meta.json").exists():
            print(f"[{i}/{len(jobs)}] SKIP {name}: no trained model at {md}", flush=True)
            continue
        if i == 1 or jobs[i - 2][0] != name:
            m, tf, meta = load_classifier(md, cfg)
        try:
            man = read_manifest(d)
            man = man[man.status != "failed"]
            C = meta["num_classes"]
            ra = closed_set(predict_logits(m, tf, man.anon_path.tolist(), cfg), man.label.to_numpy(), C)
            rc = closed_set(predict_logits(m, tf, man.src_path.tolist(), cfg), man.label.to_numpy(), C)
        except Exception as e:  # keep going
            print(f"[{i}/{len(jobs)}] ERROR {name} {d}: {e!r}", flush=True)
            continue
        rm = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
        row = dict(run=set_tag(cfg, d), method=rm.get("method", "?"), model=name, n=ra["n"],
                   acc=ra["rank1"], rir_raw=ra["rir_raw"], rir_floored=ra["rir_floored"], floor_active=ra["floor_active"],
                   rank5=ra["rank5"], acc_clean_same_images=rc["rank1"])
        _save(out, row)
        print(f"[{i}/{len(jobs)}] RESULT {name:30s} acc {ra['rank1'] * 100:6.2f}%  (clean {rc['rank1'] * 100:6.2f}%)  "
              f"RIR floored {ra['rir_floored']:.2f}%  on {set_tag(cfg, d)[:60]}  ({time.time() - t0:.0f}s, saved)", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
