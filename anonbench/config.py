"""Configuration loading and canonical work-directory layout.

Every script calls `load_config()` and then uses the `Paths` helper so that all
artifacts end up in predictable places under `paths.work_dir`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str | None, base: Path) -> Path | None:
    if p is None or p == "":
        return None
    p = os.path.expanduser(os.path.expandvars(str(p)))
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> dict:
    """Load config.yaml (or $ANONBENCH_CONFIG) and resolve relative paths against the repo root.

    `overrides` are "a.b.c=value" strings (value parsed as YAML), e.g. from --set on the CLI.
    """
    path = Path(path or os.environ.get("ANONBENCH_CONFIG", REPO_ROOT / "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
    base = path.resolve().parent
    for k, v in list(cfg["paths"].items()):
        cfg["paths"][k] = _resolve(v, base)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def config_tag(params: dict) -> str:
    """Stable short tag for an anonymizer parameter dict (used as folder name)."""
    items = sorted((k, v) for k, v in params.items() if not k.startswith("_"))
    readable = "_".join(f"{k}-{v}" for k, v in items if k != "name")
    if "name" in params:
        readable = f"{params['name']}__{readable}"
    readable = readable.replace("/", "-").replace(" ", "")
    if len(readable) > 90:
        h = hashlib.md5(json.dumps(items, default=str).encode()).hexdigest()[:8]
        readable = readable[:80] + "_" + h
    return readable or "default"


@dataclass
class Paths:
    cfg: dict
    dataset: str = "mtf"

    @property
    def work(self) -> Path:
        return Path(self.cfg["paths"]["work_dir"])

    @property
    def third_party(self) -> Path:
        return Path(self.cfg["paths"]["third_party"])

    @property
    def weights(self) -> Path:
        return Path(self.cfg["paths"]["weights_dir"])

    def index_csv(self) -> Path:
        return self.work / "index" / f"{self.dataset}.csv"

    def landmarks_csv(self) -> Path:
        return self.work / "index" / f"{self.dataset}_landmarks.csv"

    def model_dir(self, name: str, seed: int = 0) -> Path:
        return self.work / "models" / self.dataset / f"{name}_s{seed}"

    def anon_dir(self, method: str, tag: str, split: str, seed: int = 0) -> Path:
        return self.work / "anon" / self.dataset / method / tag / f"seed{seed}" / split

    def emb_dir(self) -> Path:
        return self.work / "embeddings" / self.dataset

    def results(self, exp: str) -> Path:
        d = self.work / "results" / self.dataset / exp
        d.mkdir(parents=True, exist_ok=True)
        return d


def add_third_party_to_path(cfg: dict, repo: str) -> Path:
    """Put one of the cloned original repos on sys.path (needed for their relative imports)."""
    p = Path(cfg["paths"]["third_party"]) / repo
    if not p.exists():
        raise FileNotFoundError(
            f"Original repository not found at {p}. Run ./setup.sh first (it clones the three codebases)."
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def method_configs(cfg: dict, method: str, which: str = "faithful") -> list[dict]:
    """Return a list of parameter dicts for a method: 'faithful', 'sweep', 'controls' or 'all'."""
    mc = cfg["anonymizers"].get(method, {})
    out: list[dict] = []
    if which in ("faithful", "all") and "faithful" in mc:
        out.append({**mc["faithful"], "_kind": "faithful"})
    if which in ("sweep", "all"):
        for p in mc.get("sweep", []) or []:
            out.append({**copy.deepcopy(mc.get("faithful", {})), **p, "_kind": "sweep"})
    if which in ("controls", "all"):
        for p in mc.get("controls", []) or []:
            out.append({**copy.deepcopy(mc.get("faithful", {})), **p, "_kind": "control"})
    # dedupe by tag
    seen, uniq = set(), []
    for p in out:
        t = config_tag(p)
        if t not in seen:
            seen.add(t)
            uniq.append(p)
    return uniq
