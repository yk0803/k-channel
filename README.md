# k-Channel

Code for *Recognition Rate Is Not Anonymity: Linkage Attacks on Face Anonymization and a Release With a Provable Bound*.

The repository contains

- `anonbench/`: the evaluation pipeline and the attack suite (recognition panel, generic and learned linkage, restoration, grouping);
- `kchannel/`: the k-Channel release algorithm and the experiments behind it (pool generation and description, descriptor sweep, ablation, release build);
- `evaluation/`: utility measures (attribute accuracy on 40 CelebA facial attributes, attribute fidelity, race-group agreement, FID);
- `scripts/`: the evaluation and timing runners;
- `data/pool/`: labels (face detection, age, gender, head pose) of the 160k generated pool faces.

## Setup

```bash
pip install -r requirements.txt
export KCHANNEL_ROOT=/path/to/data   # datasets, generated pool, features, releases and results live here
```

CelebA and MTF are not redistributed. Place them under `$KCHANNEL_ROOT` and set the paths in `config.yaml`, which uses `/path/to/k-channel-data` as a placeholder. The anonymization methods we evaluate are obtained from their authors' repositories. Trained checkpoints (EDIALS autoencoders, guide classifiers) are attached to the GitHub release.

## k-Channel

```bash
bash kchannel/generate_pool.sh               # StyleGAN2-FFHQ faces, truncation 0.7
python kchannel/label_pool.py                # face detection, age, gender, head pose
python kchannel/describe_pool.py             # labels and CLIP ViT-L/14 features of the added faces
python kchannel/prepare_features.py          # cached features for the probes and the pool
python kchannel/descriptor_sweep.py          # descriptor weights on the validation split
python kchannel/ablation.py                  # ablation of each component
python kchannel/build_release.py celeba      # releases at k = 5, 10, 25; shuffled, renamed export without metadata
python kchannel/build_release.py mtf
```

Each release is written twice: in the manifest format read by the attack suite, and as a publishable export (`export/v7_<dataset>_k<k>/`) with source labels and cell-mean labels. The mapping from exported files to source images is written outside the export.

## Attacks and utility

```bash
bash scripts/run_evaluation.sh               # waits for a free GPU; resumable
```

## Citation

Citation details will be added on publication.

## Checkpoints

The trained EDIALS autoencoders and the ConvNeXt-Base guide classifiers are attached to release
[v1.0](https://github.com/yk0803/k-channel/releases/tag/v1.0). Download them with

```bash
gh release download v1.0 --repo yk0803/k-channel --dir checkpoints
cd checkpoints && sha256sum -c SHA256SUMS.txt
```
