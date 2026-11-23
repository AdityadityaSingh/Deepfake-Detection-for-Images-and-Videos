# SFCA-Net: Spatial-Frequency Cross-Attention Network for Deepfake Detection

Author: Adityaditya, B.Tech CSAI, Netaji Subhas University of Technology

Enhanced deepfake detector for images and videos. It builds on the gaps found in the reviewed papers:
- Rana et al. (IEEE Access 2022): CNNs dominate. Combining methods helps.
- Ramanaharan et al. (2025): generalisation to unseen manipulations is the main open problem.
- Pandey et al. (2026): evaluate robustness (R), effectiveness (E) and cross-dataset generalisation (C) together with RECAF.

## Model

SFCA-Net has two streams:
1. **Spatial stream.** ImageNet-pretrained EfficientNet-B0. Its 3x3 feature map becomes 9 tokens.
2. **Frequency stream.** Per-image log-amplitude FFT plus 3 fixed SRM high-pass residual filters, fed to a light CNN. Its 6x6 map becomes 36 tokens.
3. **Fusion.** Bidirectional multi-head cross-attention between the two token sets, then a learned gate.
4. **Training.** BCE with label smoothing and class balancing, plus auxiliary heads on each stream (weight 0.3).
   Degradation-aware augmentation (random blur, noise, JPEG, brightness) uses ranges that differ from the test distortions.

Video detection: Haar-cascade face detection on sampled frames, per-frame P(fake), then temporal aggregation (mean logit and top-k).

Baselines: MesoNet-4 (scratch), Freq-CNN (frequency stream alone), EfficientNet-B0 (pretrained, spatial only).

## Data

FaceForensics++ face crops (Deepfakes, Face2Face, FaceSwap, NeuralTextures, and real YouTube) from the public mirror
`github.com/tranthuhoai3786/FaceForensics` (`image/` folder, original train/test split).
Train 5428 crops (1805 real), test 624 crops (201 real). Resized to 96x96. 10% of train is held out for validation.
No exact duplicates exist between train and test (dHash check). FaceForensics++ is under its own research licence.

## Experiments

| Protocol | What it measures |
|---|---|
| In-dataset (`all`) | Train and test on all four manipulations. 6 epochs. |
| Robustness | Test set under 9 distortions from Pandey et al.: Gaussian blur k3/k7, noise s5/s10, JPEG q40/q23, motion blur k9, illumination +/-30 |
| Leave-one-manipulation-out | Train without one manipulation, test on it. 4 protocols, 2 epochs (CPU budget). |
| RECAF | Pandey et al. score over the 4 leave-one-out protocols |
| Video | Frame-level scoring and aggregation on FF++ sample videos |

Hardware: 2-core CPU, no GPU. Numbers are not comparable with papers trained on GPUs at higher resolution and for more epochs.

## Files

| File | Purpose |
|---|---|
| `dfd.py` | Data loading, distortions, augmentation, all models, training loop, metrics |
| `run.py` | Runs experiments, saves `results/*.json` and checkpoints |
| `plots.py` | Builds every graph in `figures/` from the results |
| `video_infer.py` | Video pipeline and per-frame plot |
| `predict.py` | Classify one image or video from the command line |
| `results/` | JSON results per model and protocol, `recaf_scores.json`, `video_inference.json`, `ckpt/` weights |
| `figures/` | All graphs (PNG and PDF) |
| `RESULTS.md` | Result tables |

## Reproduce

```bash
pip install torch torchvision efficientnet_pytorch opencv-python-headless scikit-learn matplotlib
git clone https://github.com/tranthuhoai3786/FaceForensics   # set ROOT in dfd.py to its image/ folder
python3 run.py --models "MesoNet-4,Freq-CNN,EfficientNet-B0,SFCA-Net" --protocols all --epochs 6
python3 run.py --models "MesoNet-4,EfficientNet-B0,SFCA-Net" --protocols Deepfakes,Face2Face,FaceSwap,NeuralTextures --epochs 2
python3 plots.py
python3 video_infer.py --model SFCA-Net --videos 000_003.mp4 009.mp4 100.mp4 --labels fake real real
python3 predict.py some_face.jpg
```
On a GPU, raise `RES` in `dfd.py` to 224 and use 20+ epochs for stronger results.
