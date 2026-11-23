# Results

All numbers come from `results/*.json`, produced by `run.py` on a 2-core CPU. Test set: 624 FF++ face crops (201 real, 423 fake). Threshold 0.5. Best epoch picked by validation AUC.

## In-dataset (train and test on all four manipulations, 6 epochs)

| Model | Params (M) | Acc (%) | Bal. acc (%) | Precision | Recall | F1 | AUC | EER | CPU ms/img |
|---|---|---|---|---|---|---|---|---|---|
| MesoNet-4 | 0.01 | 46.63 | 54.24 | 0.739 | 0.329 | 0.455 | 0.606 | 0.442 | 0.5 |
| Freq-CNN | 1.17 | 64.90 | 67.98 | 0.842 | 0.593 | 0.696 | 0.725 | 0.335 | 8.9 |
| EfficientNet-B0 | 5.29 | 72.12 | 69.77 | 0.814 | 0.764 | 0.788 | 0.773 | 0.302 | 4.4 |
| SFCA-Net | 7.99 | 73.08 | 71.53 | 0.829 | 0.759 | 0.793 | 0.780 | 0.285 | 12.4 |

### AUC per manipulation (real test vs. one manipulation)

| Model | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
|---|---|---|---|---|
| MesoNet-4 | 0.703 | 0.590 | 0.567 | 0.563 |
| Freq-CNN | 0.798 | 0.809 | 0.672 | 0.623 |
| EfficientNet-B0 | 0.840 | 0.768 | 0.763 | 0.720 |
| SFCA-Net | 0.841 | 0.779 | 0.793 | 0.706 |

## Robustness (accuracy %, model trained on clean data)

| Model | Clean | Blur k3 | Blur k7 | Noise s5 | Noise s10 | JPEG q23 | JPEG q40 | Motion k9 | Illum +30 | Illum -30 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MesoNet-4 | 46.6 | 58.7 | 67.3 | 44.4 | 35.1 | 45.5 | 46.8 | 60.9 | 48.1 | 45.4 | 50.2 |
| Freq-CNN | 64.9 | 68.6 | 59.1 | 32.2 | 32.2 | 49.7 | 45.4 | 56.1 | 64.6 | 66.3 | 52.7 |
| EfficientNet-B0 | 72.1 | 67.5 | 67.1 | 33.8 | 33.2 | 33.5 | 36.2 | 66.0 | 70.2 | 74.0 | 53.5 |
| SFCA-Net | 73.1 | 69.6 | 64.6 | 61.1 | 49.8 | 58.3 | 60.6 | 63.6 | 69.4 | 74.7 | 63.5 |

## Leave-one-manipulation-out (2 epochs)

C = test on real + the manipulation held out of training. E = test on real + seen manipulations.

| Model | Held out | E AUC | C AUC | C bal. acc (%) | Gap (E-C AUC) |
|---|---|---|---|---|---|
| MesoNet-4 | Deepfakes | 0.553 | 0.614 | 53.21 | -0.061 |
| MesoNet-4 | Face2Face | 0.598 | 0.579 | 55.05 | 0.019 |
| MesoNet-4 | FaceSwap | 0.603 | 0.553 | 53.73 | 0.050 |
| MesoNet-4 | NeuralTextures | 0.603 | 0.529 | 51.32 | 0.074 |
| **MesoNet-4** | **mean** | 0.589 | 0.569 | 53.33 | |
| EfficientNet-B0 | Deepfakes | 0.670 | 0.575 | 53.86 | 0.095 |
| EfficientNet-B0 | Face2Face | 0.699 | 0.611 | 56.80 | 0.088 |
| EfficientNet-B0 | FaceSwap | 0.728 | 0.561 | 51.28 | 0.167 |
| EfficientNet-B0 | NeuralTextures | 0.694 | 0.594 | 54.91 | 0.100 |
| **EfficientNet-B0** | **mean** | 0.698 | 0.585 | 54.21 | |
| SFCA-Net | Deepfakes | 0.681 | 0.596 | 58.63 | 0.086 |
| SFCA-Net | Face2Face | 0.723 | 0.637 | 60.87 | 0.086 |
| SFCA-Net | FaceSwap | 0.747 | 0.577 | 54.67 | 0.170 |
| SFCA-Net | NeuralTextures | 0.727 | 0.570 | 55.36 | 0.157 |
| **SFCA-Net** | **mean** | 0.720 | 0.595 | 57.38 | |

## RECAF (Pandey et al., 2026) over the four leave-one-out protocols

Entropy weights: wR = 0.2947, wE = 0.6464, wC = 0.0589; alpha1 = 0.0483, alpha2 = 0.1053.

| Model | R | E | C | RECAF |
|---|---|---|---|---|
| MesoNet-4 | 0.490 | 0.478 | 0.533 | 0.458 |
| EfficientNet-B0 | 0.573 | 0.663 | 0.542 | 0.596 |
| SFCA-Net | 0.612 | 0.679 | 0.574 | 0.622 |

## Video inference (SFCA-Net, up to 120 face frames per video)

| Video | Label | Frames | P(fake) mean-logit | P(fake) top-25% | Fake frames | Verdict |
|---|---|---|---|---|---|---|
| 000_003.mp4 | fake (Deepfakes) | 120 | 0.900 | 0.942 | 100% | FAKE |
| 009.mp4 | real | 120 | 0.597 | 0.808 | 63% | FAKE |
| 100.mp4 | real | 120 | 0.166 | 0.381 | 4% | REAL |

These FF++ sample videos may overlap with the training videos; treat this as a pipeline demo, not a test result.

## Notes

- Small images (96x96), few epochs and CPU-only training keep absolute numbers well below GPU results in the literature.
- Leave-one-out runs use 2 epochs, so compare models within that table, not against the 6-epoch in-dataset table.
