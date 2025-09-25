# RadioML Source Classification

This project trains a convolutional neural network (CNN) to classify the modulation type of RadioML 2016.10a signals using a subset of the open-source dataset mirrored on GitHub.

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Download the dataset subset

The dataset is retrieved from the public mirror [`dannis999/RML2016.10a`](https://github.com/dannis999/RML2016.10a). Download a manageable subset of SNR levels with:

```bash
python scripts/download_rml2016_subset.py
```

By default the script fetches five representative SNR levels (-20, -10, 0, 10, 18 dB) for every modulation class. Use `--snr` or `--modulation` flags to customize the selection.

## Train the CNN classifier

Run the training pipeline (downloads automatically if necessary when `--auto-download` is supplied):

```bash
python train.py --auto-download --epochs 12 --batch-size 256 --max-examples-per-class 600
```

Training artifacts (best model checkpoint, metrics JSON, confusion matrix plot) are written to the `artifacts/` directory. Adjust hyperparameters or dataset filters via CLI arguments; see `python train.py --help` for the full list.

## Reproducing results quickly

For a quicker sanity check use fewer epochs and a smaller subset:

```bash
python train.py --auto-download --epochs 3 --max-examples-per-class 200
```

This produces a compact training run suitable for limited compute environments while still demonstrating the CNN's ability to learn the modulation classes.
