# LSTMSeq2ML

Code for learning how to train a LSTM

# Installation

Install this project with

```
python3 -m pip install --no-cache-dir --editable . 'tensorflow>=2.0.0'
```

# Command-line interface

## Search for targets

The `seq2ml search` program can be used to search through a dataset using keywords or a regular expression. Searches are case-insensitive.

```
seq2ml search -f path/to/data.hdf5 -k ventilator -k alveolar
```

```
seq2ml search -f path/to/data.hdf5 -r '(\bVentilator\b|\bAlveolar\b)'
```

## Display most popular targets

This will print the 25 most popular targets in the training set.

```
seq2ml popular --filepath ~/data/processed_ohdsi_sequences_20200111.hdf5
```

## Training

The following code example trains a GRU model (named `gru_tiny`) to predict the `Evaluation finding` target. This uses the GPU device `0` as defined in `CUDA_VISIBLE_DEVICES`. To use multiple GPUs, prepend the `seq2ml train` call with `CUDA_VISIBLE_DEVICES=0,1`.

```
CUDA_VISIBLE_DEVICES=0 seq2ml train \
    --filepath ~/data/processed_ohdsi_sequences_20200111.hdf5 \
    --target-name "Evaluation finding" \
    --model-name gru_tiny \
    --learning-rate 0.001 \
    --batch-size 48 \
    --epochs 5 \
    --save-history \
    --evaluate
```

The following code example trains using a GRU model that does not use the fast cuDNN implementation of GRU. Notice that the training time is much longer than that of the cuDNN-compatible model.

```
CUDA_VISIBLE_DEVICES=0 seq2ml train \
    --filepath ~/data/processed_ohdsi_sequences_20200111.hdf5 \
    --target-name "Evaluation finding" \
    --model-name gru_tiny_no_cudnn \
    --learning-rate 0.001 \
    --batch-size 48 \
    --epochs 5 \
    --save-history \
    --evaluate
```
