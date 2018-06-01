## Introduction

Dependency parser using the arc hybrid system and bidirectional LSTM features, based on [Kiperwasser & Goldberg (2016)](https://aclweb.org/anthology/Q16-1023). Main differences:
- As a loss function, cross entropy is used instead of hinge loss.
- No error exploration is performed.

## Requirements

The model was implemented in Python 3 using PyTorch. See the [website](https://pytorch.org/) to install the right version.

## Quickstart

From the command line `main.py` can be run to train a new model. E.g.:

    # Train a model on TRAIN_FILE for 5 epochs with a 2-layered LSTM with 64 hidden units
    python3 main.py --train $TRAIN_FILE --epochs 5 --num_layers_lstm 2 --hidden_units_lstm 64 --model_name 'example_model'

The script `example.sh` further illustrates the usage of `main.py`.

## TODO
- Tokenize vocabulary?
- Hyperparameter tuning (especially: balance batch size & learning rate)
- Extract internal states