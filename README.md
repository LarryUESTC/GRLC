## GRLC: Graph Representation Learning with Constraints
This repository contains the reference code for the paper Graph Representation Learning with Constraints (TNNLS submission).

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Basics](#Basics)
0. [GPU Setting](#GPU Setting)

## Installation
pip install -r requirements.txt 

## Preparation


Dataset (`--dataset-class`, `--dataset-name`,`--Custom-key`)

| Dataset class   | Dataset name        |Custom key    |
|-----------------|---------------------|--------------|
| Planetoid              | Cora         |classification|
| Planetoid              | CiteSeer     |classification|
| Planetoid              | PubMed       |classification|
| WikiCS                 | WikiCS       |classification|
| MyAmazon               | Photo        |classification|
| MyCitationFull         | CoraFull     |classification|
| MyCitationFull         | DBLP         |classification|
| Crocodile              | Crocodile    |classification|
| PygNodePropPredDataset | ogbn-arxiv   |classification|
| PygNodePropPredDataset | ogbn-mag     |classification|
| PygNodePropPredDataset | ogbn-products|classification|

Important args:
* `--usepretraining` Test checkpoints
* `--dataset-class` Planetoid, MyAmazon, WikiCS, MyCitationFull, Crocodile, PygNodePropPredDataset
* `--dataset-name` Cora, CiteSeer, PubMed, Photo, WikiCS, CoraFull, DBLP, Crocodile, ogbn-arxiv, ogbn-mag, ogbn-products
* `--custom_key` classification, link, clu

## Basics
- The main train/test code is in `Code_GRLC/train.py`.
- If you want to see the UGRL layer in PyTorch Geometric `MessagePassing` grammar, refer to `Code_GRLC/layers`.


## GPU Setting

There are three arguments for GPU settings (`--num-gpus-total`, `--num-gpus-to-use`, `--black-list`).
Default values are from the author's machine, so we recommend you modify these values from `GRLC/args.yaml` or by the command line.
- `--num-gpus-total` (default 4): The total number of GPUs in your machine.
- `--num-gpus-to-use` (default 1): The number of GPUs you want to use.
- `--black-list` (default: [1, 2, 3]): The ids of GPUs you want to not use.

