# ReTMF

Code and pretrained weights for **ReTMF** (Reference-preserving Transformer with Multimodal Fusion), the forecasting model of the paper

> **Uncertainty-Aware Multimodal Learning for Cold-Start Demand Forecasting and Inventory Decision-Making in Fashion Retail** (under review)

ReTMF forecasts the first weeks of sales of a new fashion product using only information available before launch:

- the product image
- a short attribute text (color, fabric, category)
- the launch timing
- the Google Trends of the product attributes over the year before launch

The text embedding anchors a two-stage pairwise fusion that keeps every unimodal embedding next to the fused ones. An autoregressive Transformer decoder then attends to the encoded trends.

## Results of the released weights

Each setting has seven models trained with different seeds. MAE is in sales units and WAPE in percent. The ensemble averages the seven forecasts.

| Dataset | Trend input / horizon (weeks) | Single model, mean ± sd (MAE / WAPE) | 7-seed ensemble (MAE / WAPE) |
|---|---|---|---|
| VISUELLE | 52 / 6 | 28.53 ± 0.20 / 52.22 ± 0.36 | **27.70 / 50.70** |
| VISUELLE | 28 / 6 | 28.44 ± 0.30 / 52.07 ± 0.55 | **27.54 / 50.41** |
| VISUELLE 2.0 | 52 / 12 | 0.942 ± 0.002 / 80.12 ± 0.13 | **0.931 / 79.25** |

The trend input is the number of Google Trends weeks the model receives. With 28 weeks the model uses the first 28 weeks of the 52-week window before release, following the GTM-Transformer data pipeline. On VISUELLE 2.0 the task is new-product demand per product and store.

## Installation

```bash
conda create -n retmf python=3.10 -y
conda activate retmf
pip install -r requirements.txt
```

`requirements.txt` pins the versions the release was verified with. That includes PyTorch 2.8.0, whose default wheels target CUDA 12.8. For another CUDA version, install the matching PyTorch build first (https://pytorch.org).

## Data

The datasets are not redistributed here. Download them from the official repositories:

- VISUELLE: https://github.com/HumaticsLAB/GTM-Transformer
- VISUELLE 2.0: https://github.com/HumaticsLAB/visuelle2.0-code

Place them as follows:

```
data/
├── visuelle/    train.csv  test.csv  gtrends.csv  normalization_scale.npy  images/
└── visuelle2/   stfore_train.csv  stfore_test.csv  vis2_gtrends_data.csv  stfore_sales_norm_scalar.npy  images/
```

## 1. Extract the frozen embeddings

```bash
python extract_embeddings.py --dataset visuelle
python extract_embeddings.py --dataset visuelle2
```

The script writes `embeddings/<dataset>/vision_pooled.npz`, one FashionSigLIP image vector per product. For VISUELLE 2.0 it also writes `text_bert.npz`, the BERT [CLS] vector of each product's attribute text. On VISUELLE the frozen BERT runs inside the model instead.

On first use, the script downloads `Marqo/marqo-fashionSigLIP` and `bert-base-uncased` from Hugging Face. Extraction takes a few minutes per dataset on one GPU.

## 2. Evaluate the released checkpoints

```bash
python evaluate.py --dataset visuelle  --ckpt checkpoints/visuelle_in52
python evaluate.py --dataset visuelle  --ckpt checkpoints/visuelle_in28
python evaluate.py --dataset visuelle2 --ckpt checkpoints/visuelle2_in52
```

The script prints MAE and WAPE for each checkpoint, the mean ± sd over checkpoints, and the ensemble. Add `--output forecasts.csv` to save the ensemble forecast and each checkpoint's forecast for every product and week. Expected output for `visuelle_in52`:

```
checkpoint                          seed       MAE     WAPE
seed7.pth                              7    28.621    52.39
seed11.pth                            11    28.842    52.80
seed42.pth                            42    28.308    51.82
seed99.pth                            99    28.741    52.61
seed123.pth                          123    28.321    51.84
seed777.pth                          777    28.356    51.91
seed2025.pth                        2025    28.497    52.16
single model, mean +- sd                28.526 +- 0.198   52.22 +- 0.36
ensemble of 7                               27.699    50.70
```

## 3. Train from scratch

```bash
python train.py --dataset visuelle  --trend_len 52 --seed 7
python train.py --dataset visuelle  --trend_len 28 --seed 7
python train.py --dataset visuelle2 --trend_len 52 --seed 7
```

The paper uses seeds 7, 11, 42, 99, 123, 777 and 2025, reported there as seeds 1 to 7. Each run uses the following settings:

- 100 epochs with batch size 64
- MSE loss
- Adafactor with relative step sizes

As in GTM-Transformer, the test split is passed as the validation set, and the epoch with the lowest validation MAE is kept. To select the checkpoint without the test split, add `--val_split 0.1`: the most recently released 10% of the training products are held out for selection and the model trains on the rest. The best weights are written in the release format to `checkpoints_retrained/<dataset>_in<T>/seed<S>.pth`, which `evaluate.py` accepts.

GPU training is not bitwise deterministic, so retrained models differ slightly from the released ones. Use `--wandb_project <name>` to log to Weights & Biases instead of CSV files.

## Checkpoints

| Folder | Setting | Decoder |
|---|---|---|
| `checkpoints/visuelle_in52` | VISUELLE, 52-week trends, 6-week horizon | 2 layers, 8 heads, FFN 512 |
| `checkpoints/visuelle_in28` | VISUELLE, 28-week trends, 6-week horizon | 1 layer, 4 heads, FFN 256 |
| `checkpoints/visuelle2_in52` | VISUELLE 2.0, 52-week trends, 12-week horizon | 2 layers, 8 heads, FFN 512 |

Each file is named after its training seed and holds four entries:

- `dataset`
- `config`, the model arguments
- `state_dict`
- `meta`: the paper seed index, the training seed and the selected epoch

The frozen BERT weights are left out because they are identical to `bert-base-uncased`, which is loaded from Hugging Face. This keeps each file at about 3.5 MB. The model has 0.87 M trainable parameters for VISUELLE with 52-week trends.

To use a checkpoint in your own code:

```python
from retmf.data import build_vocab, read_split
from retmf.model import load_checkpoint

test_df = read_split('data/visuelle', 'visuelle', 'test')
model, ckpt = load_checkpoint('checkpoints/visuelle_in52/seed7.pth', vocab=build_vocab(test_df))
```

## Repository structure

```
retmf/model.py         ReTMF model (encoders, pairwise fusion, autoregressive decoder)
retmf/data.py          VISUELLE and VISUELLE 2.0 datasets
extract_embeddings.py  frozen FashionSigLIP and BERT embeddings
train.py               training
evaluate.py            per-checkpoint and ensemble evaluation
checkpoints/           released weights (21 models)
```

## Reproducibility

The paper's models were trained on an RTX 4090 with the following stack:

- Python 3.10
- PyTorch 2.5.1
- PyTorch Lightning 2.4.0
- transformers 4.46.0
- open_clip_torch 3.3.0

This release was verified on an RTX PRO 6000 (Blackwell) in a fresh environment built from `requirements.txt`. With the same embeddings, its forecasts are identical to those of the original research code. They reproduce the numbers above at the reported precision.

## Citation

Citation details will be added when the paper is published.

## License and acknowledgements

The code is released under the MIT License (see `LICENSE`). Parts of the data pipeline, the trend encoder and the decoder are derived from [GTM-Transformer](https://github.com/HumaticsLAB/GTM-Transformer) (MIT License, © 2021 HumaticsLAB). The VISUELLE 2.0 loader follows the official [VISUELLE 2.0 code](https://github.com/HumaticsLAB/visuelle2.0-code). The datasets are subject to their own terms of use.

We thank the authors of VISUELLE, VISUELLE 2.0, GTM-Transformer, [Marqo-FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) and [BERT](https://huggingface.co/bert-base-uncased).
