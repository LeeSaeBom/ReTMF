"""Evaluate ReTMF checkpoints on the test split, per checkpoint and as an ensemble.

Example:
    python evaluate.py --dataset visuelle --ckpt checkpoints/visuelle_in52

MAE and WAPE are computed on sales in units over all test items and forecast weeks. The
ensemble averages the forecasts of all checkpoints given.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from retmf.data import (build_vocab, make_loader, read_gtrends, read_split, sales_scale,
                        visuelle2_dataset, visuelle_dataset)
from retmf.model import load_checkpoint

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def metrics(gt, pred):
    err = np.abs(gt - pred)
    return err.mean(), 100 * err.sum() / gt.sum()


@torch.no_grad()
def predict(model, loader, device):
    model.to(device).eval()
    preds = []
    for batch in loader:
        _, *inputs = [x.to(device) for x in batch]
        preds.append(model(*inputs)[0].cpu().numpy())
    return np.concatenate(preds)


def main(args):
    paths = sorted(glob.glob(os.path.join(args.ckpt, '*.pth'))) if os.path.isdir(args.ckpt) else [args.ckpt]
    if not paths:
        raise FileNotFoundError(f'no checkpoints found at {args.ckpt}')
    headers = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    order = sorted(range(len(paths)), key=lambda i: (headers[i]['meta'].get('run', 0), paths[i]))
    paths, headers = [paths[i] for i in order], [headers[i] for i in order]
    trend_len = {h['config']['trend_len'] for h in headers}
    if len(trend_len) != 1 or any(h['dataset'] != args.dataset for h in headers):
        raise ValueError('all checkpoints must belong to --dataset and share the trend length')
    trend_len = trend_len.pop()

    test_df = read_split(args.data_dir, args.dataset, 'test')
    gtrends = read_gtrends(args.data_dir, args.dataset)
    vocab = None
    if args.dataset == 'visuelle':
        vocab = build_vocab(test_df)
        test_set = visuelle_dataset(test_df, gtrends, args.emb_dir, vocab, trend_len)
    else:
        test_set = visuelle2_dataset(test_df, gtrends, args.emb_dir, trend_len)
    loader = make_loader(test_set, args.dataset, train=False)
    scale = sales_scale(args.data_dir, args.dataset)
    horizon = headers[0]['config']['output_len']
    gt = test_set.tensors[0][:, :horizon].numpy().astype(np.float64) * scale

    forecasts = []
    print(f'{"checkpoint":<34}{"seed":>6}{"MAE":>10}{"WAPE":>9}')
    for path in paths:
        model, ckpt = load_checkpoint(path, vocab)
        pred = predict(model, loader, args.device).astype(np.float64) * scale
        forecasts.append(pred)
        mae, wape = metrics(gt, pred)
        print(f'{os.path.basename(path):<34}{ckpt["meta"].get("seed", "-"):>6}{mae:>10.3f}{wape:>9.2f}')
    if len(forecasts) > 1:
        maes, wapes = zip(*(metrics(gt, f) for f in forecasts))
        print(f'{"single model, mean +- sd":<40}{np.mean(maes):>6.3f} +- {np.std(maes):.3f}'
              f'{np.mean(wapes):>8.2f} +- {np.std(wapes):.2f}')
        mae, wape = metrics(gt, np.mean(forecasts, axis=0))
        print(f'{f"ensemble of {len(forecasts)}":<40}{mae:>10.3f}{wape:>9.2f}')

    if args.output:
        keys = ['external_code'] + (['retail'] if args.dataset == 'visuelle2' else [])
        rows = test_df[keys].loc[test_df.index.repeat(horizon)].reset_index(drop=True)
        rows['week'] = np.tile(np.arange(1, horizon + 1), len(test_df))
        rows['sales'] = gt.reshape(-1)
        rows['forecast'] = np.mean(forecasts, axis=0).reshape(-1)
        for path, pred in zip(paths, forecasts):
            rows['forecast_' + os.path.splitext(os.path.basename(path))[0]] = pred.reshape(-1)
        rows.to_csv(args.output, index=False)
        print(f'ensemble and per-checkpoint forecasts written to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['visuelle', 'visuelle2'], default='visuelle')
    parser.add_argument('--ckpt', required=True, help='checkpoint file or directory of checkpoints')
    parser.add_argument('--data_dir', default=None, help='default: data/<dataset>')
    parser.add_argument('--emb_dir', default=None, help='default: embeddings/<dataset>')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default=None, help='optional CSV with the ensemble forecasts')
    args = parser.parse_args()
    args.data_dir = args.data_dir or os.path.join('data', args.dataset)
    args.emb_dir = args.emb_dir or os.path.join('embeddings', args.dataset)
    main(args)
