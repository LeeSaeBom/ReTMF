"""Train ReTMF on VISUELLE or VISUELLE 2.0.

Example:
    python train.py --dataset visuelle --trend_len 52 --seed 7
"""
import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from retmf.data import (build_vocab, holdout_recent, make_loader, read_gtrends, read_split,
                        sales_scale, visuelle2_dataset, visuelle_dataset)
from retmf.model import OUTPUT_LEN, PRESETS, ReTMF, save_checkpoint

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main(args):
    pl.seed_everything(args.seed)
    train_df = read_split(args.data_dir, args.dataset, 'train')
    test_df = read_split(args.data_dir, args.dataset, 'test')
    gtrends = read_gtrends(args.data_dir, args.dataset)

    # Checkpoint selection monitors the test split, as in the GTM-Transformer protocol used
    # by the paper. With --val_split, the most recently released share of the training
    # products is held out instead and the test split is never seen during training.
    val_df = test_df
    if args.val_split:
        train_df, val_df = holdout_recent(train_df, args.val_split)
        print(f'validation split: {val_df["external_code"].nunique()} held-out products, '
              f'{train_df["external_code"].nunique()} remain for training')

    vocab, n_stores = None, None
    if args.dataset == 'visuelle':
        vocab = build_vocab(train_df, val_df, test_df)
        train_set = visuelle_dataset(train_df, gtrends, args.emb_dir, vocab, args.trend_len)
        val_set = visuelle_dataset(val_df, gtrends, args.emb_dir, vocab, args.trend_len)
    else:
        n_stores = int(max(train_df['retail'].max(), val_df['retail'].max(), test_df['retail'].max())) + 1
        train_set = visuelle2_dataset(train_df, gtrends, args.emb_dir, args.trend_len)
        val_set = visuelle2_dataset(val_df, gtrends, args.emb_dir, args.trend_len)

    preset = PRESETS.get((args.dataset, args.trend_len), PRESETS[(args.dataset, 52)])
    config = dict(
        output_len=OUTPUT_LEN[args.dataset],
        trend_len=args.trend_len,
        num_layers=args.num_layers or preset['num_layers'],
        num_heads=args.num_heads or preset['num_heads'],
        ff_mult=args.ff_mult or preset['ff_mult'],
        embedding_dim=32,
        hidden_dim=64,
        vision_dim=768,
        num_trends=3,
        n_stores=n_stores,
        sales_scale=sales_scale(args.data_dir, args.dataset),
        bert_name='bert-base-uncased',
    )
    model = ReTMF(**config, vocab=vocab)

    run_name = f'{args.dataset}_in{args.trend_len}_seed{args.seed}' + ('_val' if args.val_split else '')
    # The epoch with the lowest validation MAE is kept.
    checkpoint = ModelCheckpoint(dirpath=os.path.join(args.log_dir, run_name), filename='{epoch}',
                                 monitor='val_mae', mode='min', save_top_k=1)
    if args.wandb_project:
        logger = WandbLogger(project=args.wandb_project, name=run_name)
    else:
        logger = CSVLogger(args.log_dir, name=run_name)
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=[args.gpu] if torch.cuda.is_available() else 1,
                         max_epochs=args.epochs, check_val_every_n_epoch=1,
                         logger=logger, callbacks=[checkpoint])
    trainer.fit(model, train_dataloaders=make_loader(train_set, args.dataset, train=True),
                val_dataloaders=make_loader(val_set, args.dataset, train=False))

    best = torch.load(checkpoint.best_model_path, map_location='cpu', weights_only=False)
    out = args.out or os.path.join('checkpoints_retrained', f'{args.dataset}_in{args.trend_len}',
                                   f'seed{args.seed}.pth')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    save_checkpoint(out, best['state_dict'], args.dataset, config,
                    meta=dict(seed=args.seed, epoch=int(best['epoch']), val_split=args.val_split))
    print(f'best val_mae {float(checkpoint.best_model_score):.3f} at epoch {best["epoch"]}; saved {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['visuelle', 'visuelle2'], default='visuelle')
    parser.add_argument('--data_dir', default=None, help='default: data/<dataset>')
    parser.add_argument('--emb_dir', default=None, help='default: embeddings/<dataset>')
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--val_split', type=float, default=0.0,
                        help='share of the most recently released training products to hold out '
                             'for checkpoint selection; 0 monitors the test split as in the paper')
    parser.add_argument('--num_layers', type=int, default=None, help='default: paper preset')
    parser.add_argument('--num_heads', type=int, default=None, help='default: paper preset')
    parser.add_argument('--ff_mult', type=int, default=None, help='default: paper preset')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--out', default=None, help='path of the release-format checkpoint')
    parser.add_argument('--wandb_project', default=None, help='log to Weights & Biases instead of CSV')
    args = parser.parse_args()
    args.data_dir = args.data_dir or os.path.join('data', args.dataset)
    args.emb_dir = args.emb_dir or os.path.join('embeddings', args.dataset)
    main(args)
