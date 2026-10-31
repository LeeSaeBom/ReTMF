"""Data loading for VISUELLE and VISUELLE 2.0.

Both datasets yield batches of (sales, timing, gtrends, image_emb, text[, store]):
    sales      (B, 12)  normalized weekly sales after release
    timing     (B, 4)   day, week, month and year of the release date
    gtrends    (B, 3, T) Google Trends of category, color and fabric before release
    image_emb  (B, 768) FashionSigLIP image embedding (extract_embeddings.py)
    text       VISUELLE: (B, 3) category/color/fabric indices for the in-model BERT;
               VISUELLE 2.0: (B, 768) precomputed BERT [CLS] vector
    store      (B,)     store index (VISUELLE 2.0 only)
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SALES_COLS = [str(i) for i in range(12)]
ATTRIBUTES = ('category', 'color', 'fabric')

FILES = {
    'visuelle': dict(train='train.csv', test='test.csv', gtrends='gtrends.csv',
                     scale='normalization_scale.npy'),
    'visuelle2': dict(train='stfore_train.csv', test='stfore_test.csv', gtrends='vis2_gtrends_data.csv',
                      scale='stfore_sales_norm_scalar.npy'),
}


def read_split(data_dir, dataset, split):
    return pd.read_csv(os.path.join(data_dir, FILES[dataset][split]), parse_dates=['release_date'])


def read_gtrends(data_dir, dataset):
    return pd.read_csv(os.path.join(data_dir, FILES[dataset]['gtrends']), index_col=[0], parse_dates=True)


def sales_scale(data_dir, dataset):
    """Factor that maps normalized sales back to units."""
    return float(np.load(os.path.join(data_dir, FILES[dataset]['scale'])))


def holdout_recent(df, frac):
    """Splits a training table into (train, validation) by holding out the `frac` share of
    distinct products with the most recent release dates (all of their rows)."""
    products = df.drop_duplicates('external_code').sort_values('release_date', kind='stable')
    n_val = max(1, int(len(products) * frac))
    val_codes = set(products['external_code'].iloc[-n_val:])
    mask = df['external_code'].isin(val_codes)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def build_vocab(*frames):
    """Sorted category, color and fabric names; the index of a name is its position."""
    return tuple(sorted(set().union(*(set(f[col]) for f in frames))) for col in ATTRIBUTES)


def trend_window(gtrends, keys, release_date, trend_len):
    """Google Trends of `keys` over the 52 weeks before release, truncated to the first
    `trend_len` weeks and min-max scaled per series. Returns (len(keys), trend_len)."""
    window = gtrends.loc[release_date - pd.DateOffset(weeks=52):release_date]
    rows = []
    for key in keys:
        series = window[key][-52:].values[:trend_len]
        if len(series) < trend_len:  # products released early in the trend history
            series = gtrends.loc[:release_date][key][-52:].values[:trend_len]
        rows.append(MinMaxScaler().fit_transform(series.reshape(-1, 1)).flatten())
    return np.vstack(rows).astype(np.float32)


def load_embeddings(path, codes):
    """Stacks the per-product vectors of an .npz file (keyed by external_code) in row order."""
    store = np.load(path)
    missing = sorted(set(codes) - set(store.files))
    if missing:
        raise KeyError(f'{len(missing)} products missing in {path}, e.g. {missing[:5]}')
    vectors = {c: store[c] for c in set(codes)}
    return torch.from_numpy(np.stack([vectors[c] for c in codes]).astype(np.float32))


def visuelle_dataset(df, gtrends, emb_dir, vocab, trend_len=52):
    """VISUELLE: one row per product."""
    df = df.reset_index(drop=True)
    trends = np.stack([trend_window(gtrends, (r.category, r.color, r.fabric), r.release_date, trend_len)
                       for r in tqdm(df.itertuples(), total=len(df), ascii=True, desc='trends')])
    codes = df['external_code'].astype(str).tolist()
    index = [{name: i for i, name in enumerate(names)} for names in vocab]
    text = np.stack([df[col].map(idx).values for col, idx in zip(ATTRIBUTES, index)], axis=1)
    return TensorDataset(
        torch.FloatTensor(df[SALES_COLS].values),
        torch.FloatTensor(df[['day', 'week', 'month', 'year']].values),
        torch.from_numpy(trends),
        load_embeddings(os.path.join(emb_dir, 'vision_pooled.npz'), codes),
        torch.LongTensor(text),
    )


def visuelle2_dataset(df, gtrends, emb_dir, trend_len=52):
    """VISUELLE 2.0: one row per product and store (new-product demand setting)."""
    df = df.reset_index(drop=True)
    codes = df['external_code'].astype(str).tolist()
    products = df.drop_duplicates('external_code')
    windows = {str(r.external_code): trend_window(gtrends, (r.category, r.color, r.fabric),
                                                  r.release_date, trend_len)
               for r in tqdm(products.itertuples(), total=len(products), ascii=True, desc='trends')}
    rd = df['release_date']
    timing = torch.vstack([
        torch.FloatTensor(rd.dt.day.values),
        torch.FloatTensor(rd.dt.isocalendar().week.values.astype(float)),
        torch.FloatTensor(rd.dt.month.values),
        torch.FloatTensor(rd.dt.year.values),
    ]).T
    timing = timing / timing.max(0)[0]  # scaled by the split maximum, as in the official loader
    return TensorDataset(
        torch.FloatTensor(df[SALES_COLS].values),
        timing,
        torch.from_numpy(np.stack([windows[c] for c in codes])),
        load_embeddings(os.path.join(emb_dir, 'vision_pooled.npz'), codes),
        load_embeddings(os.path.join(emb_dir, 'text_bert.npz'), codes),
        torch.LongTensor(df['retail'].values),
    )


def make_loader(dataset, name, train, batch_size=64):
    """Training batches of 64; evaluation runs one product at a time on VISUELLE (as in the
    paper, so the in-model BERT sees unpadded text) and in batches of 512 on VISUELLE 2.0."""
    if not train:
        batch_size = 1 if name == 'visuelle' else 512
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
