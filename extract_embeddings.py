"""Precompute the frozen-backbone embeddings, one vector per product (external_code).

    vision_pooled.npz  FashionSigLIP (Marqo/marqo-fashionSigLIP) image embedding, both datasets
    text_bert.npz      BERT-base [CLS] of "<color> <fabric> <category>", VISUELLE 2.0 only
                       (on VISUELLE the frozen BERT runs inside the model)

Example:
    python extract_embeddings.py --dataset visuelle
"""
import argparse
import os

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from retmf.data import read_split

ImageFile.LOAD_TRUNCATED_IMAGES = True


@torch.no_grad()
def main(args):
    products = pd.concat([read_split(args.data_dir, args.dataset, s) for s in ('train', 'test')],
                         ignore_index=True).drop_duplicates('external_code')
    print(f'{len(products)} products')

    siglip, _, preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
    siglip = siglip.to(args.device).eval()
    if args.dataset == 'visuelle2':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        bert = AutoModel.from_pretrained('bert-base-uncased').to(args.device).eval()

    vision, text = {}, {}
    for row in tqdm(products.itertuples(), total=len(products), ascii=True):
        code = str(row.external_code)
        image = Image.open(os.path.join(args.data_dir, 'images', row.image_path)).convert('RGB')
        vision[code] = siglip.encode_image(preprocess(image).unsqueeze(0).to(args.device),
                                           normalize=False).squeeze(0).float().cpu().numpy()
        if args.dataset == 'visuelle2':
            tokens = tokenizer([f'{row.color} {row.fabric} {row.category}'], return_tensors='pt',
                               padding=True, truncation=True).to(args.device)
            text[code] = bert(**tokens).last_hidden_state[0, 0].float().cpu().numpy()

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, 'vision_pooled.npz'), **vision)
    if text:
        np.savez_compressed(os.path.join(args.out_dir, 'text_bert.npz'), **text)
    print(f'saved to {args.out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['visuelle', 'visuelle2'], default='visuelle')
    parser.add_argument('--data_dir', default=None, help='default: data/<dataset>')
    parser.add_argument('--out_dir', default=None, help='default: embeddings/<dataset>')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.data_dir = args.data_dir or os.path.join('data', args.dataset)
    args.out_dir = args.out_dir or os.path.join('embeddings', args.dataset)
    main(args)
