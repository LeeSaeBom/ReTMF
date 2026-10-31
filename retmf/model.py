"""ReTMF: Reference-preserving Transformer with Multimodal Fusion.

ReTMF forecasts the first weeks of sales of a new fashion product from four inputs that exist
before launch: the product image, a short attribute text, the launch timing, and the Google
Trends of the product attributes over the year before launch.
"""
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import Adafactor

# Decoder settings used in the paper, keyed by (dataset, trend input length).
PRESETS = {
    ('visuelle', 52): dict(num_layers=2, num_heads=8, ff_mult=8),
    ('visuelle', 28): dict(num_layers=1, num_heads=4, ff_mult=4),
    ('visuelle2', 52): dict(num_layers=2, num_heads=8, ff_mult=8),
}
OUTPUT_LEN = {'visuelle': 6, 'visuelle2': 12}
FROZEN_PREFIX = 'text_encoder.bert.'


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence-first inputs of shape (L, B, d)."""

    def __init__(self, d_model, dropout=0.1, max_len=52):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])


class TimeDistributed(nn.Module):
    """Applies a module to every step of a (B, L, F) sequence."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if x.dim() <= 2:
            return self.module(x)
        y = self.module(x.contiguous().view(-1, x.size(-1)))
        return y.contiguous().view(x.size(0), -1, y.size(-1))


class TrendEncoder(nn.Module):
    """Google Trends encoder.

    The weekly trend vector (category, color, fabric) is linearly embedded, positionally
    encoded and passed through a Transformer encoder whose self-attention is restricted to
    non-overlapping blocks of gcd(T, H) weeks, where T is the input length and H the horizon.
    """

    def __init__(self, forecast_horizon, d_model, trend_len, num_trends=3):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, d_model))
        self.pos_embedding = PositionalEncoding(d_model, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def _block_mask(self, size, device):
        block = math.gcd(size, self.forecast_horizon)
        mask = torch.zeros(size, size)
        for i in range(0, size, block):
            mask[i:i + block, i:i + block] = 1
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0).to(device)

    def forward(self, gtrends):
        # (B, num_trends, T) -> memory of shape (T, B, d_model)
        x = self.input_linear(gtrends.permute(0, 2, 1))
        x = self.pos_embedding(x.permute(1, 0, 2))
        return self.encoder(x, self._block_mask(x.size(0), x.device))


class TextEncoder(nn.Module):
    """Frozen BERT [CLS] embedding of the text "<color> <fabric> <category>", followed by a
    trainable linear projection.

    With a vocabulary (VISUELLE), the frozen BERT runs inside the model and the input is a
    (B, 3) tensor of (category, color, fabric) indices. Without one (VISUELLE 2.0), the input
    is the precomputed (B, 768) [CLS] vector written by extract_embeddings.py.
    """

    def __init__(self, embedding_dim, vocab=None, bert_name='bert-base-uncased', in_dim=768):
        super().__init__()
        self.in_model = vocab is not None
        if self.in_model:
            self.categories, self.colors, self.fabrics = (list(v) for v in vocab)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
            self.bert = AutoModel.from_pretrained(bert_name)
            self.bert.requires_grad_(False)
            self.bert.eval()
        self.fc = nn.Linear(in_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def train(self, mode=True):
        super().train(mode)
        if self.in_model:
            self.bert.eval()  # frozen backbone: no dropout, also during training
        return self

    def forward(self, text):
        if self.in_model:
            descriptions = [f'{self.colors[col]} {self.fabrics[fab]} {self.categories[cat]}'
                            for cat, col, fab in text.tolist()]
            tokens = self.tokenizer(descriptions, return_tensors='pt', padding=True,
                                    truncation=True).to(self.fc.weight.device)
            with torch.no_grad():
                text = self.bert(**tokens).last_hidden_state[:, 0, :]
        return self.dropout(self.fc(text))


class TimingEncoder(nn.Module):
    """Launch-timing encoder: day, week, month and year of the release date are embedded
    separately and merged by a linear layer."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.fusion = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, timing):
        d, w, m, y = (timing[:, i].unsqueeze(1) for i in range(4))
        emb = torch.cat([self.day_embedding(d), self.week_embedding(w),
                         self.month_embedding(m), self.year_embedding(y)], dim=1)
        return self.dropout(self.fusion(emb))


def _fusion_mlp(in_dim, out_dim, dropout):
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Linear(in_dim, in_dim, bias=False),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim, out_dim),
    )


class PairwiseFusion(nn.Module):
    """Reference-preserving two-stage fusion with text as the anchor.

    Stage 1 fuses text with timing (AT) and text with image (VT). Stage 2 fuses both pairwise
    outputs together with the unchanged unimodal embeddings (and, on VISUELLE 2.0, the store
    embedding) into the product embedding.
    """

    def __init__(self, embedding_dim, hidden_dim, vision_dim=768, extra_dim=0, dropout=0.2):
        super().__init__()
        self.image_proj = nn.Linear(vision_dim, embedding_dim)
        self.fusion_at = _fusion_mlp(2 * embedding_dim, embedding_dim, dropout)
        self.fusion_vt = _fusion_mlp(2 * embedding_dim, embedding_dim, dropout)
        self.fusion_all = _fusion_mlp(5 * embedding_dim + extra_dim, hidden_dim, dropout)

    def forward(self, image_emb, text_emb, timing_emb, extra=None):
        image = self.image_proj(image_emb)
        at = self.fusion_at(torch.cat((timing_emb, text_emb), dim=1))
        vt = self.fusion_vt(torch.cat((text_emb, image), dim=1))
        parts = [timing_emb, at, text_emb, vt, image]
        if extra is not None:
            parts.append(extra)
        return self.fusion_all(torch.cat(parts, dim=1))


class DecoderLayer(nn.Module):
    """Post-norm Transformer decoder layer that also returns its cross-attention weights."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]))
        attended, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(attended))
        tgt = self.norm3(tgt + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(tgt))))))
        return tgt, attn_weights


def _causal_mask(size, device):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1).float()
    return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0).to(device)


class ReTMF(pl.LightningModule):
    """ReTMF forecaster.

    Args:
        output_len: forecast horizon in weeks (6 on VISUELLE, 12 on VISUELLE 2.0).
        trend_len: number of Google Trends weeks used as input.
        num_layers, num_heads, ff_mult: decoder depth, attention heads and FFN width multiplier.
        vocab: (categories, colors, fabrics) name lists for the in-model BERT (VISUELLE);
            None when precomputed [CLS] vectors are used (VISUELLE 2.0).
        n_stores: number of stores for the store embedding (VISUELLE 2.0), else None.
        sales_scale: factor that maps normalized sales back to units (validation metrics only).
    """

    def __init__(self, output_len=6, trend_len=52, num_layers=2, num_heads=8, ff_mult=8,
                 embedding_dim=32, hidden_dim=64, vision_dim=768, num_trends=3, vocab=None,
                 n_stores=None, sales_scale=1.0, bert_name='bert-base-uncased'):
        super().__init__()
        assert output_len <= 12, 'the decoder positional encoding covers at most 12 steps'
        self.output_len = output_len
        self.sales_scale = sales_scale

        self.timing_encoder = TimingEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim, vocab, bert_name)
        self.trend_encoder = TrendEncoder(output_len, hidden_dim, trend_len, num_trends)
        if n_stores:
            self.store_embedding = nn.Embedding(n_stores, embedding_dim)
            self.store_dropout = nn.Dropout(0.2)
        else:
            self.store_embedding = None
        self.fusion = PairwiseFusion(embedding_dim, hidden_dim, vision_dim,
                                     extra_dim=embedding_dim if n_stores else 0)

        self.pred_proj = TimeDistributed(nn.Linear(1, hidden_dim))
        self.cond_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, hidden_dim * ff_mult, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Dropout(0.2))
        self.validation_step_outputs = []

    def forward(self, timing, gtrends, image_emb, text, stores=None):
        """Returns the (B, output_len) normalized forecast and the last decoder layer's
        cross-attention weights over the trend memory."""
        timing_emb = self.timing_encoder(timing)
        text_emb = self.text_encoder(text)
        memory = self.trend_encoder(gtrends)
        store_emb = None
        if self.store_embedding is not None:
            store_emb = self.store_dropout(self.store_embedding(stores))
        product = self.fusion(image_emb, text_emb, timing_emb, store_emb).unsqueeze(0)

        # Autoregressive decoding: the product embedding is the first token; each weekly
        # forecast is projected and fused with the product embedding into the next token.
        tokens = product
        outputs = []
        for _ in range(self.output_len):
            out = self.pos_encoder(tokens)
            mask = _causal_mask(out.size(0), out.device)
            for layer in self.decoder_layers:
                out, attn_weights = layer(out, memory, mask)
            y_t = self.output_head(out[-1:])
            outputs.append(y_t)
            next_token = self.cond_proj(torch.cat([self.pred_proj(y_t), product], dim=-1))
            tokens = torch.cat([tokens, next_token], dim=0)
        forecast = torch.cat(outputs, dim=0).squeeze(-1).transpose(0, 1)
        return forecast.view(-1, self.output_len), attn_weights

    def training_step(self, batch, batch_idx):
        sales, *inputs = batch
        forecast, _ = self(*inputs)
        loss = F.mse_loss(forecast, sales[:, :self.output_len])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sales, *inputs = batch
        forecast, _ = self(*inputs)
        self.validation_step_outputs.append((sales[:, :self.output_len].detach(), forecast.detach()))

    def on_validation_epoch_end(self):
        sales = torch.cat([s for s, _ in self.validation_step_outputs])
        forecast = torch.cat([f for _, f in self.validation_step_outputs])
        self.validation_step_outputs.clear()
        sales_u, forecast_u = sales * self.sales_scale, forecast * self.sales_scale
        self.log('val_loss', F.mse_loss(forecast, sales))
        self.log('val_mae', F.l1_loss(forecast_u, sales_u), prog_bar=True)
        self.log('val_wape', (sales_u - forecast_u).abs().sum() / sales_u.abs().sum() * 100, prog_bar=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return Adafactor(params, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

    def on_save_checkpoint(self, checkpoint):
        # The frozen BERT is reloaded from Hugging Face, so it is not stored.
        state = checkpoint['state_dict']
        for key in [k for k in state if k.startswith(FROZEN_PREFIX)]:
            del state[key]


def save_checkpoint(path, model_state, dataset, config, meta=None):
    """Writes a release checkpoint: model config plus weights without the frozen BERT."""
    state = {k: v.detach().cpu() for k, v in model_state.items() if not k.startswith(FROZEN_PREFIX)}
    torch.save({'dataset': dataset, 'config': config, 'state_dict': state, 'meta': meta or {}}, path)


def load_checkpoint(path, vocab=None, map_location='cpu'):
    """Builds a ReTMF model from a release checkpoint. VISUELLE checkpoints need the vocabulary
    for the in-model BERT (see retmf.data.build_vocab)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if (ckpt['dataset'] == 'visuelle') != (vocab is not None):
        raise ValueError('a vocabulary is required for VISUELLE checkpoints and only for them')
    model = ReTMF(**ckpt['config'], vocab=vocab)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if unexpected or any(not k.startswith(FROZEN_PREFIX) for k in missing):
        raise RuntimeError(f'checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    return model, ckpt
