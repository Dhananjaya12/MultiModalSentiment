import torch
import torch.nn as nn
from transformers import DistilBertModel

from model.encoders import ModalityEncoder
from model.fusion import CrossModalFusion



class SentimentRegressor(nn.Module):
    """
    Final prediction head.

    Takes the 3 enriched sequences, averages each over time,
    concatenates, and predicts a single sentiment score (-3 to +3).

    audio  (batch, 500, 128) → avg → (batch, 128) ─┐
    vision (batch, 500, 128) → avg → (batch, 128) ─┼─► concat (batch, 384) → score
    text   (batch, 128, 128) → avg → (batch, 128) ─┘
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
       
        d_model = self.cfg['d_model']
        dropout = self.cfg['dropout']

        self.regressor = nn.Sequential(
            nn.Linear(d_model * 3, d_model),   # 384 → 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),  # 128 → 64
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)         # 64  → 1 score
        )

    def masked_mean(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Average over time, ignoring padding positions."""
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).float().unsqueeze(-1)          # (batch, T, 1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def forward(self, audio, vision, text, text_mask=None) -> torch.Tensor:
        a = self.masked_mean(audio,  mask=None)       # (batch, d_model)
        v = self.masked_mean(vision, mask=None)       # (batch, d_model)
        t = self.masked_mean(text,   mask=text_mask)  # (batch, d_model)

        fused = torch.cat([a, v, t], dim=-1)          # (batch, d_model*3)
        return self.regressor(fused).squeeze(-1)       # (batch,)


class TransformerFusionModel(nn.Module):
    """
    Full multimodal sentiment model.

    Stage 1 — Encode:  each modality reads itself over time
    Stage 2 — Fuse:    each modality attends to the other two
    Stage 3 — Predict: average sequences → one sentiment score
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Text backbone — frozen (we don't retrain DistilBERT)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # for param in self.distilbert.parameters():
        #     param.requires_grad = False

        for name, param in self.distilbert.named_parameters():
            if 'transformer.layer.4' in name or 'transformer.layer.5' in name:
                param.requires_grad = True    # unfreeze last 2 layers
            else:
                param.requires_grad = False   # freeze everything else

        # Stage 1: per-modality encoders
        self.audio_encoder  = ModalityEncoder(self.cfg['audio_dim'], self.cfg)
        self.vision_encoder = ModalityEncoder(self.cfg['vision_dim'], self.cfg)
        self.text_encoder   = ModalityEncoder(self.cfg['text_dim'], self.cfg)

        # Stage 2: cross-modal fusion
        self.fusion = CrossModalFusion(self.cfg)

        # Stage 3: regression head
        self.regressor = SentimentRegressor(self.cfg)


    def forward(self, input_ids, attention_mask,
                audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:

        # ── Get text token embeddings from DistilBERT ────────
        with torch.no_grad():
            bert_out = self.distilbert(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )
        text_raw  = bert_out.last_hidden_state  # (batch, 128, 768)
        text_mask = (attention_mask == 0)        # True = padding token

        # ── Stage 1: encode each modality ────────────────────
        a = self.audio_encoder (audio,    padding_mask=None)       # (batch, 500, 128)
        v = self.vision_encoder(vision,   padding_mask=None)       # (batch, 500, 128)
        t = self.text_encoder  (text_raw, padding_mask=text_mask)  # (batch, 128, 128)

        # ── Stage 2: cross-modal fusion ───────────────────────
        a, v, t = self.fusion(a, v, t, text_mask=text_mask)

        # ── Stage 3: average + predict ────────────────────────
        return self.regressor(a, v, t, text_mask=text_mask)        # (batch,)