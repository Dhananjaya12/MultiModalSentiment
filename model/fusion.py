import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    One directional cross-attention stream.

    query modality  ASKS:    "what's relevant from you?"
    kv    modality  ANSWERS: provides keys and values

    The query sequence gets enriched with info from kv.
    The two sequences can have completely different lengths.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        d_model    = self.cfg['d_model']
        nhead      = self.cfg['n_heads']
        num_layers = self.cfg['fuse_layers']
        dropout    = self.cfg['dropout']

        decoder_layer = nn.TransformerDecoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= d_model * 4,
            dropout        = dropout,
            batch_first    = True
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                kv_padding_mask=None) -> torch.Tensor:
        # query:     (batch, T_q,  d_model)  ← this modality gets enriched
        # key_value: (batch, T_kv, d_model)  ← this modality provides context
        return self.cross_attn(
            tgt    = query,
            memory = key_value,
            memory_key_padding_mask = kv_padding_mask
        )  # (batch, T_q, d_model)


class CrossModalFusion(nn.Module):
    """
    Full 6-stream cross-modal fusion.

    Every modality attends to every other modality:
      audio  ← vision    audio  ← text
      vision ← audio     vision ← text
      text   ← audio     text   ← vision

    After this, each modality has absorbed information
    from the other two.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = self.cfg['d_model']

        # 6 directional streams
        self.audio_from_vision  = CrossModalAttention(self.cfg)
        self.audio_from_text    = CrossModalAttention(self.cfg)
        self.vision_from_audio  = CrossModalAttention(self.cfg)
        self.vision_from_text   = CrossModalAttention(self.cfg)
        self.text_from_audio    = CrossModalAttention(self.cfg)
        self.text_from_vision   = CrossModalAttention(self.cfg)

        # After attending to 2 modalities, merge results: d_model*2 → d_model
        self.audio_combine  = nn.Linear(d_model * 2, d_model)
        self.vision_combine = nn.Linear(d_model * 2, d_model)
        self.text_combine   = nn.Linear(d_model * 2, d_model)

    def forward(self, audio: torch.Tensor, vision: torch.Tensor,
                text: torch.Tensor, text_mask=None):
        # Audio enriched by vision AND text
        a = torch.cat([
            self.audio_from_vision(audio,  vision),
            self.audio_from_text  (audio,  text,   text_mask)
        ], dim=-1)
        a = self.audio_combine(a)    # (batch, 500, d_model)

        # Vision enriched by audio AND text
        v = torch.cat([
            self.vision_from_audio(vision, audio),
            self.vision_from_text (vision, text,   text_mask)
        ], dim=-1)
        v = self.vision_combine(v)   # (batch, 500, d_model)

        # Text enriched by audio AND vision
        t = torch.cat([
            self.text_from_audio (text,   audio),
            self.text_from_vision(text,   vision)
        ], dim=-1)
        t = self.text_combine(t)     # (batch, 128, d_model)

        return a, v, t