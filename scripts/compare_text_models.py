"""Compare two custom multimodal sentiment checkpoints on text-only examples.

This script does not use Gemini. It loads two TransformerFusionModel checkpoints,
runs text-only inference with zero audio/video features, and writes predictions +
confidence scores to CSV.

Example:
    python scripts/compare_text_models.py \
        --config config.json \
        --model-a outputs/best_model_a.pt \
        --model-b outputs/best_model_b.pt \
        --sentences scripts/sentiment_test_sentences.csv \
        --output outputs/model_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.model import TransformerFusionModel


CLASS_NAMES = ["Negative", "Neutral", "Positive"]
AUDIO_DIM = 768
VISION_DIM = 512
SEQ_LEN = 300


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["d_model"] = int(cfg.get("d_model", 128))
    cfg["enc_layers"] = int(cfg.get("enc_layers", 1))
    cfg["fuse_layers"] = int(cfg.get("fuse_layers", 1))
    cfg["text_dim"] = int(cfg.get("text_dim", 768))
    return cfg


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> TransformerFusionModel:
    model = TransformerFusionModel(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_text(model: TransformerFusionModel, tokenizer, text: str, cfg: dict, device: torch.device) -> tuple[str, float]:
    max_len = int(cfg.get("max_text_len", 128))
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    audio = torch.zeros((1, SEQ_LEN, AUDIO_DIM), dtype=torch.float32, device=device)
    vision = torch.zeros((1, SEQ_LEN, VISION_DIM), dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, audio, vision)
        probs = F.softmax(logits, dim=1)[0]
        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())

    return CLASS_NAMES[class_idx], round(confidence * 100, 2)


def read_sentences(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"text", "expected_label"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two model checkpoints on text examples.")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--model-a", required=True, help="Path to first checkpoint")
    parser.add_argument("--model-b", required=True, help="Path to second checkpoint")
    parser.add_argument("--sentences", required=True, help="CSV with text,expected_label columns")
    parser.add_argument("--output", default="outputs/model_comparison.csv", help="Output CSV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    print("Loading model A...")
    model_a = load_model(args.model_a, cfg, device)
    print("Loading model B...")
    model_b = load_model(args.model_b, cfg, device)

    rows = read_sentences(args.sentences)
    output_rows = []

    for i, row in enumerate(rows, start=1):
        text = row["text"]
        expected = row["expected_label"]
        pred_a, conf_a = predict_text(model_a, tokenizer, text, cfg, device)
        pred_b, conf_b = predict_text(model_b, tokenizer, text, cfg, device)
        output_rows.append({
            "id": row.get("id", i),
            "text": text,
            "expected_label": expected,
            "model_a_prediction": pred_a,
            "model_a_confidence": conf_a,
            "model_a_correct": pred_a.lower() == expected.lower(),
            "model_b_prediction": pred_b,
            "model_b_confidence": conf_b,
            "model_b_correct": pred_b.lower() == expected.lower(),
        })
        print(f"{i:03d}/{len(rows)} | expected={expected} | A={pred_a} ({conf_a}%) | B={pred_b} ({conf_b}%)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    total = len(output_rows)
    acc_a = sum(r["model_a_correct"] for r in output_rows) / total
    acc_b = sum(r["model_b_correct"] for r in output_rows) / total
    print("\nSaved:", output_path)
    print(f"Model A accuracy: {acc_a:.2%}")
    print(f"Model B accuracy: {acc_b:.2%}")


if __name__ == "__main__":
    main()
