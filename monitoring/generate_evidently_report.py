"""Generate an Evidently AI HTML report from saved prediction CSVs.

Expected input CSV columns:
    y_true, y_pred

Optional columns used when present:
    confidence, provider, transcript, text

Example:
    python monitoring/generate_evidently_report.py \
        --input test_predictions.csv \
        --output outputs/evidently_report.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "label": "y_true",
        "true_label": "y_true",
        "actual": "y_true",
        "prediction": "y_pred",
        "pred": "y_pred",
        "pred_label": "y_pred",
        "predicted_label": "y_pred",
    }
    renamed = {col: aliases[col] for col in df.columns if col in aliases}
    return df.rename(columns=renamed)


def _build_report(df: pd.DataFrame, output_path: Path) -> None:
    try:
        from evidently import Report
        from evidently.presets import ClassificationPreset

        report = Report([ClassificationPreset()])
        report.run(df, None)
        report.save_html(output_path)
        return
    except Exception:
        pass

    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset
        from evidently.pipeline.column_mapping import ColumnMapping

        column_mapping = ColumnMapping()
        column_mapping.target = "y_true"
        column_mapping.prediction = "y_pred"

        report = Report(metrics=[ClassificationPreset()])
        report.run(reference_data=None, current_data=df, column_mapping=column_mapping)
        report.save_html(str(output_path))
        return
    except Exception as exc:
        raise RuntimeError(
            "Could not generate Evidently report. Install Evidently with `pip install evidently` "
            "and ensure the CSV has y_true and y_pred columns."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently classification report.")
    parser.add_argument("--input", required=True, help="Path to predictions CSV.")
    parser.add_argument("--output", default="outputs/evidently_report.html", help="HTML output path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _normalize_columns(pd.read_csv(input_path))
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    _build_report(df, output_path)
    print(f"Evidently report saved to: {output_path}")


if __name__ == "__main__":
    main()
