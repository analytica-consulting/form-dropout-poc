"""Batch proof-of-concept pipeline — uplift over naive OCR (no form removal)."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from form_dropout_poc.dropout import edge_sensitive_dropout, naive_subtract, prepare_for_ocr
from form_dropout_poc.evaluate import FieldResult, SampleResult, aggregate_results
from form_dropout_poc.font_id import detect_form_font
from form_dropout_poc.form_spec import FORM_FONT_IDS, build_form_spec, export_form_spec_json
from form_dropout_poc.ocr import get_default_ocr
from form_dropout_poc.register import (
    crop_field,
    estimate_affine_scan_to_template,
    form_ink_xor_score,
    refine_template_shift,
    registration_qa,
    render_form_mask_for_font,
    shift_form_mask,
    warp_scan_with_affine,
)
from form_dropout_poc.render import composite_filled_form, is_fragile_field_value, mean_overlap
from form_dropout_poc.scan_sim import simulate_scan


def process_one_sample(
    sample_id: str,
    rng: random.Random,
    spec,
    artifacts_dir: Path,
    ocr,
    min_overlap: float,
) -> SampleResult | None:
    font_id = rng.choice(FORM_FONT_IDS)
    ink_mode = rng.choice(["non_dropout", "dropout_bw"])

    filled, _form_mask_gt, _fill_mask, gt = composite_filled_form(
        spec, font_id, ink_mode, rng, min_overlap=min_overlap
    )
    if mean_overlap(gt) < min_overlap * 0.85:
        return None

    scan_binary, scan_params = simulate_scan(filled, ink_mode, rng)
    M_fid = estimate_affine_scan_to_template(scan_binary, spec)
    aligned = warp_scan_with_affine(scan_binary, M_fid, spec)
    aligned_ink = 255 - aligned
    qa = registration_qa(scan_binary, M_fid, spec)

    detected_font, font_scores = detect_form_font(aligned_ink, spec)
    form_mask = render_form_mask_for_font(spec, detected_font, ink_mode)
    dx, dy = refine_template_shift(aligned, form_mask, spec)
    form_mask = shift_form_mask(form_mask, dx, dy)
    xor_score = form_ink_xor_score(aligned, form_mask, spec)

    sample_dir = artifacts_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sample_dir / "scan_binary.png"), scan_binary)
    cv2.imwrite(str(sample_dir / "aligned_ink.png"), aligned_ink)
    overlay = np.clip(
        aligned_ink.astype(np.int16) + (form_mask > 127).astype(np.int16) * 128,
        0,
        255,
    ).astype(np.uint8)
    cv2.imwrite(str(sample_dir / "overlay_registration.png"), overlay)

    meta = {
        "ground_truth": {
            "font_id": gt.font_id,
            "ink_mode": gt.ink_mode,
            "field_values": gt.field_values,
            "overlap_fractions": gt.overlap_fractions,
            "fill_offsets": {k: list(v) for k, v in gt.fill_offsets.items()},
        },
        "scan_params": asdict(scan_params),
        "registration": {
            "method": "fiducial_geometry",
            "M_scan_to_template": M_fid.tolist(),
            "template_shift": [dx, dy],
            "form_xor_score": xor_score,
            **qa,
        },
        "font_detection": {
            "detected": detected_font,
            "true": font_id,
            "scores": {k: float(v) for k, v in font_scores.items()},
        },
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    field_results: list[FieldResult] = []
    for fld in spec.fields:
        scan_crop, form_crop, _bbox = crop_field(aligned, form_mask, fld)
        if scan_crop.size == 0:
            continue
        naive_img = prepare_for_ocr(scan_crop)
        sub_img = prepare_for_ocr(naive_subtract(scan_crop, form_crop))
        edge_img = prepare_for_ocr(edge_sensitive_dropout(scan_crop, form_crop))

        expected = gt.field_values[fld.id]
        field_results.append(
            FieldResult(
                field_id=fld.id,
                expected=expected,
                naive=ocr.recognize(naive_img, fld.charset),
                naive_subtract=ocr.recognize(sub_img, fld.charset),
                edge_dropout=ocr.recognize(edge_img, fld.charset),
                fragile=is_fragile_field_value(expected),
            )
        )

        fld_dir = sample_dir / "fields" / fld.id
        fld_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fld_dir / "naive.png"), naive_img)
        cv2.imwrite(str(fld_dir / "naive_subtract.png"), sub_img)
        cv2.imwrite(str(fld_dir / "edge_dropout.png"), edge_img)

    return SampleResult(
        sample_id=sample_id,
        font_id=font_id,
        ink_mode=ink_mode,
        font_detected=detected_font,
        font_correct=detected_font == font_id,
        reg_reproj_mean_px=qa["fiducial_reproj_mean_px"],
        mean_overlap=mean_overlap(gt),
        fields=field_results,
    )


def run_batch(n: int, seed: int, artifacts_dir: Path, use_tesseract: bool) -> dict:
    rng = random.Random(seed)
    spec = build_form_spec()
    export_form_spec_json(artifacts_dir.parent / "form_spec.json")
    ocr = get_default_ocr(use_tesseract=use_tesseract)

    results: list[SampleResult] = []
    attempts = 0
    max_attempts = n * 6
    min_overlap = 0.17
    while len(results) < n and attempts < max_attempts:
        attempts += 1
        sid = f"sample_{len(results):04d}"
        sr = process_one_sample(sid, rng, spec, artifacts_dir, ocr, min_overlap=min_overlap)
        if sr is not None:
            results.append(sr)

    summary = aggregate_results(results)
    summary["attempts"] = attempts
    summary["min_overlap_threshold"] = min_overlap
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Form dropout PoC — uplift over naive OCR")
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts",
    )
    parser.add_argument("--no-tesseract", action="store_true")
    args = parser.parse_args()

    args.artifacts.mkdir(parents=True, exist_ok=True)
    summary = run_batch(args.n, args.seed, args.artifacts, use_tesseract=not args.no_tesseract)
    headline = {
        k: summary[k]
        for k in summary
        if k
        in (
            "uplift_fields",
            "uplift_rate",
            "rescue_rate_on_naive_fail",
            "mean_cer_naive",
            "mean_cer_edge",
            "cer_delta_naive_minus_edge",
            "exact_naive",
            "exact_edge",
            "font_detection_accuracy",
            "mean_reg_reproj_px",
            "total_samples",
        )
        and k in summary
    }
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
