# Showcase: `sample_0045` — patient name field

Best visual + OCR story in the seed-42 n=48 batch: fill crosses the **top rule** of the name box; naive OCR reads garbage; edge dropout strips the box while keeping letter strokes at the crossing.

**Source:** `artifacts/sample_0045/` (dropout_bw scan, fiducial reg ~0.24 px reproj, font detected mono vs true sans).

## Ground truth

`ROBERT PRICE`

## OCR (Tesseract, production path crops)

| Arm | Hypothesis | CER vs GT |
|-----|------------|-----------|
| **naive** (no form removal) | `EE` | 0.83 |
| **edge_dropout** | `RUBERTPRICE` | 0.17 |
| **naive_subtract** (dumb punch) | `KUBERTPRILE Z L` | higher |

Overlap on this field: **32%** of fill ink on form mask.

## Files

| File | What |
|------|------|
| `scan_binary.png` | Full antagonistic scan |
| `overlay_registration.png` | Aligned scan + detected template overlay |
| `naive.png` | Field crop — box + baseline still present |
| `edge_dropout.png` | Same crop after edge-sensitive dropout |
| `naive_subtract.png` | Hole-punch baseline (damages ascenders at rule) |
| `insured_id_naive.png` | Same sample — label bleed (`DATE OF BIRTH`-style noise) on another field |

Reproduce: `PYTHONPATH=src python -m form_dropout_poc.pipeline --n 48 --seed 42` then open `artifacts/sample_0045/`.
