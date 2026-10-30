# Form dropout PoC (synthetic CMS-1500-style claims)

Measures **OCR uplift** from edge-sensitive form dropout vs the real baseline: **field OCR with no form line or label removal** (naive crop only).

**Executive summary:** [`SUMMARY.md`](SUMMARY.md) (batch metrics + showcase example with figures).

Production-like path only: **fiducial registration**, **detected form font** for template `F`, antagonistic scan (scale, illumination, mis-scanned dropout B&W, speckle). No oracle warp or ground-truth font for scoring.

## Setup

```bash
cd form_dropout_poc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract
```

## Run

```bash
PYTHONPATH=src python -m form_dropout_poc.pipeline --n 48 --seed 42
```

## Headline metrics (`summary.json`)

| Metric | Meaning |
|--------|---------|
| `uplift_fields` | Fields where naive ≠ truth and edge_dropout = truth |
| `rescue_rate_on_naive_fail` | Share of naive-wrong fields fixed by edge |
| `mean_cer_naive` / `mean_cer_edge` | Character error rate |
| `cer_delta_naive_minus_edge` | Positive ⇒ edge helps |
| `by_ink_mode` | Stratified naive vs edge |
| `by_overlap_bucket` / `by_reg_error_quartile` | Diagnostic stratification |
| `fragile` | Subset with B/P/E/F/O/C/8/3 in ground truth |

OCR arms per field:

- **naive** — overscan crop, form ink **not** removed
- **edge_dropout** — registered template + extra-mass protection
- **naive_subtract** — dumb hole-punch (secondary baseline in artifacts)

## Layout

Thicker form strokes (`BOX_OUTLINE`, `RULE`, `GRID` in `form_spec.py`). See `artifacts/*/overlay_registration.png` for scan vs template alignment.
