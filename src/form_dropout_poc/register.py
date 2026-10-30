"""Fiducial registration, QA metrics, optional template shift refine."""

from __future__ import annotations

import cv2
import numpy as np

from form_dropout_poc.form_spec import FieldSpec, FontId, FormSpec, InkMode
from form_dropout_poc.render import render_blank_form
from form_dropout_poc.scan_sim import SCAN_PAD


def _geometry_fiducial_template(size: int = 48) -> np.ndarray:
    """Font-independent hollow square for corner matching."""
    tmpl = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(tmpl, (0, 0), (size - 1, size - 1), 255, 4)
    cv2.rectangle(tmpl, (12, 12), (size - 13, size - 13), 255, -1)
    return tmpl


def _find_fiducials_geometry(scan_binary: np.ndarray, spec: FormSpec) -> np.ndarray:
    ink = 255 - scan_binary
    tmpl = _geometry_fiducial_template()
    h, w = ink.shape
    regions = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
    ]
    found: list[tuple[float, float]] = []
    search_r = 180
    for idx, (rx0, ry0, rx1, ry1) in enumerate(regions):
        if idx >= len(spec.fiducials):
            break
        fid = spec.fiducials[idx]
        expected_x = fid.x + fid.size / 2 + SCAN_PAD
        expected_y = fid.y + fid.size / 2 + SCAN_PAD
        x0 = max(rx0, int(expected_x - search_r))
        y0 = max(ry0, int(expected_y - search_r))
        x1 = min(rx1, int(expected_x + search_r))
        y1 = min(ry1, int(expected_y + search_r))
        roi = ink[y0:y1, x0:x1]
        if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
            found.append((expected_x, expected_y))
            continue
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.25:
            found.append((expected_x, expected_y))
            continue
        mx, my = max_loc
        cx = x0 + mx + tmpl.shape[1] / 2
        cy = y0 + my + tmpl.shape[0] / 2
        found.append((cx, cy))
    while len(found) < 3:
        found.append((SCAN_PAD + 20.0, SCAN_PAD + 20.0))
    return np.float32(found[:3])


def template_fiducial_points(spec: FormSpec, pad: int = SCAN_PAD) -> np.ndarray:
    pts = []
    for fid in spec.fiducials:
        pts.append([fid.x + fid.size / 2 + pad, fid.y + fid.size / 2 + pad])
    return np.float32(pts)


def estimate_affine_scan_to_template(
    scan_binary: np.ndarray,
    spec: FormSpec,
) -> np.ndarray:
    """Map scan coordinates -> template coordinates (unpadded template grid)."""
    scan_pts = _find_fiducials_geometry(scan_binary, spec)
    tmpl_pts = template_fiducial_points(spec, SCAN_PAD)
    M, _ = cv2.estimateAffinePartial2D(scan_pts, tmpl_pts, method=cv2.RANSAC)
    if M is None:
        return np.float32([[1, 0, -SCAN_PAD], [0, 1, -SCAN_PAD]])
    M_to_template = M.copy()
    M_to_template[0, 2] -= SCAN_PAD
    M_to_template[1, 2] -= SCAN_PAD
    return M_to_template


def warp_scan_with_affine(
    scan_binary: np.ndarray,
    M_scan_to_template: np.ndarray,
    spec: FormSpec,
) -> np.ndarray:
    M_padded = M_scan_to_template.copy()
    M_padded[0, 2] += SCAN_PAD
    M_padded[1, 2] += SCAN_PAD
    warped = cv2.warpAffine(
        scan_binary,
        M_padded,
        (spec.page_w + 2 * SCAN_PAD, spec.page_h + 2 * SCAN_PAD),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped[SCAN_PAD : SCAN_PAD + spec.page_h, SCAN_PAD : SCAN_PAD + spec.page_w]


def registration_qa(
    scan_binary: np.ndarray,
    M_scan_to_template: np.ndarray,
    spec: FormSpec,
) -> dict[str, float]:
    scan_pts = _find_fiducials_geometry(scan_binary, spec)
    tmpl_pts = template_fiducial_points(spec, SCAN_PAD)
    reproj = []
    for i in range(min(3, len(scan_pts))):
        p = np.array([scan_pts[i, 0], scan_pts[i, 1], 1.0])
        pred = M_scan_to_template @ p
        tgt = tmpl_pts[i] - np.array([SCAN_PAD, SCAN_PAD])
        reproj.append(float(np.linalg.norm(pred - tgt)))
    return {
        "fiducial_reproj_mean_px": float(np.mean(reproj)) if reproj else 999.0,
        "fiducial_reproj_max_px": float(np.max(reproj)) if reproj else 999.0,
    }


def refine_template_shift(
    aligned_scan: np.ndarray,
    form_mask: np.ndarray,
    spec: FormSpec,
    max_shift: int = 3,
) -> tuple[int, int]:
    """
    Search small (dx,dy) to minimize XOR on non-field form ink.
    Returns shift to apply to form_mask (crop same shift on form crops).
    """
    scan_ink = (255 - aligned_scan) > 127
    form = np.array(form_mask) > 127
    field_mask = np.zeros_like(form, dtype=bool)
    for fld in spec.fields:
        field_mask[fld.y : fld.y + fld.h, fld.x : fld.x + fld.w] = True
    eval_mask = form & ~field_mask
    if eval_mask.sum() < 100:
        return 0, 0
    best = (0, 0)
    best_score = float("inf")
    h, w = form.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.zeros_like(form)
            y0 = max(0, dy)
            y1 = min(h, h + dy)
            x0 = max(0, dx)
            x1 = min(w, w + dx)
            sy0 = max(0, -dy)
            sx0 = max(0, -dx)
            shifted[y0:y1, x0:x1] = form[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
            xor = np.logical_xor(scan_ink, shifted) & eval_mask
            score = float(xor.sum())
            if score < best_score:
                best_score = score
                best = (dx, dy)
    return best


def shift_form_mask(form_mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    form = (np.array(form_mask) > 127).astype(np.uint8) * 255
    h, w = form.shape
    out = np.zeros_like(form)
    y0 = max(0, dy)
    y1 = min(h, h + dy)
    x0 = max(0, dx)
    x1 = min(w, w + dx)
    sy0 = max(0, -dy)
    sx0 = max(0, -dx)
    out[y0:y1, x0:x1] = form[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    return out


def form_ink_xor_score(aligned_scan: np.ndarray, form_mask: np.ndarray, spec: FormSpec) -> float:
    scan_ink = (255 - aligned_scan) > 127
    form = np.array(form_mask) > 127
    field_mask = np.zeros_like(form, dtype=bool)
    for fld in spec.fields:
        field_mask[fld.y : fld.y + fld.h, fld.x : fld.x + fld.w] = True
    eval_mask = form & ~field_mask
    if eval_mask.sum() == 0:
        return 0.0
    xor = np.logical_xor(scan_ink, form) & eval_mask
    return float(xor.sum()) / float(eval_mask.sum())


def crop_field(
    aligned_scan: np.ndarray,
    form_mask: np.ndarray,
    field: FieldSpec,
    overscan_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    ox = int(field.w * overscan_ratio)
    oy = int(field.h * overscan_ratio)
    x0 = max(0, field.x - ox)
    y0 = max(0, field.y - oy)
    x1 = min(aligned_scan.shape[1], field.x + field.w + ox)
    y1 = min(aligned_scan.shape[0], field.y + field.h + oy)
    scan_ink = 255 - aligned_scan
    if hasattr(form_mask, "size") and not isinstance(form_mask, np.ndarray):
        form_mask = np.array(form_mask)
    form_ink = (form_mask > 127).astype(np.uint8) * 255
    return (
        scan_ink[y0:y1, x0:x1],
        form_ink[y0:y1, x0:x1],
        (x0, y0, x1, y1),
    )


def render_form_mask_for_font(spec: FormSpec, font_id: FontId, ink_mode: InkMode) -> np.ndarray:
    _, mask = render_blank_form(spec, font_id, ink_mode)
    return np.array(mask)
