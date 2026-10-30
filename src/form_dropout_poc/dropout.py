"""Form dropout: naive subtract vs edge-sensitive extra-mass protection."""

from __future__ import annotations

import cv2
import numpy as np

from form_dropout_poc.form_spec import RULE


def _vertical_run_length(ink: np.ndarray, x: int, y: int) -> int:
    h = ink.shape[0]
    if ink[y, x] <= 127:
        return 0
    y0 = y
    while y0 > 0 and ink[y0 - 1, x] > 127:
        y0 -= 1
    y1 = y
    while y1 + 1 < h and ink[y1 + 1, x] > 127:
        y1 += 1
    return y1 - y0 + 1


def _horizontal_run_length(ink: np.ndarray, x: int, y: int) -> int:
    w = ink.shape[1]
    if ink[y, x] <= 127:
        return 0
    x0 = x
    while x0 > 0 and ink[y, x0 - 1] > 127:
        x0 -= 1
    x1 = x
    while x1 + 1 < w and ink[y, x1 + 1] > 127:
        x1 += 1
    return x1 - x0 + 1


def dropout_tolerances() -> tuple[int, int]:
    extra_tol = max(2, RULE // 2)
    protect_radius = max(3, RULE)
    return extra_tol, protect_radius


def naive_subtract(scan_ink: np.ndarray, form_ink: np.ndarray) -> np.ndarray:
    out = scan_ink.copy()
    out[form_ink > 127] = 0
    return out


def _protect_from_thickness_profile(
    scan_ink: np.ndarray,
    form_ink: np.ndarray,
    protect: np.ndarray,
) -> None:
    """Along each row, widen protect where scan run length exceeds form on form pixels."""
    h, w = scan_ink.shape
    form = form_ink > 127
    extra_tol, protect_radius = dropout_tolerances()
    for y in range(h):
        row_form = np.where(form[y])[0]
        if row_form.size == 0:
            continue
        x0, x1 = int(row_form[0]), int(row_form[-1])
        for x in range(x0, x1 + 1):
            if not form[y, x]:
                continue
            ht_s = _horizontal_run_length(scan_ink, x, y)
            ht_f = _horizontal_run_length(form_ink, x, y)
            if ht_s > ht_f + extra_tol:
                x_left = max(0, x - protect_radius)
                x_right = min(w, x + protect_radius + 1)
                protect[y, x_left:x_right] = True


def edge_sensitive_dropout(
    scan_ink: np.ndarray,
    form_ink: np.ndarray,
    extra_tol: int | None = None,
    protect_radius: int | None = None,
) -> np.ndarray:
    if extra_tol is None or protect_radius is None:
        extra_tol, protect_radius = dropout_tolerances()
    h, w = scan_ink.shape
    protect = np.zeros((h, w), dtype=bool)
    form = form_ink > 127

    for y in range(h):
        xs = np.where(form[y])[0]
        if xs.size == 0:
            continue
        for x in xs:
            vt_s = _vertical_run_length(scan_ink, x, y)
            vt_f = _vertical_run_length(form_ink, x, y)
            ht_s = _horizontal_run_length(scan_ink, x, y)
            ht_f = _horizontal_run_length(form_ink, x, y)
            if vt_s > vt_f + extra_tol or ht_s > ht_f + extra_tol:
                y0 = max(0, y - protect_radius)
                y1 = min(h, y + protect_radius + 1)
                x0 = max(0, x - protect_radius)
                x1 = min(w, x + protect_radius + 1)
                protect[y0:y1, x0:x1] = True

    _protect_from_thickness_profile(scan_ink, form_ink, protect)

    out = scan_ink.copy()
    remove = form & ~protect
    out[remove] = 0
    return out


def prepare_for_ocr(ink_image: np.ndarray, min_h: int = 64) -> np.ndarray:
    if ink_image.size == 0:
        return ink_image
    h, w = ink_image.shape[:2]
    if h < min_h:
        scale = min_h / float(h)
        ink_image = cv2.resize(
            ink_image,
            (max(1, int(w * scale)), min_h),
            interpolation=cv2.INTER_LINEAR,
        )
    border = 16
    ink_image = cv2.copyMakeBorder(
        ink_image, border, border, border, border, cv2.BORDER_CONSTANT, value=0
    )
    return ink_image
