"""Identify which of three form fonts was used."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from form_dropout_poc.form_spec import FORM_FONT_IDS, FORM_FONT_PATHS, FontId, FormSpec, resolve_font_path


def _render_label_patch(text: str, font_id: FontId, w: int, h: int) -> np.ndarray:
    path = resolve_font_path(FORM_FONT_PATHS[font_id])
    font = ImageFont.truetype(path, 22)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), text, fill=0, font=font)
    return np.array(img)


def detect_form_font(
    aligned_scan_ink: np.ndarray,
    spec: FormSpec,
    true_font: FontId | None = None,
) -> tuple[FontId, dict[FontId, float]]:
    """
    aligned_scan_ink: 255=ink, 0=paper (template coordinates).
    """
    scores: dict[FontId, float] = {fid: 0.0 for fid in FORM_FONT_IDS}
    rois = spec.font_id_rois if spec.font_id_rois else spec.label_rois
    for roi in rois:
        patch_ink = aligned_scan_ink[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        if patch_ink.size == 0:
            continue
        patch_norm = 255 - patch_ink
        for font_id in FORM_FONT_IDS:
            tmpl = _render_label_patch(roi.text, font_id, roi.w, roi.h)
            if patch_norm.shape != tmpl.shape:
                tmpl = cv2.resize(tmpl, (patch_norm.shape[1], patch_norm.shape[0]))
            res = cv2.matchTemplate(patch_norm, tmpl, cv2.TM_CCOEFF_NORMED)
            scores[font_id] += float(res.max()) if res.size else 0.0

    best = max(FORM_FONT_IDS, key=lambda k: scores[k])
    return best, scores
