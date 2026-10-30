"""Render blank CMS-like form and composite fill with enforced overlap."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from form_dropout_poc.form_spec import (
    BOX_OUTLINE,
    FIDUCIAL_STROKE,
    FILL_FONT_PATHS,
    FORM_FONT_PATHS,
    FRAGILE_CHARS,
    GRID,
    RULE,
    WOBBLE_PX,
    FontId,
    FormSpec,
    InkMode,
    resolve_font_path,
)


@dataclass
class SampleGroundTruth:
    font_id: FontId
    ink_mode: InkMode
    field_values: dict[str, str]
    overlap_fractions: dict[str, float]
    fill_offsets: dict[str, tuple[int, int]]


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wobble_y(y: int, rng: random.Random | None) -> int:
    if rng is None:
        return y
    return y + rng.randint(-WOBBLE_PX, WOBBLE_PX)


def render_blank_form(
    spec: FormSpec,
    font_id: FontId,
    ink_mode: InkMode,
    rng: random.Random | None = None,
) -> tuple[Image.Image, Image.Image]:
    """
    Returns (display_rgb, form_mask_binary).
    form_mask is L mode: 255 = form ink (labels, rules, fiducials).
    """
    form_path = resolve_font_path(FORM_FONT_PATHS[font_id])
    label_font = _load_font(form_path, 22)
    small_font = _load_font(form_path, 18)

    page = Image.new("RGB", (spec.page_w, spec.page_h), (255, 255, 255))
    mask = Image.new("L", (spec.page_w, spec.page_h), 0)
    draw = ImageDraw.Draw(page)
    mask_draw = ImageDraw.Draw(mask)

    if ink_mode == "dropout_bw":
        form_color = (200, 40, 40)  # red dropout — may survive antagonistic B&W scan
    else:
        form_color = (0, 0, 0)

    ink_val = 255
    half_box = BOX_OUTLINE // 2
    half_rule = RULE // 2
    half_grid = max(1, GRID // 2)

    def thick_hline(x1: int, x2: int, y: int, half: int) -> None:
        y0 = max(0, y - half)
        y1 = min(spec.page_h, y + half + 1)
        draw.rectangle([x1, y0, x2, y1], fill=form_color)
        mask_draw.rectangle([x1, y0, x2, y1], fill=ink_val)

    def thick_vline(x: int, y1: int, y2: int, half: int) -> None:
        x0 = max(0, x - half)
        x1 = min(spec.page_w, x + half + 1)
        draw.rectangle([x0, y1, x1, y2], fill=form_color)
        mask_draw.rectangle([x0, y1, x1, y2], fill=ink_val)

    def thick_box(x1: int, y1: int, x2: int, y2: int) -> None:
        for edge_y in (y1, y2):
            thick_hline(x1, x2, edge_y, half_box)
        for edge_x in (x1, x2):
            thick_vline(edge_x, y1, y2, half_box)

    # Fiducials
    for fid in spec.fiducials:
        thick_box(fid.x, fid.y, fid.x + fid.size, fid.y + fid.size)
        inner = 8
        mask_draw.rectangle(
            [fid.x + inner, fid.y + inner, fid.x + fid.size - inner, fid.y + fid.size - inner],
            fill=ink_val,
        )
        draw.rectangle(
            [fid.x + inner, fid.y + inner, fid.x + fid.size - inner, fid.y + fid.size - inner],
            fill=form_color,
        )

    draw.text((100, 100), spec.header_title, fill=form_color, font=label_font)
    bbox = draw.textbbox((100, 100), spec.header_title, font=label_font)
    mask_draw.rectangle(bbox, fill=ink_val)

    for roi in spec.font_id_rois:
        draw.text((roi.x, roi.y), roi.text, fill=form_color, font=small_font)
        tb = draw.textbbox((roi.x, roi.y), roi.text, font=small_font)
        mask_draw.rectangle(tb, fill=ink_val)

    for roi in spec.label_rois:
        draw.text((roi.x, roi.y), roi.text, fill=form_color, font=small_font)
        tb = draw.textbbox((roi.x, roi.y), roi.text, font=small_font)
        mask_draw.rectangle(tb, fill=ink_val)

    for fld in spec.fields:
        thick_box(fld.x, fld.y, fld.x + fld.w, fld.y + fld.h)
        by = fld.y + fld.h - 12
        thick_hline(fld.x + 4, fld.x + fld.w - 4, by, half_rule)

    grid_y = 900
    grid_rng = rng or random.Random(0)
    for i in range(4):
        y = _wobble_y(grid_y + i * 55, grid_rng)
        thick_hline(100, spec.page_w - 100, y, half_grid)
    for x in (100, 600, 1200):
        thick_vline(x, grid_y, grid_y + 220, half_grid)

    return page, mask


def _generate_field_value(field_id: str, charset: str, rng: random.Random) -> str:
    fragile_pool = list(FRAGILE_CHARS)
    if charset == "digits":
        base = "".join(rng.choice("0123456789") for _ in range(8))
        if rng.random() < 0.7:
            pos = rng.randint(0, len(base) - 1)
            base = base[:pos] + rng.choice("83") + base[pos + 1 :]
        return base
    if charset == "date":
        return f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/{rng.randint(1950, 2005)}"
    if charset == "name":
        parts = [
            rng.choice(["ROBERT", "PETER", "BARBARA", "FRED", "COLE"]),
            rng.choice(["BROWN", "PRICE", "FOSTER", "OCHOA", "QUINN"]),
        ]
        return " ".join(parts)
    length = rng.randint(6, 10)
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    s = "".join(rng.choice(chars) for _ in range(length))
    if rng.random() < 0.6:
        pos = rng.randint(0, len(s) - 1)
        s = s[:pos] + rng.choice(fragile_pool) + s[pos + 1 :]
    return s


def _fill_layer_only(
    spec: FormSpec,
    field_values: dict[str, str],
    offsets: dict[str, tuple[int, int]],
) -> Image.Image:
    fill_path = resolve_font_path(FILL_FONT_PATHS)
    font = _load_font(fill_path, 28)
    layer = Image.new("L", (spec.page_w, spec.page_h), 0)
    draw = ImageDraw.Draw(layer)
    for fld in spec.fields:
        val = field_values.get(fld.id, "")
        if not val:
            continue
        ox, oy = offsets.get(fld.id, (8, 8))
        tx = fld.x + ox
        ty = fld.y + oy
        draw.text((tx, ty), val, fill=255, font=font)
    return layer


def _overlap_fraction(fill_layer: Image.Image, form_mask: Image.Image) -> float:
    f = np.array(fill_layer) > 127
    form = np.array(form_mask) > 127
    fill_count = int(f.sum())
    if fill_count == 0:
        return 0.0
    return float((f & form).sum()) / fill_count


def composite_filled_form(
    spec: FormSpec,
    font_id: FontId,
    ink_mode: InkMode,
    rng: random.Random,
    min_overlap: float = 0.17,
    max_nudge_steps: int = 50,
) -> tuple[Image.Image, Image.Image, Image.Image, SampleGroundTruth]:
    """
    Returns (rgb_page, form_mask, fill_mask, ground_truth).
    """
    page, form_mask = render_blank_form(spec, font_id, ink_mode, rng)
    field_values = {
        fld.id: _generate_field_value(fld.id, fld.charset, rng) for fld in spec.fields
    }
    offsets: dict[str, tuple[int, int]] = {}
    overlap_fractions: dict[str, float] = {}

    fill_path = resolve_font_path(FILL_FONT_PATHS)
    font = _load_font(fill_path, 28)

    for fld in spec.fields:
        val = field_values[fld.id]
        ox = rng.randint(-30, 10)
        oy = rng.randint(-24, 8)
        best_ox, best_oy = ox, oy
        best_frac = 0.0
        for _step in range(max_nudge_steps):
            layer = Image.new("L", (spec.page_w, spec.page_h), 0)
            ImageDraw.Draw(layer).text((fld.x + ox, fld.y + oy), val, fill=255, font=font)
            frac = _overlap_fraction(layer, form_mask)
            if frac > best_frac:
                best_frac = frac
                best_ox, best_oy = ox, oy
            if frac >= min_overlap:
                break
            ox -= 2
            oy -= 1
        offsets[fld.id] = (best_ox, best_oy)
        overlap_fractions[fld.id] = best_frac

    fill_mask = _fill_layer_only(spec, field_values, offsets)
    fill_arr = np.array(fill_mask) > 127
    page_arr = np.array(page)
    page_arr[fill_arr] = [0, 0, 0]
    filled = Image.fromarray(page_arr)

    gt = SampleGroundTruth(
        font_id=font_id,
        ink_mode=ink_mode,
        field_values=field_values,
        overlap_fractions=overlap_fractions,
        fill_offsets=offsets,
    )
    return filled, form_mask, fill_mask, gt


def mean_overlap(gt: SampleGroundTruth) -> float:
    if not gt.overlap_fractions:
        return 0.0
    return sum(gt.overlap_fractions.values()) / len(gt.overlap_fractions)


def is_fragile_field_value(value: str) -> bool:
    return any(c in FRAGILE_CHARS for c in value)
