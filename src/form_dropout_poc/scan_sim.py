"""Simulate scanner: rotation, scale, translation, blur, illumination, B&W threshold."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from form_dropout_poc.form_spec import InkMode

SCAN_PAD = 60


@dataclass
class ScanParams:
    angle_deg: float
    scale: float
    tx: float
    ty: float
    blur_sigma: float
    threshold: int
    ink_mode: str
    illumination_strength: float


def simulate_scan(
    filled_rgb: Image.Image,
    ink_mode: InkMode,
    rng: random.Random,
) -> tuple[np.ndarray, ScanParams]:
    """
    Returns binary scan image (uint8 0/255) and parameters.
    Antagonistic: scale drift, uneven light, dropout ink surviving B&W threshold.
    """
    angle = rng.uniform(-2.0, 2.0)
    scale = rng.uniform(0.992, 1.008)
    tx = rng.uniform(-12, 12)
    ty = rng.uniform(-12, 12)
    blur_sigma = rng.uniform(0.5, 1.4)
    illum = rng.uniform(0.08, 0.22)

    rgb = np.array(filled_rgb)
    h, w = rgb.shape[:2]
    pad = SCAN_PAD
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 255, dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = rgb

    center = (canvas.shape[1] / 2, canvas.shape[0] / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    warped = cv2.warpAffine(
        canvas,
        m,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    # Uneven illumination (horizontal gradient + mild vignette)
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    grad = 1.0 - illum * (xx / max(1, gray.shape[1]) - 0.5)
    vignette = 1.0 - illum * 0.5 * ((yy / gray.shape[0] - 0.5) ** 2 + (xx / gray.shape[1] - 0.5) ** 2)
    gray = np.clip(gray * grad * vignette, 0, 255).astype(np.uint8)

    if ink_mode == "dropout_bw":
        # Mis-scanned dropout: red form channel often survives as mid-gray → black
        red = warped[:, :, 0].astype(np.float32)
        if blur_sigma > 0:
            red = cv2.GaussianBlur(red, (0, 0), blur_sigma)
        red = np.clip(red * grad * vignette, 0, 255).astype(np.uint8)
        combo = np.minimum(gray, red)
        # Biased toward thresholds that keep dropout residue
        threshold = rng.randint(125, 175)
        _, binary = cv2.threshold(combo, threshold, 255, cv2.THRESH_BINARY)
    else:
        threshold = rng.randint(105, 150)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if rng.random() < 0.55:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(0, 256, size=gray.shape, dtype=np.uint8)
        speck = noise < 5
        binary[speck] = 0
    if rng.random() < 0.35:
        ink = 255 - binary
        kernel = np.ones((2, 2), np.uint8)
        ink = cv2.dilate(ink, kernel, iterations=1)
        binary = 255 - ink

    params = ScanParams(
        angle_deg=angle,
        scale=scale,
        tx=tx,
        ty=ty,
        blur_sigma=blur_sigma,
        threshold=threshold,
        ink_mode=ink_mode,
        illumination_strength=illum,
    )
    return binary, params
