"""CMS-1500-like form geometry and font registry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

InkMode = Literal["non_dropout", "dropout_bw"]
FontId = Literal["sans", "serif", "mono"]

PAGE_W = 2550
PAGE_H = 3300
MARGIN = 80

# macOS Supplemental paths (DejaVu/Liberation fallbacks attempted at runtime)
FORM_FONT_PATHS: dict[FontId, list[str]] = {
    "sans": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "serif": [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
    "mono": [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
}

FILL_FONT_PATHS: list[str] = [
    "/System/Library/Fonts/Supplemental/Arial Narrow.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
]

FORM_FONT_IDS: tuple[FontId, ...] = ("sans", "serif", "mono")

# Fragile pairs for ground-truth strings
FRAGILE_CHARS = "BPEFOC83QO"

# Form stroke widths (pixels) — mask uses filled bands matching scan thickness
BOX_OUTLINE = 4
RULE = 3
GRID = 2
FIDUCIAL_STROKE = 4
WOBBLE_PX = 1


@dataclass
class Fiducial:
    x: int
    y: int
    size: int = 40


@dataclass
class LabelRoi:
    """Static label region for whole-page font identification."""
    id: str
    text: str
    x: int
    y: int
    w: int
    h: int


@dataclass
class FieldSpec:
    id: str
    label: str
    x: int
    y: int
    w: int
    h: int
    charset: str  # alnum | digits | date | name


@dataclass
class FormSpec:
    page_w: int = PAGE_W
    page_h: int = PAGE_H
    fiducials: list[Fiducial] = field(default_factory=list)
    label_rois: list[LabelRoi] = field(default_factory=list)
    font_id_rois: list[LabelRoi] = field(default_factory=list)
    fields: list[FieldSpec] = field(default_factory=list)
    header_title: str = "HEALTH INSURANCE CLAIM FORM (CMS-1500 STYLE)"


def build_form_spec() -> FormSpec:
    """Synthetic CMS-like layout in template pixel coordinates."""
    fiducials = [
        Fiducial(MARGIN, MARGIN),
        Fiducial(PAGE_W - MARGIN - 40, MARGIN),
        Fiducial(MARGIN, PAGE_H - MARGIN - 40),
    ]
    label_rois = [
        LabelRoi("lbl_patient", "PATIENT NAME", 100, 220, 420, 50),
        LabelRoi("lbl_insured", "INSURED ID", 100, 320, 320, 50),
        LabelRoi("lbl_dob", "DATE OF BIRTH", 550, 320, 380, 50),
        LabelRoi("lbl_diag", "DIAGNOSIS CODE", 100, 520, 450, 50),
        LabelRoi("lbl_charges", "TOTAL CHARGES", 1500, 520, 400, 50),
    ]
    font_id_rois = [
        LabelRoi("fid_hdr1", "HEALTH INSURANCE", 100, 128, 480, 38),
        LabelRoi("fid_hdr2", "CMS-1500 STYLE", 100, 168, 420, 38),
        LabelRoi("fid_carrier", "CARRIER", 1200, 128, 280, 38),
    ]
    fields = [
        FieldSpec("patient_name", "Patient name", 520, 200, 900, 70, "name"),
        FieldSpec("insured_id", "Insured ID", 420, 300, 500, 70, "alnum"),
        FieldSpec("dob", "DOB", 920, 300, 400, 70, "date"),
        FieldSpec("diag_code", "Diagnosis", 540, 500, 450, 70, "alnum"),
        FieldSpec("procedure", "Procedure", 100, 650, 600, 70, "alnum"),
        FieldSpec("amount", "Amount", 1900, 500, 400, 70, "digits"),
        FieldSpec("provider_npi", "NPI", 100, 780, 500, 70, "digits"),
        FieldSpec("policy", "Policy", 700, 780, 550, 70, "alnum"),
    ]
    return FormSpec(
        fiducials=fiducials,
        label_rois=label_rois,
        font_id_rois=font_id_rois,
        fields=fields,
    )


def export_form_spec_json(path: Path) -> None:
    spec = build_form_spec()
    data = asdict(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_font_path(candidates: list[str]) -> str:
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"No font found among: {candidates}")
