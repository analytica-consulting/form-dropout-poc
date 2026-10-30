"""Pluggable OCR engines."""

from __future__ import annotations

import abc
import re
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore


class OcrEngine(abc.ABC):
    @abc.abstractmethod
    def recognize(self, ink_image: np.ndarray, charset: str) -> str:
        """ink_image: 255=ink on 0=paper."""


class StubOcr(OcrEngine):
    """Returns empty string — swap-in point for other engines."""

    def recognize(self, ink_image: np.ndarray, charset: str) -> str:
        return ""


class TesseractOcr(OcrEngine):
    def __init__(self, psm: int = 7):
        self.psm = psm

    def recognize(self, ink_image: np.ndarray, charset: str) -> str:
        if pytesseract is None:
            return ""
        if ink_image.size == 0:
            return ""
        # Tesseract expects dark text on light bg
        gray = 255 - ink_image
        if gray.ndim == 2:
            pass
        else:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        config = f"--psm {self.psm} -c tessedit_char_whitelist={self._whitelist(charset)}"
        try:
            text = pytesseract.image_to_string(gray, config=config)
            if not text.strip():
                text = pytesseract.image_to_string(gray, config=f"--psm 6 -c tessedit_char_whitelist={self._whitelist(charset)}")
        except Exception:
            return ""
        return self._normalize(text, charset)

    @staticmethod
    def _whitelist(charset: str) -> str:
        if charset == "digits":
            return "0123456789"
        if charset == "date":
            return "0123456789/"
        if charset == "name":
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    @staticmethod
    def _normalize(text: str, charset: str) -> str:
        text = text.strip().upper()
        text = re.sub(r"\s+", " ", text)
        if charset == "digits":
            return re.sub(r"[^0-9]", "", text)
        if charset == "date":
            m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", text)
            return m.group(0) if m else re.sub(r"[^0-9/]", "", text)
        if charset == "name":
            return re.sub(r"[^A-Z ]", "", text).strip()
        return re.sub(r"[^A-Z0-9]", "", text)


def get_default_ocr(use_tesseract: bool = True) -> OcrEngine:
    if use_tesseract and pytesseract is not None:
        try:
            pytesseract.get_tesseract_version()
            return TesseractOcr()
        except Exception:
            pass
    return StubOcr()
