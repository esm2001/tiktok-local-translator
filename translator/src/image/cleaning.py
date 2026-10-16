"""Remove detected text with the existing OpenCV TELEA algorithm.

The persisted PNG mask is also the exclusion mask used by source validation.
This module never draws or renders text.
"""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read Unicode Windows paths as well as ordinary container paths."""
    try:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    except (OSError, cv2.error) as exc:
        raise RuntimeError(f"No se pudo abrir {path.name}") from exc
    if image is None:
        raise RuntimeError(f"No se pudo abrir {path.name}")
    return image


def write_image(path: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, parameters or [])
    if not ok:
        raise RuntimeError(f"No se pudo guardar {path.name}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise RuntimeError(f"No se pudo guardar {path.name}") from exc


def mask_padding(height: int, width: int) -> int:
    return max(5, round(min(height, width) * 0.008))


def build_text_mask(shape: tuple[int, ...], polygons: list) -> np.ndarray:
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        cv2.fillPoly(mask, [points], 255)
    pad = mask_padding(height, width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def make_clean_image(src: Path, dest: Path, polygons: list, mask_path: Path | None = None) -> dict:
    image = read_image(src)
    mask = build_text_mask(image.shape, polygons)
    pad = mask_padding(*image.shape[:2])
    cleaned = (
        cv2.inpaint(image, mask, inpaintRadius=max(4, pad // 2), flags=cv2.INPAINT_TELEA)
        if np.any(mask)
        else image.copy()
    )
    write_image(dest, cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    if mask_path is not None:
        write_image(mask_path, mask)
    return {
        "text_regions_removed": len(polygons),
        "mask_pixels": int(np.count_nonzero(mask)),
        "mask_coverage": round(float(np.count_nonzero(mask) / mask.size), 6),
    }

